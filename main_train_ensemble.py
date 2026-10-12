"""
Ensemble Learning Execution Pipeline for Corruption Risk Analysis.
Combines LightGBM (via GradientBoosting / HistGradientBoosting) and FT-Transformer (PyTorch)
with a soft-voting ensemble for public procurement risk assessment.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import json
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score

# Add current folder to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ft_transformer import TabularFTTransformer, ProcurementDataset, train_ft_transformer


def prepare_features(df: pd.DataFrame, is_train: bool = True, scaler: StandardScaler = None, label_encoder: LabelEncoder = None):
    """
    Prepares numerical risk features and target labels for tabular ensemble modeling.
    """
    feature_cols = [
        'party_count',
        'cpv_count',
        'has_participation_fee',
        'tender_value_amount_num',
        'tender_value_log_zscore',
        'flag_short_description',
        'text_risk_keyword_score',
        'flag_brand_specs',
        'complaint_filed_flag',
        'complaint_severity_score',
        'text_word_count',
        'flag_high_value_outlier'
    ]
    
    X = df[feature_cols].copy()
    X = X.fillna(0.0)
    
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        label_encoder = LabelEncoder()
        # Persist LabelEncoder's sorted three-class order with all model artifacts.
        y = label_encoder.fit_transform(df['risk_level'])
    else:
        X_scaled = scaler.transform(X)
        y = label_encoder.transform(df['risk_level'])
        
    return X_scaled, y, feature_cols, scaler, label_encoder


def evaluate_ensemble(train_path: str, val_path: str, test_path: str, output_dir: str = 'models'):
    """
    Trains Gradient Boosting (LightGBM-style) & FT-Transformer and evaluates Ensemble soft voting predictions.
    """
    print("Step 1: Loading preprocessed train, validation, and test datasets...")
    np.random.seed(42)
    torch.manual_seed(42)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    for frame in (train_df, val_df, test_df):
        frame['risk_level'] = frame['risk_level'].astype(str).str.strip().str.upper()
    expected_classes = {'LOW', 'MEDIUM', 'HIGH'}
    missing_train_classes = expected_classes - set(train_df['risk_level'].unique())
    if missing_train_classes:
        raise ValueError(f"Training split is missing risk classes: {sorted(missing_train_classes)}")
    for name, frame in [('validation', val_df), ('test', test_df)]:
        missing = expected_classes - set(frame['risk_level'].unique())
        if missing:
            raise ValueError(f"{name.title()} split is missing risk classes: {sorted(missing)}")
    
    print(f"-> Train records: {len(train_df)} | Val records: {len(val_df)} | Test records: {len(test_df)}")
    
    # 1. Feature Preprocessing
    X_train, y_train, feat_cols, scaler, label_enc = prepare_features(train_df, is_train=True)
    X_val, y_val, _, _, _ = prepare_features(val_df, is_train=False, scaler=scaler, label_encoder=label_enc)
    X_test, y_test, _, _, _ = prepare_features(test_df, is_train=False, scaler=scaler, label_encoder=label_enc)
    
    # 2. Train Official LightGBM Model
    print("\nStep 2: Training Official LightGBM Classifier...")
    lgb_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    
    val_preds_lgb = lgb_model.predict(X_val)
    val_probs_lgb = lgb_model.predict_proba(X_val)
    print(f"-> LightGBM Validation Accuracy: {accuracy_score(y_val, val_preds_lgb):.4f} | Macro-F1: {f1_score(y_val, val_preds_lgb, average='macro'):.4f}")
    
    # 3. Train FT-Transformer Neural Model
    print("\nStep 3: Training PyTorch FT-Transformer (Feature Tokenizer Transformer)...")
    num_classes = len(label_enc.classes_)
    ft_model = train_ft_transformer(X_train, y_train, X_val, y_val, num_classes=num_classes, epochs=15, batch_size=32)
    
    ft_model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        logits_val = ft_model(X_val_tensor)
        val_probs_ft = F.softmax(logits_val, dim=1).numpy()
        val_preds_ft = np.argmax(val_probs_ft, axis=1)
        
    print(f"-> FT-Transformer Validation Accuracy: {accuracy_score(y_val, val_preds_ft):.4f} | Macro-F1: {f1_score(y_val, val_preds_ft, average='macro'):.4f}")
    
    # 4. Soft-Voting Ensemble Combination (LightGBM + FT-Transformer)
    print("\nStep 4: Evaluating Soft-Voting Ensemble (LightGBM + FT-Transformer) on Test Set...")
    test_probs_lgb = lgb_model.predict_proba(X_test)
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits_test = ft_model(X_test_tensor)
        test_probs_ft = F.softmax(logits_test, dim=1).numpy()
        
    # Align probability dimensions across all target classes present in LabelEncoder
    full_probs_lgb = np.zeros((len(X_test), num_classes))
    for i, cls_idx in enumerate(lgb_model.classes_):
        full_probs_lgb[:, cls_idx] = test_probs_lgb[:, i]
        
    # Ensure test_probs_ft matches num_classes
    if test_probs_ft.shape[1] > num_classes:
        test_probs_ft = test_probs_ft[:, :num_classes]
        
    # Ensemble probability soft voting (50% LightGBM + 50% FT-Transformer)
    ensemble_test_probs = 0.5 * full_probs_lgb + 0.5 * test_probs_ft
    ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)
    
    # Test Performance Metrics
    acc_lgb = accuracy_score(y_test, lgb_model.predict(X_test))
    acc_ft = accuracy_score(y_test, np.argmax(test_probs_ft, axis=1))
    acc_ensemble = accuracy_score(y_test, ensemble_test_preds)
    f1_ensemble = f1_score(y_test, ensemble_test_preds, average='weighted')
    lgb_test_preds = lgb_model.predict(X_test)
    ft_test_preds = np.argmax(test_probs_ft, axis=1)
    
    print("\n================ FINAL TEST EVALUATION RESULTS ================")
    print(f"1. LightGBM Test Accuracy              : {acc_lgb:.4f}")
    print(f"2. FT-Transformer Test Accuracy         : {acc_ft:.4f}")
    print(f"3. Soft-Voting Ensemble Test Accuracy   : {acc_ensemble:.4f}")
    print(f"4. Soft-Voting Ensemble Weighted F1-Score: {f1_ensemble:.4f}")
    print(f"5. Ensemble Balanced Accuracy            : {balanced_accuracy_score(y_test, ensemble_test_preds):.4f}")
    print(f"6. Ensemble Macro-F1                     : {f1_score(y_test, ensemble_test_preds, average='macro'):.4f}")
    print("===============================================================\n")
    
    target_names = label_enc.classes_
    print("Classification Report (LightGBM):")
    print(classification_report(y_test, lgb_test_preds, labels=np.arange(num_classes), target_names=target_names, zero_division=0))
    print("Classification Report (FT-Transformer):")
    print(classification_report(y_test, ft_test_preds, labels=np.arange(num_classes), target_names=target_names, zero_division=0))
    print("Classification Report (Soft-Voting Ensemble):")
    print(classification_report(y_test, ensemble_test_preds, labels=np.arange(num_classes), target_names=target_names, zero_division=0))
    
    print("\nStep 5: Saving Trained Models and Transformers...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lgb_model, os.path.join(output_dir, 'lgb_model.pkl'))
    torch.save(ft_model.state_dict(), os.path.join(output_dir, 'ft_model.pth'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(label_enc, os.path.join(output_dir, 'label_enc.pkl'))
    eval_metrics = {
        'test_rows': int(len(test_df)),
        'classes': [str(c) for c in label_enc.classes_],
        'test_class_counts': {str(k): int(v) for k, v in test_df['risk_level'].value_counts().items()},
        'metrics': {
            'lightgbm': {
                'accuracy': float(acc_lgb),
                'balanced_accuracy': float(balanced_accuracy_score(y_test, lgb_test_preds)),
                'macro_f1': float(f1_score(y_test, lgb_test_preds, average='macro')),
            },
            'ft_transformer': {
                'accuracy': float(acc_ft),
                'balanced_accuracy': float(balanced_accuracy_score(y_test, ft_test_preds)),
                'macro_f1': float(f1_score(y_test, ft_test_preds, average='macro')),
            },
            'soft_voting_ensemble': {
                'accuracy': float(acc_ensemble),
                'balanced_accuracy': float(balanced_accuracy_score(y_test, ensemble_test_preds)),
                'macro_f1': float(f1_score(y_test, ensemble_test_preds, average='macro')),
                'weighted_f1': float(f1_ensemble),
                'classification_report': classification_report(
                    y_test, ensemble_test_preds, labels=np.arange(num_classes),
                    target_names=target_names, output_dict=True, zero_division=0,
                ),
            },
        },
        'label_note': 'Test metrics measure agreement with rule-generated proxy categories, not confirmed corruption outcomes.',
    }
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
    print(f"-> Models and evaluation metrics saved to '{output_dir}'.")
    
    return {
        'lgb_acc': acc_lgb,
        'ft_acc': acc_ft,
        'ensemble_acc': acc_ensemble,
        'ensemble_f1': f1_ensemble
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate three-class procurement models.')
    parser.add_argument('--train', default=r'C:\Users\varsh\Downloads\data\processed\train.csv')
    parser.add_argument('--validation', default=r'C:\Users\varsh\Downloads\data\processed\val.csv')
    parser.add_argument('--test', default=r'C:\Users\varsh\Downloads\data\processed\test.csv')
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()
    train_csv, val_csv, test_csv = args.train, args.validation, args.test
    
    if not os.path.exists(train_csv):
        print(f"Dataset split files not found in {train_csv}. Running main_preprocess.py first...")
        from main_preprocess import run_pipeline
        run_pipeline(
            r'c:\Users\varsh\Downloads\albania_public_procurement_commission_2025.csv\2025',
            r'c:\Users\varsh\Downloads\data\processed',
            r'c:\Users\varsh\Downloads\reports'
        )
        
    evaluate_ensemble(train_csv, val_csv, test_csv, args.output_dir)
