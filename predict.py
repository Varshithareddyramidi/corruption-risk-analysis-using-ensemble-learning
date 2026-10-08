import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.ft_transformer import TabularFTTransformer

def load_models():
    """Loads the trained LightGBM, FT-Transformer, scaler, and label encoder."""
    print("Loading models and transformers...")
    lgb_model = joblib.load('models/lgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_enc = joblib.load('models/label_enc.pkl')
    
    num_classes = len(label_enc.classes_)
    num_features = len(scaler.scale_)
    
    ft_model = TabularFTTransformer(num_features=num_features, num_classes=num_classes)
    ft_model.load_state_dict(torch.load('models/ft_model.pth'))
    ft_model.eval()
    
    return lgb_model, ft_model, scaler, label_enc

def predict(df_new, lgb_model, ft_model, scaler, label_enc):
    """Makes a prediction using the soft-voting ensemble."""
    
    # Define expected columns based on training
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
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0.0
            
    X = df_new[feature_cols].copy().fillna(0.0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # 1. LightGBM Predictions
    lgb_probs = lgb_model.predict_proba(X_scaled)
    num_classes = len(label_enc.classes_)
    
    full_probs_lgb = np.zeros((len(X_scaled), num_classes))
    for i, cls_idx in enumerate(lgb_model.classes_):
        full_probs_lgb[:, cls_idx] = lgb_probs[:, i]
        
    # 2. FT-Transformer Predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = ft_model(X_tensor)
        ft_probs = F.softmax(logits, dim=1).numpy()
        
    if ft_probs.shape[1] > num_classes:
        ft_probs = ft_probs[:, :num_classes]
        
    # 3. Ensemble Voting
    ensemble_probs = 0.5 * full_probs_lgb + 0.5 * ft_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # Convert numerical predictions back to string labels (LOW, MEDIUM, HIGH)
    predicted_labels = label_enc.inverse_transform(ensemble_preds)
    
    return predicted_labels, ensemble_probs

if __name__ == '__main__':
    # Load Models
    lgb_model, ft_model, scaler, label_enc = load_models()
    
    # Sample new data to predict
    sample_data = pd.DataFrame([{
        'party_count': 1,
        'cpv_count': 3,
        'has_participation_fee': 1,
        'tender_value_amount_num': 50000000.0,
        'tender_value_log_zscore': 2.5,  # High value outlier
        'flag_short_description': 0,
        'text_risk_keyword_score': 5.0,  # High risk keywords
        'flag_brand_specs': 1,           # Brand names used
        'complaint_filed_flag': 1,       # Has complaints
        'complaint_severity_score': 3.0, # High severity
        'text_word_count': 15,
        'flag_high_value_outlier': 1
    }])
    
    print("\nPredicting on sample data:")
    print(sample_data.T)
    
    labels, probs = predict(sample_data, lgb_model, ft_model, scaler, label_enc)
    
    print("\nResult:")
    print(f"Predicted Corruption Risk: {labels[0]}")
    for idx, cls_name in enumerate(label_enc.classes_):
        print(f"Probability ({cls_name}): {probs[0][idx]*100:.2f}%")
