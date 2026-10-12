import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
import joblib
import shap
from sentence_transformers import SentenceTransformer
import os
import sys
import json

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.ft_transformer import TabularFTTransformer

# --- Configuration & Caching ---
st.set_page_config(page_title="Procurement Corruption Risk AI", layout="wide", page_icon="⚖️")
ROOT = os.path.dirname(os.path.abspath(__file__))
SBERT_DIR = os.path.join(ROOT, 'models', 'sbert')

FEATURE_COLS = [
    'party_count', 'cpv_count', 'has_participation_fee',
    'tender_value_amount_num', 'tender_value_log_zscore',
    'flag_short_description', 'text_risk_keyword_score',
    'flag_brand_specs', 'complaint_filed_flag',
    'complaint_severity_score', 'text_word_count',
    'flag_high_value_outlier'
]

FEATURE_NAMES = {
    'party_count': 'Number of bidders',
    'cpv_count': 'Procurement category count',
    'has_participation_fee': 'Participation fee',
    'tender_value_amount_num': 'Estimated tender value',
    'tender_value_log_zscore': 'Category-relative tender value',
    'flag_short_description': 'Short description',
    'text_risk_keyword_score': 'Risk-related description terms',
    'flag_brand_specs': 'Restrictive or brand specifications',
    'complaint_filed_flag': 'Complaint filed',
    'complaint_severity_score': 'Complaint count / severity proxy',
    'text_word_count': 'Description word count',
    'flag_high_value_outlier': 'High-value outlier',
}

# Accept common procurement CSV spellings without requiring users to rename columns.
SOURCE_ALIASES = {
    'tender_description': ('description', 'contract_description', 'procurement_description',
                          'object_description', 'tender_title', 'contract_title', 'title', 'subject'),
    'number_of_bidders': ('party_count', 'bidder_count', 'bidders_count', 'number_bids', 'bid_count',
                          'no_of_bidders', 'participants_count', 'supplier_count'),
    'estimated_value': ('estimated_contract_value', 'tender_value', 'contract_value',
                        'tender_value_amount_num', 'awarded_value', 'estimated_amount', 'budget_amount', 'tender_amount'),
    'complaint_count': ('complaints_count', 'number_of_complaints', 'appeal_count',
                        'appeals_count', 'challenge_count'),
    'restrictive_specs_flag': ('flag_brand_specs', 'restrictive_specifications_flag', 'restricted_specs_flag',
                               'narrow_eligibility_flag', 'restrictive_spec_flag'),
    'contract_amendments': ('amendment_count', 'number_of_amendments', 'contract_amendment_count'),
    'contract_value_increase_pct': ('value_increase_pct', 'amendment_value_increase_pct',
                                    'contract_increase_pct'),
    'repeat_winner_flag': ('repeat_supplier_flag', 'repeat_awardee_flag', 'repeat_winner'),
    'award_justification_status': ('justification_status', 'award_reason_status',
                                   'award_justification', 'award_reason'),
    'procurement_method': ('procedure_type', 'tender_procedure', 'procurement_procedure'),
    'bid_submission_days': ('submission_period_days', 'bid_period_days', 'tender_period_days'),
}

def canonicalize_source_columns(df):
    out = df.copy()
    normalized = {''.join(ch.lower() for ch in str(c) if ch.isalnum()): c for c in out.columns}
    for canonical, aliases in SOURCE_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in (canonical, *aliases):
            source = normalized.get(''.join(ch.lower() for ch in alias if ch.isalnum()))
            if source is not None:
                out[canonical] = out[source]
                break
    return out

def get_descriptions(df):
    for col in ('text_description', 'tender_description', 'description', 'title', 'tender_title'):
        if col in df.columns:
            return df[col].fillna('').astype(str).tolist()
    return None

@st.cache_resource
def load_all_models(artifact_revision):
    """Load the existing Albania tabular models and trained text classifier."""
    metadata_path = os.path.join(SBERT_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            'Sentence-BERT classifier files are missing. Run '
            '`python train_sbert_classifier.py <processed-csv-path>` first.'
        )
    import json
    with open(metadata_path, encoding='utf-8') as f:
        sbert_metadata = json.load(f)
    # Community Cloud may start with an empty Hugging Face cache; download on first run if needed.
    sbert_model = SentenceTransformer(sbert_metadata['sbert_model'], local_files_only=False)
    sbert_classifier = joblib.load(os.path.join(SBERT_DIR, 'classifier.pkl'))
    sbert_label_enc = joblib.load(os.path.join(SBERT_DIR, 'label_encoder.pkl'))
    
    # 2. Load LightGBM, Scaler, Encoder
    lgb_model = joblib.load(os.path.join(ROOT, 'models', 'lgb_model.pkl'))
    scaler = joblib.load(os.path.join(ROOT, 'models', 'scaler.pkl'))
    label_enc = joblib.load(os.path.join(ROOT, 'models', 'label_enc.pkl'))
    
    # 3. Load FT-Transformer
    num_classes = len(label_enc.classes_)
    num_features = len(scaler.scale_)
    ft_model = TabularFTTransformer(num_features=num_features, num_classes=num_classes)
    ft_model.load_state_dict(torch.load(os.path.join(ROOT, 'models', 'ft_model.pth'), map_location=torch.device('cpu')))
    ft_model.eval()
    
    return sbert_model, sbert_classifier, sbert_label_enc, sbert_metadata, lgb_model, ft_model, scaler, label_enc

def current_artifact_revision():
    paths = [
        os.path.join(ROOT, 'models', name)
        for name in ('lgb_model.pkl', 'ft_model.pth', 'scaler.pkl', 'label_enc.pkl')
    ] + [
        os.path.join(SBERT_DIR, name)
        for name in ('classifier.pkl', 'label_encoder.pkl', 'metadata.json')
    ]
    return tuple((os.path.basename(path), os.stat(path).st_mtime_ns, os.stat(path).st_size) for path in paths)

sbert, sbert_classifier, sbert_label_enc, sbert_metadata, lgb_model, ft_model, scaler, label_enc = load_all_models(current_artifact_revision())
tabular_metrics_path = os.path.join(ROOT, 'models', 'evaluation_metrics.json')
try:
    with open(tabular_metrics_path, encoding='utf-8') as f:
        tabular_metrics = json.load(f)
except (OSError, json.JSONDecodeError):
    tabular_metrics = None

def predict_text_probabilities(texts):
    """Use the trained Sentence-BERT classifier for description-only prediction."""
    embeddings = sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    model_probs = sbert_classifier.predict_proba(embeddings)
    aligned = np.zeros((len(texts), len(label_enc.classes_)), dtype=float)
    for col, encoded_class in enumerate(sbert_classifier.classes_):
        class_name = sbert_label_enc.inverse_transform([int(encoded_class)])[0]
        if class_name in label_enc.classes_:
            aligned[:, int(np.where(label_enc.classes_ == class_name)[0][0])] = model_probs[:, col]
    return aligned

def predict_description(description):
    probabilities = predict_text_probabilities([description])[0]
    prediction = label_enc.inverse_transform([int(np.argmax(probabilities))])[0]
    return prediction, probabilities

# --- Utility Functions ---
def compute_rule_labels(df):
    """Reproduce the project's exact rule-generated risk target for tabular rows."""
    short = pd.to_numeric(df['flag_short_description'], errors='coerce').fillna(0).to_numpy()
    keywords = pd.to_numeric(df['text_risk_keyword_score'], errors='coerce').fillna(0).to_numpy()
    brand = pd.to_numeric(df['flag_brand_specs'], errors='coerce').fillna(0).to_numpy()
    complaint = pd.to_numeric(df['complaint_filed_flag'], errors='coerce').fillna(0).to_numpy()
    severity = pd.to_numeric(df['complaint_severity_score'], errors='coerce').fillna(0).to_numpy()
    value_z = pd.to_numeric(df['tender_value_log_zscore'], errors='coerce').fillna(0).to_numpy()
    scores = (
        short * 15.0
        + np.minimum(10.0, keywords) * 3.0
        + brand * 20.0
        + complaint * 25.0
        + severity * 5.0
        + np.maximum(0.0, value_z) * 5.0
    )
    scores = np.minimum(100.0, np.round(scores, 2))
    labels = np.where(scores >= 40.0, 'HIGH', np.where(scores >= 20.0, 'MEDIUM', 'LOW'))
    return labels, scores

def adapt_alternate_procurement_schema(df):
    """Map the supplied realistic-synthetic schema onto the model's training features.

    This is a compatibility adapter, not a claim that the source variables are
    identical to the Albania training data. Unavailable features are left at 0.
    """
    out = canonicalize_source_columns(df)
    if all(col in out.columns for col in FEATURE_COLS):
        return out, False
    if not {'tender_description', 'number_of_bidders', 'estimated_value'}.issubset(out.columns):
        return out, False

    descriptions = out['tender_description'].fillna('').astype(str)
    words = descriptions.str.findall(r"[A-Za-z0-9]+")
    out['party_count'] = pd.to_numeric(out['number_of_bidders'], errors='coerce').fillna(0)
    out['cpv_count'] = 1.0  # This file has no CPV code field.
    out['has_participation_fee'] = 0.0  # This file has no participation-fee field.
    out['tender_value_amount_num'] = pd.to_numeric(out['estimated_value'], errors='coerce').fillna(0)
    # There is no category-relative value baseline in this file; avoid a
    # batch-dependent z-score that would change when unrelated rows are added.
    out['tender_value_log_zscore'] = 0.0
    out['text_word_count'] = words.str.len()
    out['flag_short_description'] = (out['text_word_count'] < 5).astype(int)
    text = descriptions.str.lower()
    risk_phrases = ('limited supplier participation', 'narrow eligibility',
                    'restrictive specifications', 'urgent procurement',
                    'favorizim', 'kufizim', 'skualifikim', 'monopol', 'brend')
    out['text_risk_keyword_score'] = sum(text.str.contains(p, regex=False).astype(int) for p in risk_phrases)
    out['flag_brand_specs'] = pd.to_numeric(
        out.get('restrictive_specs_flag', pd.Series(0, index=out.index)), errors='coerce'
    ).fillna(0)
    complaints = pd.to_numeric(out.get('complaint_count', pd.Series(0, index=out.index)), errors='coerce').fillna(0)
    out['complaint_filed_flag'] = (complaints > 0).astype(int)
    out['complaint_severity_score'] = complaints.clip(upper=3)
    out['flag_high_value_outlier'] = 0.0  # No reliable category baseline is provided.
    # Additional relevant fields are retained as transparent review signals.
    out['signal_low_competition'] = (pd.to_numeric(out['number_of_bidders'], errors='coerce').fillna(0) <= 2).astype(int)
    out['signal_restricted_method'] = out.get('procurement_method', pd.Series('', index=out.index)).fillna('').astype(str).str.lower().str.contains(
        r'restrict|direct|limited|negotiat|single.source', regex=True
    ).astype(int)
    out['signal_repeat_winner'] = pd.to_numeric(out.get('repeat_winner_flag', pd.Series(0, index=out.index)), errors='coerce').fillna(0).astype(int)
    out['signal_amendments'] = (pd.to_numeric(out.get('contract_amendments', pd.Series(0, index=out.index)), errors='coerce').fillna(0) >= 2).astype(int)
    out['signal_large_value_increase'] = (pd.to_numeric(out.get('contract_value_increase_pct', pd.Series(0, index=out.index)), errors='coerce').fillna(0) >= 15).astype(int)
    justification = out.get('award_justification_status', pd.Series('', index=out.index)).fillna('').astype(str).str.lower()
    out['signal_justification_missing'] = justification.str.contains(r'missing|absent|not published|unknown', regex=True).astype(int)
    out['signal_short_submission'] = (pd.to_numeric(out.get('bid_submission_days', pd.Series(999, index=out.index)), errors='coerce').fillna(999) <= 10).astype(int)
    return out, True

def predict_risk(df):
    """Use all three models when tabular fields exist; text alone uses SBERT."""
    df, adapted = adapt_alternate_procurement_schema(df)
    descriptions = get_descriptions(df)
    if not all(col in df.columns for col in FEATURE_COLS):
        if descriptions is None:
            raise ValueError('Provide all required tabular features or a description column.')
        probs = predict_text_probabilities(descriptions)
        return label_enc.inverse_transform(np.argmax(probs, axis=1)), probs, None, None, df, adapted

    X = df[FEATURE_COLS].copy().fillna(0.0)
    X_scaled = scaler.transform(X)
    
    # LightGBM Predictions
    lgb_probs = lgb_model.predict_proba(X_scaled)
    num_classes = len(label_enc.classes_)
    full_probs_lgb = np.zeros((len(X_scaled), num_classes))
    for i, cls_idx in enumerate(lgb_model.classes_):
        full_probs_lgb[:, cls_idx] = lgb_probs[:, i]
        
    # FT-Transformer Predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = ft_model(X_tensor)
        ft_probs = F.softmax(logits, dim=1).numpy()
    
    if ft_probs.shape[1] > num_classes:
        ft_probs = ft_probs[:, :num_classes]
        
    # Include SBERT classifier when text is present. The three model weights are equal.
    if descriptions is not None:
        sbert_probs = predict_text_probabilities(descriptions)
        ensemble_probs = (full_probs_lgb + ft_probs + sbert_probs) / 3.0
    else:
        ensemble_probs = 0.5 * full_probs_lgb + 0.5 * ft_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    predicted_labels = label_enc.inverse_transform(ensemble_preds)
    
    rule_labels, rule_scores = compute_rule_labels(df)
    return predicted_labels, ensemble_probs, rule_labels, rule_scores, df, adapted

def rule_score_components(row):
    """Exact per-feature contributions to the displayed rule-based category score."""
    def value(column):
        return float(pd.to_numeric(pd.Series([row[column]]), errors='coerce').fillna(0).iloc[0])
    return {
        'Short description': value('flag_short_description') * 15,
        'Risk-related description terms': min(10.0, value('text_risk_keyword_score')) * 3,
        'Restrictive or brand specifications': value('flag_brand_specs') * 20,
        'Complaint filed': value('complaint_filed_flag') * 25,
        'Complaint count / severity proxy': value('complaint_severity_score') * 5,
        'Category-relative tender value': max(0.0, value('tender_value_log_zscore')) * 5,
    }

def class_shap_values(shap_values, class_index, row_index):
    """Normalize SHAP's legacy list and current ndarray multi-class outputs."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])[row_index]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # SHAP may return (rows, features, classes) or (rows, classes, features).
        if values.shape[1] == len(FEATURE_COLS):
            return values[row_index, :, class_index]
        return values[row_index, class_index, :]
    return values[row_index]

def get_additional_signals(row):
    signal_labels = {
        'signal_low_competition': 'Very few bidders (2 or fewer)',
        'signal_restricted_method': 'Restricted, direct, or negotiated procedure',
        'signal_repeat_winner': 'Repeat winner flag',
        'signal_amendments': 'Multiple contract amendments',
        'signal_large_value_increase': 'Contract value increase of 15% or more',
        'signal_justification_missing': 'Award justification missing or unpublished',
        'signal_short_submission': 'Short bid submission period (10 days or less)',
    }
    return [label for col, label in signal_labels.items() if col in row.index and row[col] == 1]

@st.cache_resource
def get_text_shap_explainer():
    masker = shap.maskers.Text(sbert.tokenizer)
    return shap.Explainer(predict_text_probabilities, masker, algorithm='partition')

def show_text_shap(description, class_index):
    explanation = get_text_shap_explainer()([description], max_evals=100)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[0, :, class_index]
    else:
        values = values[0]
    tokens = list(explanation.data[0])
    text_shap = pd.DataFrame({'Term': tokens, 'SHAP contribution': values})
    text_shap = text_shap[text_shap['Term'].astype(str).str.strip() != '']
    text_shap = text_shap.reindex(text_shap['SHAP contribution'].abs().sort_values(ascending=False).index)
    raise_terms = text_shap[text_shap['SHAP contribution'] > 0].head(5)
    lower_terms = text_shap[text_shap['SHAP contribution'] < 0].head(5)
    class_name = str(label_enc.classes_[class_index]).lower()
    if len(raise_terms):
        reasons = ', '.join(f'“{str(row.Term).strip()}”' for row in raise_terms.itertuples())
        st.write(f"**Why the description leans {class_name}:** The terms {reasons} pushed the model’s {class_name} score upward.")
    if len(lower_terms):
        reasons = ', '.join(f'“{str(row.Term).strip()}”' for row in lower_terms.itertuples())
        st.write(f"**What pulled the score down:** The terms {reasons} pushed the {class_name} score downward.")
    if not len(raise_terms) and not len(lower_terms):
        st.write("SHAP did not find a clear word-level contribution for this description.")
    st.caption("These are word-level influences on Sentence-BERT’s selected class score, not proof that an incident occurred.")

def feature_reason(feature, value):
    """Turn a tabular feature and its value into a readable procurement signal."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    reasons = {
        'party_count': 'Only one bidder was recorded, which can indicate limited competition.' if number == 1 else f'{number:g} bidders were recorded.',
        'cpv_count': f'{number:g} procurement category code(s) were recorded.',
        'has_participation_fee': 'A participation fee was recorded.' if number > 0 else 'No participation fee was recorded.',
        'tender_value_amount_num': f'The estimated tender value was {number:,.0f}.',
        'tender_value_log_zscore': 'The value is above its category-relative baseline.' if number > 0 else 'The value is at or below its category-relative baseline.',
        'flag_short_description': 'The description was flagged as unusually short.' if number > 0 else 'The description was not flagged as unusually short.',
        'text_risk_keyword_score': f'{number:g} risk-related description term indicator(s) were detected.',
        'flag_brand_specs': 'Restrictive or brand-specific specifications were flagged.' if number > 0 else 'No restrictive-specification flag was recorded.',
        'complaint_filed_flag': 'A complaint or challenge was recorded.' if number > 0 else 'No complaint was recorded.',
        'complaint_severity_score': f'The complaint count/severity indicator was {number:g}.',
        'text_word_count': f'The description contains {number:g} words.',
        'flag_high_value_outlier': 'The value was flagged as a high-value outlier.' if number > 0 else 'No high-value outlier flag was recorded.',
    }
    return reasons.get(feature, f'{FEATURE_NAMES.get(feature, feature)} had value {number:g}.')

def explain_shap_in_plain_language(shap_row, features_row, class_name, model_name):
    """Render row-level SHAP contributions as simple upward/downward reasons."""
    entries = []
    for feature, contribution, raw_value in zip(FEATURE_COLS, shap_row, features_row[FEATURE_COLS]):
        entries.append((feature, float(contribution), feature_reason(feature, raw_value)))
    entries.sort(key=lambda item: abs(item[1]), reverse=True)
    # Use contribution direction, with each feature's value supplying the human-readable context.
    raised = [(name, reason) for name, impact, reason in entries if impact > 0]
    lowered = [(name, reason) for name, impact, reason in entries if impact < 0]
    if raised:
        st.markdown(f"**Why {model_name} leaned {class_name}:**")
        for name, reason in raised[:4]:
            st.write(f"- {reason} This pushed {model_name}’s {class_name} score up.")
    if lowered:
        st.markdown(f"**Factors that pulled {model_name} away from {class_name}:**")
        for name, reason in lowered[:3]:
            st.write(f"- {reason} This pulled {model_name}’s {class_name} score down.")
    if not raised and not lowered:
        st.write("No clear feature-level reason was available for this row.")

def predict_ft_probabilities(scaled_values):
    """Return FT-Transformer probabilities for scaled feature arrays (SHAP adapter)."""
    values = np.asarray(scaled_values, dtype=np.float32)
    with torch.no_grad():
        logits = ft_model(torch.tensor(values, dtype=torch.float32))
        return F.softmax(logits, dim=1).cpu().numpy()

def get_ft_row_shap(scaled_values, row_index, class_index):
    """Explain one FT-Transformer row with permutation SHAP around training means."""
    # The scaler's zero point is the training-set mean, giving a stable baseline
    # that does not change when unrelated rows are added to the upload.
    baseline = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
    explainer = shap.Explainer(
        predict_ft_probabilities,
        shap.maskers.Independent(baseline),
        algorithm='permutation',
    )
    values = explainer(
        np.asarray(scaled_values[row_index:row_index + 1], dtype=np.float32),
        max_evals=2 * len(FEATURE_COLS) + 1,
    )
    return class_shap_values(values.values, class_index, 0)

def source_mapping_table(source_df):
    normalized = {''.join(ch.lower() for ch in str(c) if ch.isalnum()): str(c) for c in source_df.columns}
    rows = []
    for canonical, aliases in SOURCE_ALIASES.items():
        for alias in (canonical, *aliases):
            actual = normalized.get(''.join(ch.lower() for ch in alias if ch.isalnum()))
            if actual:
                feature = {
                    'tender_description': 'SBERT text input and text-derived indicators',
                    'number_of_bidders': FEATURE_NAMES['party_count'],
                    'estimated_value': FEATURE_NAMES['tender_value_amount_num'],
                    'complaint_count': 'Complaint presence and severity proxy',
                    'restrictive_specs_flag': FEATURE_NAMES['flag_brand_specs'] + ' (proxy mapping)',
                    'contract_amendments': 'Additional review signal only',
                    'contract_value_increase_pct': 'Additional review signal only',
                    'repeat_winner_flag': 'Additional review signal only',
                    'award_justification_status': 'Additional review signal only',
                    'procurement_method': 'Additional review signal only',
                    'bid_submission_days': 'Additional review signal only',
                }.get(canonical, 'Supporting review signal')
                rows.append({'Input column': actual, 'Mapped term': canonical, 'App use': feature})
                break
    return pd.DataFrame(rows)

def compute_sbert_similarity(texts, risk_keywords=["korrupsion", "favorizim", "kufizim", "skualifikim"]):
    """Computes semantic similarity to corruption risk concepts using SBERT"""
    embeddings = sbert.encode(texts, convert_to_tensor=True)
    risk_embs = sbert.encode(risk_keywords, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), risk_embs.unsqueeze(0), dim=2)
    max_scores, _ = torch.max(cosine_scores, dim=1)
    return max_scores.cpu().numpy()

# --- UI Sidebar ---
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Public Procurement CSV", type=["csv"])

# --- Main Dashboard ---
st.title("⚖️ Public Procurement Corruption Risk Intelligence")
st.markdown("Use Sentence-BERT to score a description by itself, or upload a structured dataset to use the existing LightGBM + FT-Transformer models together with Sentence-BERT.")

with st.expander("Held-out model evaluation"):
    st.caption("Scores below measure agreement with rule-generated risk categories. They are not validated corruption prediction accuracy. Text and tabular scores use separate test sets.")
    eval_columns = st.columns(4)
    if tabular_metrics:
        tab_metrics = tabular_metrics.get('metrics', {})
        for col, key, title in zip(eval_columns[:3], ('lightgbm', 'ft_transformer', 'soft_voting_ensemble'),
                                   ('LightGBM', 'FT-Transformer', 'Tabular ensemble')):
            values = tab_metrics.get(key, {})
            col.metric(f'{title} accuracy', f"{values.get('accuracy', 0) * 100:.1f}%")
            col.caption(f"Balanced accuracy {values.get('balanced_accuracy', 0) * 100:.1f}% · Macro-F1 {values.get('macro_f1', 0) * 100:.1f}%")
        eval_columns[3].metric('Tabular test rows', str(tabular_metrics.get('test_rows', 'N/A')))
    text_eval = sbert_metadata.get('held_out_test_metrics', {})
    if text_eval:
        st.markdown(
            f"**Sentence-BERT** — {text_eval.get('accuracy', 0) * 100:.1f}% accuracy, "
            f"{text_eval.get('balanced_accuracy', 0) * 100:.1f}% balanced accuracy, "
            f"{text_eval.get('macro_f1', 0) * 100:.1f}% macro-F1 on {text_eval.get('rows', 0)} held-out rows."
        )
        report = text_eval.get('classification_report', {})
        if report:
            st.dataframe(pd.DataFrame(report).T.loc[[c for c in label_enc.classes_ if c in report],
                                                    ['precision', 'recall', 'f1-score', 'support']],
                         use_container_width=True)

st.header("Predict from a description")
with st.form("description_only_form"):
    description_input = st.text_area(
        "Tender description", height=150,
        placeholder="Paste one procurement description. No other fields are required.",
    )
    description_submitted = st.form_submit_button("Predict from description")

if description_submitted:
    if not description_input.strip():
        st.warning("Enter a description first.")
    else:
        with st.spinner("Encoding description with Sentence-BERT..."):
            text_label, text_probs = predict_description(description_input.strip())
        st.subheader(f"Sentence-BERT estimate: {text_label}")
        score_cols = st.columns(len(label_enc.classes_))
        for i, class_name in enumerate(label_enc.classes_):
            score_cols[i].metric(f"{class_name} score", f"{text_probs[i] * 100:.1f}%")
        st.caption("These are model scores for the dataset's rule-generated risk categories, not validated corruption probabilities.")
        st.subheader("SHAP explanation for this description")
        st.caption("Token SHAP shows which words moved the selected Sentence-BERT class score up or down. It explains this model output, not a verified corruption outcome.")
        try:
            show_text_shap(description_input.strip(), int(np.argmax(text_probs)))
        except Exception as exc:
            st.warning(f"Text SHAP could not be generated for this input: {exc}")

if uploaded_file is not None:
    with st.spinner("Processing Data and Running AI Models..."):
        source_df = pd.read_csv(uploaded_file)
        
        # SBERT NLP Processing (if text available)
        descriptions = get_descriptions(canonicalize_source_columns(source_df))
        df = source_df.copy()
        if descriptions is not None:
            df['sbert_semantic_risk'] = compute_sbert_similarity(descriptions)
        else:
            df['sbert_semantic_risk'] = 0.0
            
        # Ensemble Risk Prediction
        model_preds, probs, rule_preds, rule_scores, df, schema_adapted = predict_risk(df)
        if rule_preds is not None:
            df['Model_Ensemble_Prediction'] = model_preds
            df['Composite_Risk_Score'] = rule_scores
            df['Predicted_Risk_Level'] = rule_preds
        else:
            df['Predicted_Risk_Level'] = model_preds
        high_idx = int(np.where(label_enc.classes_ == 'HIGH')[0][0]) if 'HIGH' in label_enc.classes_ else int(np.argmax(probs[0]))
        df['Model_HIGH_Score'] = probs[:, high_idx] * 100
        mapping_df = source_mapping_table(source_df)
        lgb_shap_values = None
        if all(col in df.columns for col in FEATURE_COLS):
            try:
                lgb_shap_values = shap.TreeExplainer(lgb_model).shap_values(
                    scaler.transform(df[FEATURE_COLS].copy().fillna(0.0))
                )
            except Exception as exc:
                shap_error = str(exc)
            else:
                shap_error = None
        else:
            shap_error = 'No structured model feature columns were available.'
        
    st.success("Analysis Complete!")
    if schema_adapted:
        st.warning(
            "Some source terms were mapped to the trained feature set. This mapping is approximate. "
            "New indicators such as procedure type, bid period, repeat winner, amendments, value increases, "
            "and award-justification status are shown as review signals when available; the trained models "
            "were not trained on those fields. This file has no verified risk labels, so its accuracy "
            "cannot be measured from this file alone."
        )
    with st.expander("Recognized procurement terms and mappings"):
        if not mapping_df.empty:
            st.dataframe(mapping_df, hide_index=True, use_container_width=True)
        else:
            st.write("No alternate field names were mapped. The app used the provided engineered features or description text.")
    if 'Composite_Risk_Score' in df.columns:
        model_mismatches = int((df['Model_Ensemble_Prediction'] != df['Predicted_Risk_Level']).sum())
        st.info(
            "For structured rows, Predicted_Risk_Level applies the same documented scoring rule that generated "
            "the project's training labels. Model_Ensemble_Prediction is retained for comparison. "
            f"The LightGBM/FT/SBERT ensemble differs from that rule on {model_mismatches} of {len(df)} rows."
        )
    
    # KPIs
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Contracts Evaluated", len(df))
    col2.metric("High Risk Contracts", len(df[df['Predicted_Risk_Level'] == 'HIGH']))
    avg_sbert = f"{df['sbert_semantic_risk'].mean():.2f}" if 'sbert_semantic_risk' in df.columns else "N/A"
    col3.metric("Avg SBERT Semantic Risk", avg_sbert)
    
    # Prediction charts
    st.header("Prediction charts")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        counts = df['Predicted_Risk_Level'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'], fill_value=0)
        risk_counts = counts.rename_axis('Risk level').reset_index(name='Contracts')
        fig1 = px.bar(risk_counts, x='Risk level', y='Contracts', color='Risk level',
                      color_discrete_map={'HIGH': '#d62728', 'MEDIUM': '#ff9f1c', 'LOW': '#2ca02c'},
                      title='Predicted risk category counts', text='Contracts')
        st.plotly_chart(fig1, use_container_width=True)
    with chart_right:
        if 'Composite_Risk_Score' in df.columns:
            fig2 = px.histogram(df, x='Composite_Risk_Score', color='Predicted_Risk_Level', nbins=15,
                                color_discrete_map={'HIGH': '#d62728', 'MEDIUM': '#ff9f1c', 'LOW': '#2ca02c'},
                                title='Proxy risk score distribution')
        else:
            fig2 = px.histogram(df, x='Model_HIGH_Score', color='Predicted_Risk_Level', nbins=15,
                                title='Model HIGH class score distribution')
        st.plotly_chart(fig2, use_container_width=True)

    if 'Model_Ensemble_Prediction' in df.columns:
        compare = pd.concat([
            df['Predicted_Risk_Level'].value_counts().rename_axis('Risk level').rename('Displayed category'),
            df['Model_Ensemble_Prediction'].value_counts().rename_axis('Risk level').rename('ML ensemble category'),
        ], axis=1).fillna(0).reset_index().melt(id_vars='Risk level', var_name='Prediction source', value_name='Contracts')
        fig_compare = px.bar(compare, x='Risk level', y='Contracts', color='Prediction source', barmode='group',
                             title='Rule category vs. model ensemble')
        st.plotly_chart(fig_compare, use_container_width=True)

    if lgb_shap_values is not None:
        st.subheader("Plain-language SHAP explanation (LightGBM component)")
        shap_class = st.selectbox("Explain model class", list(label_enc.classes_),
                                  index=int(np.where(label_enc.classes_ == 'HIGH')[0][0]) if 'HIGH' in label_enc.classes_ else 0)
        class_index = int(np.where(label_enc.classes_ == shap_class)[0][0])
        global_values = np.asarray([class_shap_values(lgb_shap_values, class_index, i) for i in range(len(df))])
        avg_signed = np.mean(global_values, axis=0)
        influential = np.argsort(np.abs(avg_signed))[::-1][:5]
        st.write(f"Across the uploaded rows, these factors had the strongest average influence on LightGBM’s **{shap_class}** score:")
        for i in influential:
            direction = 'raised' if avg_signed[i] > 0 else 'lowered'
            st.write(f"- **{FEATURE_NAMES.get(FEATURE_COLS[i], FEATURE_COLS[i])}** generally {direction} that class score.")
        st.caption("This summary describes LightGBM’s class score, not the final rule-based category shown for each row.")
    elif shap_error:
        st.info(f"Structured SHAP chart unavailable: {shap_error}")

    if any(col.startswith('signal_') for col in df.columns):
        signal_cols = [col for col in df.columns if col.startswith('signal_')]
        signal_counts = pd.DataFrame({
            'Additional review signal': [col.replace('signal_', '').replace('_', ' ').title() for col in signal_cols],
            'Contracts flagged': [int(pd.to_numeric(df[col], errors='coerce').fillna(0).sum()) for col in signal_cols],
        }).sort_values('Contracts flagged')
        fig_signals = px.bar(signal_counts, x='Contracts flagged', y='Additional review signal', orientation='h',
                             title='Other mapped procurement review signals')
        st.plotly_chart(fig_signals, use_container_width=True)

    st.subheader("Why a contract received its category")
    selected_row = st.selectbox('Choose a contract row', range(1, len(df) + 1), format_func=lambda n: f'Row {n}') - 1
    selected = df.iloc[selected_row]
    st.markdown(f"**Displayed category:** {selected['Predicted_Risk_Level']}")
    if 'Composite_Risk_Score' in df.columns:
        st.markdown(f"**Rule score:** {selected['Composite_Risk_Score']:.2f}/100. LOW < 20, MEDIUM 20–39.99, HIGH ≥ 40.")
        rule_points = rule_score_components(selected)
        positive_points = [(name, points) for name, points in rule_points.items() if points > 0]
        if positive_points:
            st.write("**Signals that added points:**")
            for name, points in sorted(positive_points, key=lambda item: item[1], reverse=True):
                if name == 'Complaint filed':
                    reason = 'A complaint was recorded'
                elif name == 'Restrictive or brand specifications':
                    reason = 'Restrictive or brand-specific specifications were flagged'
                elif name == 'Risk-related description terms':
                    reason = 'Risk-related terms appeared in the description'
                elif name == 'Short description':
                    reason = 'The description was unusually short'
                elif name == 'Complaint count / severity proxy':
                    reason = 'The complaint count/severity indicator added points'
                else:
                    reason = 'The category-relative tender value was unusually high'
                st.write(f"- {reason} (+{points:g} points).")
        else:
            st.write("No positive risk indicators in the rule score added points for this row.")
        st.caption("These point contributions show how the displayed rule category was calculated.")
    more_signals = get_additional_signals(selected)
    if more_signals:
        st.write("Other review signals (not included in the trained model score): " + '; '.join(more_signals))
    elif any(col.startswith('signal_') for col in df.columns):
        st.write("No additional mapped review signals were triggered for this contract.")
    if lgb_shap_values is not None and 'Model_Ensemble_Prediction' in df.columns:
        model_class = str(selected['Model_Ensemble_Prediction'])
        model_class_index = int(np.where(label_enc.classes_ == model_class)[0][0])
        st.caption(
            "The following explanations describe each model’s contribution to the ML ensemble class. "
            "The displayed rule-based category and its point contributions are shown separately above."
        )
        selected_shap = class_shap_values(lgb_shap_values, model_class_index, selected_row)
        st.markdown("**LightGBM explanation**")
        explain_shap_in_plain_language(selected_shap, selected, model_class, 'LightGBM')
        st.markdown("**FT-Transformer explanation**")
        try:
            ft_row_shap = get_ft_row_shap(X_scaled, selected_row, model_class_index)
            explain_shap_in_plain_language(ft_row_shap, selected, model_class, 'FT-Transformer')
            st.caption("FT-Transformer SHAP compares this row with the model’s average training-feature values.")
        except Exception as exc:
            st.info(f"FT-Transformer explanation could not be generated for this row: {exc}")
        selected_description = get_descriptions(df)
        if selected_description is not None and str(selected_description[selected_row]).strip():
            st.markdown("**Sentence-BERT description explanation**")
            try:
                show_text_shap(str(selected_description[selected_row]), model_class_index)
            except Exception as exc:
                st.info(f"Sentence-BERT text explanation could not be generated for this row: {exc}")
        else:
            st.caption("Sentence-BERT is not part of this row’s prediction because no description was provided.")
    
    # Detailed Data Table
    st.header("High-Risk Contracts Identified")
    sort_col = 'Composite_Risk_Score' if 'Composite_Risk_Score' in df.columns else 'Model_HIGH_Score'
    high_risk_df = df[df['Predicted_Risk_Level'] == 'HIGH'].sort_values(by=sort_col, ascending=False)
    st.dataframe(high_risk_df.head(50))
    
    st.download_button("Download Full Risk Report (CSV)", df.to_csv(index=False).encode('utf-8'), "corruption_risk_report.csv", "text/csv")
else:
    st.info("Enter a description above for a Sentence-BERT prediction, or upload a structured CSV for the tabular ensemble.")
    st.caption("Risk categories are screening signals learned from proxy labels. They are not confirmed corruption findings or calibrated real-world probabilities.")
