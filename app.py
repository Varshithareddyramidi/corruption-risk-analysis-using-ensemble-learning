import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
import joblib
from sentence_transformers import SentenceTransformer
import os
import sys

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.ft_transformer import TabularFTTransformer

# --- Configuration & Caching ---
st.set_page_config(page_title="Procurement Corruption Risk AI", layout="wide", page_icon="⚖️")

@st.cache_resource
def load_all_models():
    """Load Ensemble Models and NLP SBERT model"""
    # 1. Load SBERT NLP Module
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Load LightGBM, Scaler, Encoder
    lgb_model = joblib.load('models/lgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_enc = joblib.load('models/label_enc.pkl')
    
    # 3. Load FT-Transformer
    num_classes = len(label_enc.classes_)
    num_features = len(scaler.scale_)
    ft_model = TabularFTTransformer(num_features=num_features, num_classes=num_classes)
    ft_model.load_state_dict(torch.load('models/ft_model.pth', map_location=torch.device('cpu')))
    ft_model.eval()
    
    return sbert_model, lgb_model, ft_model, scaler, label_enc

sbert, lgb_model, ft_model, scaler, label_enc = load_all_models()

# --- Utility Functions ---
def predict_risk(df):
    """Predicts risk using Ensemble of LightGBM, FT-Transformer and SBERT contextual scores"""
    
    feature_cols = [
        'party_count', 'cpv_count', 'has_participation_fee', 
        'tender_value_amount_num', 'tender_value_log_zscore', 
        'flag_short_description', 'text_risk_keyword_score', 
        'flag_brand_specs', 'complaint_filed_flag', 
        'complaint_severity_score', 'text_word_count', 
        'flag_high_value_outlier'
    ]
    
    # Ensure columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[feature_cols].copy().fillna(0.0)
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
        
    # Soft Voting Ensemble
    ensemble_probs = 0.5 * full_probs_lgb + 0.5 * ft_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    predicted_labels = label_enc.inverse_transform(ensemble_preds)
    
    return predicted_labels, ensemble_probs

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
st.markdown("A unified AI system combining **Sentence-BERT (NLP)** and an **Ensemble Learning pipeline (LightGBM + FT-Transformer)** for robust corruption risk detection.")

if uploaded_file is not None:
    with st.spinner("Processing Data and Running AI Models..."):
        df = pd.read_csv(uploaded_file)
        
        # SBERT NLP Processing (if text available)
        if 'text_description' in df.columns or 'title' in df.columns:
            text_col = 'text_description' if 'text_description' in df.columns else 'title'
            df['sbert_semantic_risk'] = compute_sbert_similarity(df[text_col].fillna("").tolist())
        else:
            df['sbert_semantic_risk'] = 0.0
            
        # Ensemble Risk Prediction
        preds, probs = predict_risk(df)
        df['Predicted_Risk_Level'] = preds
        df['Risk_Probability'] = np.max(probs, axis=1) * 100
        
    st.success("Analysis Complete!")
    
    # KPIs
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Contracts Evaluated", len(df))
    col2.metric("High Risk Contracts", len(df[df['Predicted_Risk_Level'] == 'HIGH']))
    avg_sbert = f"{df['sbert_semantic_risk'].mean():.2f}" if 'sbert_semantic_risk' in df.columns else "N/A"
    col3.metric("Avg SBERT Semantic Risk", avg_sbert)
    
    # Visualizations
    st.header("Risk Analysis Reports")
    
    fig1 = px.pie(df, names='Predicted_Risk_Level', title="Risk Level Distribution", hole=0.4, color='Predicted_Risk_Level', 
                  color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'})
    
    st.plotly_chart(fig1, use_container_width=True)
    
    if 'tender_value_amount_num' in df.columns:
        fig2 = px.scatter(df, x='tender_value_amount_num', y='Risk_Probability', color='Predicted_Risk_Level',
                          size='sbert_semantic_risk' if 'sbert_semantic_risk' in df.columns else None,
                          hover_data=['party_count', 'cpv_count'],
                          title="Tender Value vs. Risk Probability (Bubble Size = SBERT Semantic Risk)",
                          color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed Data Table
    st.header("High-Risk Contracts Identified")
    high_risk_df = df[df['Predicted_Risk_Level'] == 'HIGH'].sort_values(by='Risk_Probability', ascending=False)
    st.dataframe(high_risk_df.head(50))
    
    st.download_button("Download Full Risk Report (CSV)", df.to_csv(index=False).encode('utf-8'), "corruption_risk_report.csv", "text/csv")
else:
    st.info("Please upload a procurement dataset (CSV) from the sidebar to begin analysis.")
    st.markdown("""
    ### System Architecture:
    1. **Sentence-BERT NLP Module**: Analyzes descriptive tender fields (titles, descriptions) to detect semantic similarities to corrupt practices (favoritism, artificial restrictions).
    2. **LightGBM Model**: High-performance gradient boosting decision tree detecting anomalies in tender values, CPV counts, and procedural flags.
    3. **FT-Transformer (PyTorch)**: Deep learning tabular transformer identifying non-linear interactions across structured financial and legal red flags.
    4. **Ensemble Soft-Voting**: Merges all probabilistic signals for a highly calibrated final risk score.
    """)
