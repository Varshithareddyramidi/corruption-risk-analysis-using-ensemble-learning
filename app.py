import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import joblib
from sentence_transformers import SentenceTransformer
import shap
import os
import sys

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.ft_transformer import TabularFTTransformer

# --- Feature Metadata & Explanations for Common People ---
FEATURE_COLS = [
    'party_count', 'cpv_count', 'has_participation_fee', 
    'tender_value_amount_num', 'tender_value_log_zscore', 
    'flag_short_description', 'text_risk_keyword_score', 
    'flag_brand_specs', 'complaint_filed_flag', 
    'complaint_severity_score', 'text_word_count', 
    'flag_high_value_outlier'
]

FEATURE_LABELS = {
    'party_count': 'Number of Participating Bidders',
    'cpv_count': 'Procurement Categories Involved',
    'has_participation_fee': 'Requirement to Pay Entry/Participation Fee',
    'tender_value_amount_num': 'Total Contract Amount (Value)',
    'tender_value_log_zscore': 'Value Discrepancy Compared to Normal Tenders',
    'flag_short_description': 'Tender Description Unusually Brief or Opaque',
    'text_risk_keyword_score': 'Suspicious or Restrictive Language in Terms',
    'flag_brand_specs': 'Demanding Specific Commercial Brand Names',
    'complaint_filed_flag': 'Formal Citizen or Vendor Complaint Filed',
    'complaint_severity_score': 'Seriousness Level of Filed Complaints',
    'text_word_count': 'Detail Level of Tender Documentation',
    'flag_high_value_outlier': 'Abnormally Expensive Contract Threshold'
}

PLAIN_EXPLANATIONS = {
    'party_count': "Very few bidders competing (or only 1 sole bidder), eliminating healthy price competition.",
    'has_participation_fee': "Mandating an entry fee before companies can bid, which discourages smaller honest businesses from participating.",
    'flag_brand_specs': "Tender explicitly demands specific private brand names instead of generic quality standards, effectively tailoring the deal for one favored vendor.",
    'complaint_filed_flag': "Disqualified suppliers or watchdogs filed official legal complaints alleging unfair evaluation criteria.",
    'complaint_severity_score': "Severe grievances or disputes were formally lodged against the tender terms.",
    'flag_high_value_outlier': "The contract budget is unusually massive compared to standard market benchmarks.",
    'tender_value_log_zscore': "Price is mathematically abnormal compared to similar government procurement projects.",
    'text_risk_keyword_score': "Contains high concentrations of exclusionary words (e.g., restricted eligibility, disqualification clauses).",
    'flag_short_description': "The tender description is intentionally vague and brief, hiding key technical scopes from other bidders.",
    'tender_value_amount_num': "High overall budget scale amplifies the fiscal damage of potential corruption.",
    'text_word_count': "Unusually sparse documentation lacking standard transparency details.",
    'cpv_count': "Bundling unrelated product or service categories to confuse or restrict competitive bidders."
}

# --- Configuration & Caching ---
st.set_page_config(page_title="Public Tender Transparency & Risk Portal", layout="wide", page_icon="⚖️")

@st.cache_resource
def load_all_models():
    """Load evaluation models, transformers, and SHAP explainers"""
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    lgb_model = joblib.load('models/lgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_enc = joblib.load('models/label_enc.pkl')
    
    num_classes = len(label_enc.classes_)
    num_features = len(scaler.scale_)
    ft_model = TabularFTTransformer(num_features=num_features, num_classes=num_classes)
    ft_model.load_state_dict(torch.load('models/ft_model.pth', map_location=torch.device('cpu')))
    ft_model.eval()
    
    # Pre-initialize SHAP TreeExplainer
    shap_explainer = shap.TreeExplainer(lgb_model)
    
    return sbert_model, lgb_model, ft_model, scaler, label_enc, shap_explainer

sbert, lgb_model, ft_model, scaler, label_enc, shap_explainer = load_all_models()

# --- Utility Functions ---
def predict_risk(df):
    """Predicts risk using calibrated ensemble and returns probabilities and scaled features"""
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[FEATURE_COLS].copy().fillna(0.0)
    X_scaled = scaler.transform(X)
    
    lgb_probs = lgb_model.predict_proba(X_scaled)
    num_classes = len(label_enc.classes_)
    full_probs_lgb = np.zeros((len(X_scaled), num_classes))
    for i, cls_idx in enumerate(lgb_model.classes_):
        full_probs_lgb[:, cls_idx] = lgb_probs[:, i]
        
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = ft_model(X_tensor)
        ft_probs = F.softmax(logits, dim=1).numpy()
    
    if ft_probs.shape[1] > num_classes:
        ft_probs = ft_probs[:, :num_classes]
        
    ensemble_probs = 0.5 * full_probs_lgb + 0.5 * ft_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    predicted_labels = label_enc.inverse_transform(ensemble_preds)
    
    return predicted_labels, ensemble_probs, X_scaled, X

def compute_sbert_similarity(texts, risk_keywords=["korrupsion", "favorizim", "kufizim", "skualifikim"]):
    """Computes semantic similarity to suspicious contract clauses"""
    embeddings = sbert.encode(texts, convert_to_tensor=True)
    risk_embs = sbert.encode(risk_keywords, convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), risk_embs.unsqueeze(0), dim=2)
    max_scores, _ = torch.max(cosine_scores, dim=1)
    return max_scores.cpu().numpy()

def compute_shap_explanations(X_scaled):
    """Computes SHAP feature attributions oriented towards HIGH corruption risk"""
    sv = shap_explainer.shap_values(X_scaled)
    
    # In binary classification with classes ['HIGH'=0, 'MEDIUM'=1]:
    # Negative raw score means higher probability of HIGH (class 0).
    # Therefore, negative SHAP contribution increases HIGH risk probability.
    # We negate the values so that positive contribution indicates driving UP HIGH risk.
    if isinstance(sv, list):
        if len(sv) == 2:
            high_risk_shap = sv[0]
        else:
            high_risk_shap = -sv[0]
    elif isinstance(sv, np.ndarray):
        if sv.ndim == 3:
            high_risk_shap = sv[:, :, 0]
        else:
            high_risk_shap = -sv
    else:
        high_risk_shap = np.zeros(X_scaled.shape)
        
    return high_risk_shap

# --- Sidebar ---
st.sidebar.title("📁 Upload Tender Data")
uploaded_file = st.sidebar.file_uploader("Upload Public Procurement CSV file", type=["csv"])

# --- Main Page Header (Clean, Public-Friendly) ---
st.title("⚖️ Public Tender Integrity & Corruption Risk Monitor")
st.markdown("""
Welcome to the **Public Tender Risk Intelligence Portal**. This tool helps citizens, journalists, 
and oversight committees examine public government contracts to see if any tender exhibits warning signs 
of corruption, favoritism, or lack of fair competition.
""")

if uploaded_file is None:
    st.info("👈 **To get started:** Please upload a tender records CSV file using the sidebar on the left.")
    
    st.markdown("---")
    st.subheader("💡 What Warning Signs Does This System Detect?")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        - 🚫 **Unfair Single-Bidder Competitions**: Tenders where only one vendor participates, eliminating healthy price competition.
        - 🏷️ **Brand-Specific Requirements**: Specifications that explicitly demand proprietary brand names, tailoring the deal for one favored company.
        - 💰 **Participation Entry Fees**: Requiring vendors to pay mandatory entry fees, deterring small or independent competitors.
        """)
    with col_b:
        st.markdown("""
        - ⚖️ **Formal Bidder Complaints**: Contested tenders where disqualified businesses filed formal challenges alleging unfair evaluation.
        - 📈 **Abnormal Contract Valuations**: Budgets that deviate dramatically from market rates for similar public projects.
        - 📝 **Opaque Documentation**: Vague or secretive contract descriptions that hide critical details from genuine competitors.
        """)
    
    st.markdown("---")
    st.markdown("""
    > **Transparency Guarantee:** Every flagged contract includes an easy-to-read breakdown powered by **SHAP (Shapley Explanations)**, 
    > showing regular citizens exactly which red flags caused the tender to be categorized as High Risk.
    """)

else:
    with st.spinner("Analyzing contracts and calculating transparency indicators..."):
        df = pd.read_csv(uploaded_file)
        
        # SBERT semantic risk if text available
        if 'text_description' in df.columns or 'title' in df.columns:
            text_col = 'text_description' if 'text_description' in df.columns else 'title'
            df['suspicious_language_score'] = compute_sbert_similarity(df[text_col].fillna("").tolist())
        else:
            df['suspicious_language_score'] = 0.0
            
        # Predictions
        preds, probs, X_scaled, X_raw = predict_risk(df)
        df['Risk_Level'] = preds
        df['Risk_Confidence'] = np.max(probs, axis=1) * 100
        
        # Calculate SHAP Explanations
        shap_values_matrix = compute_shap_explanations(X_scaled)
        
    st.success("✅ Analysis Complete!")
    
    # Key Summary Cards
    st.header("📊 Overview Summary")
    c1, c2, c3, c4 = st.columns(4)
    total_count = len(df)
    high_count = len(df[df['Risk_Level'] == 'HIGH'])
    med_count = len(df[df['Risk_Level'] == 'MEDIUM'])
    low_count = len(df[df['Risk_Level'] == 'LOW']) if 'LOW' in df['Risk_Level'].values else 0
    
    c1.metric("Contracts Evaluated", f"{total_count:,}")
    c2.metric("High Risk Tenders Flagged", f"{high_count:,}", delta=f"{(high_count/total_count*100):.1f}%" if total_count else "0%", delta_color="inverse")
    c3.metric("Medium Risk Tenders", f"{med_count:,}")
    c4.metric("Avg Suspicious Language Score", f"{df['suspicious_language_score'].mean():.2f}")
    
    # Visual Overview
    st.markdown("---")
    st.subheader("📈 Risk Breakdown Across Evaluated Contracts")
    vcol1, vcol2 = st.columns([1, 1])
    
    with vcol1:
        fig_pie = px.pie(
            df, 
            names='Risk_Level', 
            title="Distribution of Contract Risk Levels",
            hole=0.45, 
            color='Risk_Level',
            color_discrete_map={'HIGH': '#d32f2f', 'MEDIUM': '#f57c00', 'LOW': '#388e3c'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with vcol2:
        if 'tender_value_amount_num' in df.columns:
            fig_scatter = px.scatter(
                df, 
                x='tender_value_amount_num', 
                y='Risk_Confidence', 
                color='Risk_Level',
                size='suspicious_language_score' if 'suspicious_language_score' in df.columns else None,
                hover_data=['party_count', 'cpv_count'] if 'party_count' in df.columns else None,
                title="Tender Value vs. Risk Confidence Level",
                labels={'tender_value_amount_num': 'Contract Value', 'Risk_Confidence': 'Risk Confidence (%)'},
                color_discrete_map={'HIGH': '#d32f2f', 'MEDIUM': '#f57c00', 'LOW': '#388e3c'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    # --- SHAP EXPLAINABILITY SECTION (Designed for Common People) ---
    st.markdown("---")
    st.header("🔍 Why is a Tender Marked as High Risk? (SHAP Transparency Breakdown)")
    st.markdown("""
    **Understanding the Red Flags:** Ordinary citizens should not need a computer science degree to know why public funds are at risk.
    Below, our **SHAP AI Explainer** reveals the exact real-world factors that pushed each tender into the **High Risk** category.
    """)
    
    # Filter high risk contracts
    high_risk_indices = df[df['Risk_Level'] == 'HIGH'].index.tolist()
    
    if len(high_risk_indices) == 0:
        st.info("🎉 Great news! No high-risk contracts were detected in this dataset.")
    else:
        # Create an intuitive selector for common people
        tender_labels = []
        for idx in high_risk_indices:
            name_str = f"Tender #{idx + 1}"
            if 'title' in df.columns and pd.notna(df.loc[idx, 'title']):
                title_snip = str(df.loc[idx, 'title'])[:45]
                name_str += f" - {title_snip}..."
            if 'tender_value_amount_num' in df.columns:
                val = df.loc[idx, 'tender_value_amount_num']
                name_str += f" | Value: ${val:,.0f}" if val > 0 else ""
            tender_labels.append(name_str)
            
        selected_label = st.selectbox(
            "👉 Select a flagged high-risk tender to inspect its detailed explanation:",
            options=tender_labels
        )
        selected_idx = high_risk_indices[tender_labels.index(selected_label)]
        row_data = df.loc[selected_idx]
        
        # Show key attributes in easy-to-read cards
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Risk Status", f"⚠️ {row_data.get('Risk_Level', 'HIGH')}")
        with info_col2:
            st.metric("Risk Certainty", f"{row_data.get('Risk_Confidence', 0):.1f}%")
        with info_col3:
            val_num = row_data.get('tender_value_amount_num', 0)
            st.metric("Contract Value", f"${val_num:,.2f}" if val_num else "Not specified")
            
        if 'title' in df.columns and pd.notna(row_data['title']):
            st.markdown(f"**Tender Title:** *{row_data['title']}*")
        if 'text_description' in df.columns and pd.notna(row_data['text_description']):
            with st.expander("📄 Read Original Tender Description"):
                st.write(row_data['text_description'])
                
        # Compute top contributing factors for this row
        row_shap = shap_values_matrix[selected_idx]
        shap_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'label': [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
            'shap_value': row_shap,
            'raw_value': [row_data.get(f, 0.0) for f in FEATURE_COLS]
        })
        
        # Sort factors pushing risk higher (positive shap_value)
        top_risk_drivers = shap_df.sort_values(by='shap_value', ascending=False)
        primary_reasons = top_risk_drivers[top_risk_drivers['shap_value'] > 0.001]
        
        st.subheader("📋 Plain English Explanation of Red Flags")
        if len(primary_reasons) > 0:
            st.markdown("The system flagged this contract primarily because of the following warning signs:")
            for rank, (_, reason) in enumerate(primary_reasons.head(4).iterrows(), 1):
                f_name = reason['feature']
                explanation = PLAIN_EXPLANATIONS.get(f_name, "This metric displayed values significantly outside standard procurement benchmarks.")
                st.markdown(f"""
                **{rank}. {reason['label']}** *(Impact Score: +{reason['shap_value']:.2f})*  
                - **Observed Value in Record:** `{reason['raw_value']}`  
                - **What this means in plain terms:** {explanation}
                """)
        else:
            st.write("Multiple subtle risk indicators combined across contract valuation and participation patterns to raise this contract's risk level.")
            
        # Visual SHAP Bar Chart
        st.subheader("📊 Visual Breakdown of What Pushed the Risk Up or Down")
        st.caption("Red bars pushed the tender **towards High Risk**. Green bars indicate factors that were normal or helped lower the risk.")
        
        # Display top 8 most influential factors for clarity
        top_plot_df = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index).head(8)
        top_plot_df = top_plot_df.sort_values(by='shap_value', ascending=True)
        
        fig_bar = go.Figure()
        colors = ['#d32f2f' if v >= 0 else '#388e3c' for v in top_plot_df['shap_value']]
        
        fig_bar.add_trace(go.Bar(
            y=top_plot_df['label'],
            x=top_plot_df['shap_value'],
            orientation='h',
            marker_color=colors,
            text=[f"{'+' if v>=0 else ''}{v:.2f}" for v in top_plot_df['shap_value']],
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            title="Key Factors Influencing This Tender's Risk Evaluation",
            xaxis_title="Contribution to High Risk (Positive = Increases Risk, Negative = Decreases Risk)",
            yaxis_title="",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Summary Box
        st.info("""
        💡 **Summary for Citizens:**  
        In fair procurement, multiple bidders compete freely without restrictive brand requirements, without arbitrary entrance fees, 
        and without official complaints. When these red flags are present, public funds are at substantial risk of being misallocated.
        """)
        
    # --- Table of All Flagged High Risk Contracts ---
    st.markdown("---")
    st.header("📑 All Flagged High-Risk Contracts")
    high_risk_table = df[df['Risk_Level'] == 'HIGH'].sort_values(by='Risk_Confidence', ascending=False)
    
    # Friendly column presentation
    display_cols = [c for c in ['title', 'tender_value_amount_num', 'party_count', 'complaint_filed_flag', 'Risk_Confidence', 'Risk_Level'] if c in df.columns]
    st.dataframe(high_risk_table[display_cols].head(50), use_container_width=True)
    
    st.download_button(
        "📥 Download Full Integrity Audit Report (CSV)", 
        df.to_csv(index=False).encode('utf-8'), 
        "public_tender_risk_report.csv", 
        "text/csv"
    )
