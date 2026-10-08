"""
Corruption Risk Feature Extractor Module.
Engineers textual, legal, financial, and procedural red flags for public procurement corruption risk analysis.
"""

import re
import math
from collections import defaultdict


# High-risk corruption / procedural conflict keywords in Albanian procurement text
RISK_KEYWORDS = {
    # Disqualification / Exclusion
    'skualifikim': 2.5,
    'perjashtim': 2.5,
    'skualifikuar': 2.0,
    'jashtezakonshme': 1.5,
    
    # Specification / Restriction / Favoritism
    'specifikim': 1.5,
    'specifikime': 1.5,
    'kufizim': 2.0,
    'kriter': 1.0,
    'kriteri': 1.0,
    'favorizim': 3.0,
    'monopol': 3.0,
    'brend': 2.0,
    'patente': 2.0,
    'eksklusiv': 2.5,

    # Dispute / Appeal / Procedural violation
    'ankese': 1.5,
    'ankesekpp': 2.0,
    'shkelje': 2.5,
    'pezullim': 2.0,
    'anulim': 2.0,
    'korigjim': 1.5,
    'hetim': 2.5
}

# Decision severity weight mapping for Public Procurement Commission (KPP) decisions
DECISION_SEVERITY_WEIGHTS = {
    'VP - Vendim Perfundimtar': 3.0,        # Final decision by commission
    'VPA - Vendim Pezullim Ankandi': 2.5,   # Tender auction suspended
    'UP - Vendim Pezullimi': 2.5,           # Suspension decision
    'VN - Vendim i Ndermjetem': 2.0,       # Intermediate decision
    'VD-P - Vendim Deklarativ/AK ka pranuar plotesisht ankesen': 2.0, # Fully accepted complaint
    'VD - Vendim Deklarativ/AK pranon pjeserisht ankesen': 1.5,      # Partially accepted complaint
    'KF - Korigjim Formulari': 1.0,          # Form correction
    'HDA - Heqje dore nga ankesa': 0.5,      # Complaint withdrawn
    'HP - Heqje Pezullimi': 0.5             # Suspension lifted
}


def detect_brand_or_proprietary_specs(text_normalized: str) -> int:
    """
    Detects potential brand names, trademarks, or restrictive serial/patent numbers in text.
    In procurement, explicitly specifying brand names without 'or equivalent' is a red flag.
    """
    pattern = r'\b(cisco|dell|hp|lenovo|intel|microsoft|oracle|apple|samsung|toyota|mercedes|bmw|bosch|siemens)\b'
    match = re.search(pattern, text_normalized, re.IGNORECASE)
    return 1 if match else 0


def calculate_text_risk_score(text_normalized: str) -> float:
    """
    Calculates weighted risk score based on high-risk keyword frequencies.
    """
    words = text_normalized.split()
    total_score = 0.0
    for word in words:
        if word in RISK_KEYWORDS:
            total_score += RISK_KEYWORDS[word]
    return round(total_score, 2)


def compute_category_value_zscores(records: list) -> list:
    """
    Computes log-transformed category-relative z-scores for tender amounts.
    High value relative to procurement category is a potential risk indicator.
    """
    category_values = defaultdict(list)

    # 1. Group log values by category
    for rec in records:
        cat = rec.get('tender_mainProcurementCategory', 'Other') or 'Other'
        val = rec.get('tender_value_amount_num', 0.0)
        log_val = math.log1p(max(0.0, val))
        category_values[cat].append(log_val)

    # 2. Compute category mean & std
    cat_stats = {}
    for cat, log_vals in category_values.items():
        n = len(log_vals)
        mean = sum(log_vals) / max(1, n)
        variance = sum((x - mean) ** 2 for x in log_vals) / max(1, n)
        std = math.sqrt(variance) if variance > 0 else 1.0
        cat_stats[cat] = (mean, std)

    # 3. Assign z-scores to records
    for rec in records:
        cat = rec.get('tender_mainProcurementCategory', 'Other') or 'Other'
        val = rec.get('tender_value_amount_num', 0.0)
        log_val = math.log1p(max(0.0, val))
        mean, std = cat_stats.get(cat, (0.0, 1.0))
        zscore = round((log_val - mean) / std, 4)
        rec['tender_value_log_zscore'] = zscore
        rec['flag_high_value_outlier'] = 1 if zscore > 2.0 else 0

    return records


def extract_risk_features(records: list) -> list:
    """
    Extracts all textual, legal, financial, and procedural corruption risk features.
    Computes a composite normalized corruption risk proxy score [0.0 - 100.0].
    """
    # First compute financial value z-scores
    records = compute_category_value_zscores(records)

    for rec in records:
        text_norm = rec.get('text_normalized', '')
        word_count = rec.get('text_word_count', 0)

        # 1. Textual Red Flags
        # Abnormally short / non-descriptive title & description
        rec['flag_short_description'] = 1 if word_count < 5 else 0

        # High-risk keywords score
        rec['text_risk_keyword_score'] = calculate_text_risk_score(text_norm)

        # Brand / proprietary spec flag
        rec['flag_brand_specs'] = detect_brand_or_proprietary_specs(text_norm)

        # 2. Complaint & Procedural Red Flags
        complaint_id = rec.get('complaints_id', '') or ''
        rec['complaint_filed_flag'] = 1 if complaint_id.strip() else 0

        dec_type = rec.get('complaints_decision_type', '') or ''
        # Canonicalize decision type lookup key
        dec_type_clean = dec_type.translate(str.maketrans({'ë': 'e', 'ç': 'c'}))
        severity = 0.0
        for k, v in DECISION_SEVERITY_WEIGHTS.items():
            k_clean = k.translate(str.maketrans({'ë': 'e', 'ç': 'c'}))
            if k_clean in dec_type_clean or dec_type_clean in k_clean:
                severity = max(severity, v)
        rec['complaint_severity_score'] = severity

        # 3. Composite Corruption Risk Proxy Score (0 - 100)
        # Weighted aggregate of textual red flags, complaints, and value outliers
        raw_risk = (
            (rec['flag_short_description'] * 15.0) +
            (min(10.0, rec['text_risk_keyword_score']) * 3.0) +
            (rec['flag_brand_specs'] * 20.0) +
            (rec['complaint_filed_flag'] * 25.0) +
            (rec['complaint_severity_score'] * 5.0) +
            (max(0.0, rec['tender_value_log_zscore']) * 5.0)
        )
        
        composite_risk = min(100.0, round(raw_risk, 2))
        rec['composite_risk_score'] = composite_risk

        # Categorical risk level for classification models
        if composite_risk >= 40.0:
            rec['risk_level'] = 'HIGH'
        elif composite_risk >= 20.0:
            rec['risk_level'] = 'MEDIUM'
        else:
            rec['risk_level'] = 'LOW'

    return records


if __name__ == '__main__':
    sample_records = [
        {
            'text_normalized': 'blerje pajisje cisco me skualifikim te operatorit dhe kufizim',
            'text_word_count': 3,
            'tender_mainProcurementCategory': 'Mallra',
            'tender_value_amount_num': 50000000.0,
            'complaints_id': '123/2025',
            'complaints_decision_type': 'VP - Vendim Perfundimtar'
        }
    ]
    featured = extract_risk_features(sample_records)
    print("Featured sample record:", featured[0])
