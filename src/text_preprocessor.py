"""
Text Normalization and Albanian Language NLP Preprocessing Module.
Handles cleaning, diacritic canonicalization, procurement stopword removal, and text metric calculation.
"""

import re
import string


# Custom list of Albanian general & procurement stopwords
ALBANIAN_STOPWORDS = {
    # General stopwords
    'i', 'e', 'të', 'së', 'më', 'në', 'me', 'për', 'nga', 'dhe', 'ose', 'se', 'ku', 'si',
    'kjo', 'ky', 'këtë', 'këtij', 'këto', 'këtyre', 'ai', 'ajo', 'ata', 'ato', 'tij', 'saj',
    'tyre', 'që', 'ka', 'janë', 'ishte', 'tërt', 'bëjë', 'bëjnë', 'pra', 'edhe', 'jo', 'nuk',
    'do', 'mund', 'tek', 'deri', 'pas', 'para', 'gjatë', 'mbi', 'nën', 'afër',
    
    # Domain-specific public procurement stopwords
    'blerje', 'blerjen', 'blerjes', 'furnizim', 'furnizimi', 'shërbim', 'shërbime', 'shërbimesh',
    'punë', 'punëve', 'kuadër', 'kuadrit', 'projektit', 'objekti', 'objektit', 'pajisje', 'pajisjesh',
    'lot', 'loti', 'nr', 'ref', 'kpp', 'dhe', 'sipas', 'lloji', 'nr.'
}

# Albanian diacritic map (canonicalization for key matching)
DIACRITIC_MAP = str.maketrans({
    'ë': 'e', 'Ë': 'E',
    'ç': 'c', 'Ç': 'C',
    'ä': 'a', 'Ä': 'A',
    'ö': 'o', 'Ö': 'O',
    'ü': 'u', 'Ü': 'U'
})


def clean_text_raw(text: str) -> str:
    """
    Cleans raw text while preserving original letters/diacritics for transformer embedding models.
    """
    if not text:
        return ""
    
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace non-printable / control characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Strip multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def canonicalize_text_albanian(text: str) -> str:
    """
    Normalizes Albanian text by removing diacritics (ë -> e, ç -> c) and converting to lower case.
    Useful for keyword matching, TF-IDF vectorization, and exact string comparisons.
    """
    cleaned = clean_text_raw(text)
    lower = cleaned.lower()
    canonical = lower.translate(DIACRITIC_MAP)
    # Strip non-alphanumeric punctuation except space
    canonical = re.sub(r'[^a-z0-9\s]', ' ', canonical)
    canonical = re.sub(r'\s+', ' ', canonical).strip()
    return canonical


def remove_stopwords(text: str) -> str:
    """
    Removes general and domain-specific stopwords from normalized text.
    """
    tokens = text.split()
    filtered = [t for t in tokens if t not in ALBANIAN_STOPWORDS and len(t) > 1]
    return ' '.join(filtered)


def extract_text_metrics(title: str, description: str) -> dict:
    """
    Calculates key text complexity and length metrics for corruption risk analysis.
    Short or abnormally vague descriptions are often flags for evasive procurement entries.
    """
    combined_raw = f"{title or ''} {description or ''}".strip()
    words = combined_raw.split()
    
    word_count = len(words)
    char_count = len(combined_raw)
    avg_word_len = round(char_count / max(1, word_count), 2)
    
    # Ratio of uppercase characters (excluding spaces)
    non_space_chars = [c for c in combined_raw if not c.isspace()]
    upper_chars = [c for c in non_space_chars if c.isupper()]
    uppercase_ratio = round(len(upper_chars) / max(1, len(non_space_chars)), 4)
    
    # Title vs description similarity / overlap (jaccard)
    title_words = set(canonicalize_text_albanian(title or "").split())
    desc_words = set(canonicalize_text_albanian(description or "").split())
    if title_words and desc_words:
        jaccard_overlap = round(len(title_words & desc_words) / len(title_words | desc_words), 4)
    else:
        jaccard_overlap = 0.0

    return {
        'text_word_count': word_count,
        'text_char_count': char_count,
        'text_avg_word_len': avg_word_len,
        'text_uppercase_ratio': uppercase_ratio,
        'title_desc_jaccard_overlap': jaccard_overlap
    }


def preprocess_text_record(record: dict) -> dict:
    """
    Preprocesses text fields in a single procurement record dictionary.
    Adds 'text_clean', 'text_normalized', 'text_stopwords_removed', and text metrics.
    """
    title = record.get('tender_title', '') or ''
    desc = record.get('tender_description', '') or ''
    
    combined_raw = f"{title} [SEP] {desc}" if desc and desc != title else title
    
    record['text_raw_clean'] = clean_text_raw(combined_raw)
    record['text_normalized'] = canonicalize_text_albanian(combined_raw)
    record['text_stopwords_removed'] = remove_stopwords(record['text_normalized'])
    
    metrics = extract_text_metrics(title, desc)
    record.update(metrics)
    
    return record


if __name__ == '__main__':
    sample = {
        'tender_title': 'BLERJE PAJISJE NË KUADËR TË PROJEKTIT AKKSHI',
        'tender_description': 'Blerje pajisje laboratorike dhe specifikime teknike për shkollën.'
    }
    processed = preprocess_text_record(sample)
    print("Processed sample:", processed)
