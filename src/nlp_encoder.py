"""
NLP Encoder and TF-IDF Feature Extraction Module.
Vectorizes cleaned procurement text using TF-IDF (unigrams + bigrams) and exports sparse matrix representations.
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


def build_tfidf_features(records: list, max_features: int = 1000, text_key: str = 'text_stopwords_removed'):
    """
    Fits a TfidfVectorizer on the preprocessed procurement text and transforms documents.
    Returns (vectorizer, tfidf_matrix, feature_names).
    """
    corpus = [rec.get(text_key, '') or '' for rec in records]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    return vectorizer, tfidf_matrix, feature_names


def save_tfidf_artifacts(vectorizer, tfidf_matrix, output_dir: str):
    """
    Saves TF-IDF vectorizer and sparse matrix artifacts to disk.
    """
    matrix_path = f"{output_dir}/tfidf_features.npz"
    vectorizer_path = f"{output_dir}/tfidf_vectorizer.pkl"

    save_npz(matrix_path, tfidf_matrix)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    return matrix_path, vectorizer_path


if __name__ == '__main__':
    sample_records = [
        {'text_stopwords_removed': 'blerje pajisje akkshi laboratorium'},
        {'text_stopwords_removed': 'furnizim leter zyre sherbime kompjuterike'}
    ]
    vec, mat, names = build_tfidf_features(sample_records, max_features=10)
    print("TF-IDF Matrix shape:", mat.shape)
    print("Feature names:", names)
