"""Fit a classifier on frozen Sentence-BERT embeddings and processed risk labels."""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report


DEFAULT_SBERT = "paraphrase-multilingual-MiniLM-L12-v2"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def train(csv_path, output_dir, sbert_name=DEFAULT_SBERT):
    data = pd.read_csv(csv_path, encoding="utf-8")
    required = {"tender_description", "risk_level"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    valid = data["tender_description"].notna() & data["risk_level"].notna()
    texts = data.loc[valid, "tender_description"].astype(str).str.strip().tolist()
    labels = data.loc[valid, "risk_level"].astype(str).str.strip().to_numpy()
    nonempty = np.array([bool(t) for t in texts])
    texts = [text for text, keep in zip(texts, nonempty) if keep]
    labels = labels[nonempty]
    if len(texts) < 10 or len(np.unique(labels)) < 2:
        raise ValueError("Need at least 10 non-empty descriptions and two risk_level classes.")

    print(f"Encoding {len(texts):,} descriptions with Sentence-BERT: {sbert_name}")
    # Use the local Hugging Face cache when present; training can run offline.
    encoder = SentenceTransformer(sbert_name, local_files_only=True)
    embeddings = encoder.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    all_indices = np.arange(len(y))
    dev_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=2025, stratify=y,
    )
    train_indices, val_indices = train_test_split(
        dev_indices, test_size=0.2, random_state=42, stratify=y[dev_indices],
    )

    # Select class weighting on validation data, then report once on an untouched test split.
    candidates = []
    for class_weight in (None, 'balanced'):
        candidate = LogisticRegression(max_iter=2000, random_state=42, class_weight=class_weight)
        candidate.fit(embeddings[train_indices], y[train_indices])
        val_pred = candidate.predict(embeddings[val_indices])
        candidates.append((
            f1_score(y[val_indices], val_pred, average='macro'),
            balanced_accuracy_score(y[val_indices], val_pred),
            class_weight,
        ))
    validation_macro_f1, validation_balanced_accuracy, selected_weight = max(candidates, key=lambda row: (row[0], row[1]))
    test_model = LogisticRegression(max_iter=2000, random_state=42, class_weight=selected_weight)
    test_model.fit(embeddings[dev_indices], y[dev_indices])
    test_pred = test_model.predict(embeddings[test_indices])
    test_metrics = {
        'rows': int(len(test_indices)),
        'accuracy': float(accuracy_score(y[test_indices], test_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y[test_indices], test_pred)),
        'macro_f1': float(f1_score(y[test_indices], test_pred, average='macro')),
        'classification_report': classification_report(
            y[test_indices], test_pred, labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_, output_dict=True, zero_division=0,
        ),
    }
    # The serving classifier uses all labeled rows after the model choice is locked.
    classifier = LogisticRegression(max_iter=2000, random_state=42, class_weight=selected_weight)
    classifier.fit(embeddings, y)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(classifier, os.path.join(output_dir, "classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    class_counts = pd.Series(labels).value_counts().to_dict()
    metadata = {
        "sbert_model": sbert_name,
        "target_column": "risk_level",
        "training_rows": len(texts),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "classes": [str(c) for c in label_encoder.classes_],
        "label_note": "These are the processed dataset's rule-generated risk categories, not confirmed corruption outcomes.",
        "method": "Frozen Sentence-BERT embeddings with a multinomial logistic-regression classifier.",
        "class_weight": selected_weight,
        "validation_selection": {
            "macro_f1": float(validation_macro_f1),
            "balanced_accuracy": float(validation_balanced_accuracy),
            "candidate_scores": {
                "unweighted_macro_f1": float(candidates[0][0]),
                "balanced_macro_f1": float(candidates[1][0]),
            },
        },
        "held_out_test_metrics": test_metrics,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved Sentence-BERT classifier artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Processed CSV with tender_description and risk_level columns")
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_DIR, "models", "sbert"))
    args = parser.parse_args()
    train(args.csv_path, args.output_dir, args.sbert_model)
