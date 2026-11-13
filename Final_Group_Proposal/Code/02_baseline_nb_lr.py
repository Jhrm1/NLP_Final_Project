"""
02_baseline_nb_lr.py
Owner: Group A
Purpose: TF-IDF + Naive Bayes / Logistic Regression baseline.
Run: python 02_baseline_nb_lr.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from utils import load_data, evaluate_model, get_label_frequency_vector, print_metrics


def _json_metrics(metrics):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }


def main():
    print("Loading data ...")
    train_texts, train_labels, mlb = load_data("train")
    test_texts, test_labels, _ = load_data("test")
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}, Labels: {len(mlb.classes_)}")

    freq = get_label_frequency_vector(train_labels)

    # --- Naive Bayes ---
    print("\n=== Naive Bayes ===")
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words="english")),
        ("clf", OneVsRestClassifier(MultinomialNB(alpha=1.0))),
    ])
    nb_pipe.fit(train_texts, train_labels)
    nb_score = nb_pipe.predict_proba(test_texts)
    nb_metrics = evaluate_model(test_labels, nb_score, threshold=0.3, label_freq=freq)
    print_metrics(nb_metrics)

    # --- Logistic Regression ---
    print("\n=== Logistic Regression ===")
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words="english")),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))),
    ])
    lr_pipe.fit(train_texts, train_labels)
    lr_score = lr_pipe.predict_proba(test_texts)
    lr_metrics = evaluate_model(test_labels, lr_score, threshold=0.35, label_freq=freq)
    print_metrics(lr_metrics)

    report = {
        "data": {
            "train_records": len(train_texts),
            "test_records": len(test_texts),
            "num_labels": len(mlb.classes_),
        },
        "models": {
            "naive_bayes": {
                "threshold": 0.3,
                "metrics": _json_metrics(nb_metrics),
            },
            "logistic_regression": {
                "threshold": 0.35,
                "metrics": _json_metrics(lr_metrics),
            },
        },
    }
    output_path = Path(__file__).resolve().parent / "baseline_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")

    print(f"\nBaseline finished. Metrics saved to {output_path}")

if __name__ == "__main__":
    main()