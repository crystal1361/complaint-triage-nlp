"""TF-IDF + logistic regression baseline: cheap, fast, explainable."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                  sublinear_tf=True, max_features=200_000)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                   random_state=seed)),
    ])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    """Fit on train, return (model, predictions DataFrame with probabilities, timing)."""
    model = build_model(seed)
    t0 = time.perf_counter()
    model.fit(train["text"], train["queue"])
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict_proba(test["text"])
    pred_s = time.perf_counter() - t0

    classes = model.classes_
    out = pd.DataFrame({
        "complaint_id": test["complaint_id"].values,
        "pred": classes[proba.argmax(1)],
        "confidence": proba.max(1),
    })
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]
    timing = {"fit_seconds": fit_s, "predict_ms_per_item": 1000 * pred_s / max(len(test), 1)}
    return model, out, timing


def top_terms(model: Pipeline, n: int = 12) -> dict[str, list[str]]:
    """Highest-weight n-grams per class, for an explainability sanity check."""
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coefs = model.named_steps["clf"].coef_
    return {c: vocab[np.argsort(coefs[i])[::-1][:n]].tolist()
            for i, c in enumerate(model.named_steps["clf"].classes_)}
