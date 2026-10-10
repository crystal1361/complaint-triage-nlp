"""Metrics, cost accounting, hybrid routing, and gold-set evaluation."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .flags import FLAG_DEFINITIONS, rule_flags
from .taxonomy import QUEUES

LABELS = list(QUEUES)


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True,
                                           zero_division=0),
    }


def price_per_million() -> tuple[float | None, float | None]:
    """USD per 1M input/output tokens, from env. Left unset = cost is reported in tokens only."""
    def get(name):
        v = os.getenv(name)
        return float(v) if v else None
    return get("LLM_PRICE_IN_PER_MTOK"), get("LLM_PRICE_OUT_PER_MTOK")


def cost_summary(llm: pd.DataFrame) -> dict:
    p_in, p_out = price_per_million()
    n = len(llm)
    tin, tout = llm["input_tokens"].sum(), llm["output_tokens"].sum()
    out = {
        "n_calls": n,
        "parse_fail_rate": float((~llm["parse_ok"].astype(bool)).mean()) if n else None,
        "avg_input_tokens": tin / n if n else None,
        "avg_output_tokens": tout / n if n else None,
        "median_latency_s": float(llm["latency_s"].median()) if n else None,
    }
    if p_in is not None and p_out is not None and n:
        per_call = (tin * p_in + tout * p_out) / 1e6 / n
        out["usd_per_1k_complaints"] = 1000 * per_call
    return out


def hybrid_sweep(y_true: pd.Series, base: pd.DataFrame, llm: pd.DataFrame,
                 thresholds=None) -> pd.DataFrame:
    """Route to the LLM only when the cheap model's confidence is below a threshold.

    All three inputs must be aligned on complaint_id. A failed LLM parse falls back
    to the baseline prediction. Threshold 0 = baseline only; 1.01 = LLM for everything.
    """
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0, 1.01, 0.05), 2).tolist() + [1.01]
    llm_ok = llm["parse_ok"].astype(bool).values
    llm_pred = np.where(llm_ok, llm["queue"].fillna("").values, base["pred"].values)
    rows = []
    for t in thresholds:
        to_llm = base["confidence"].values < t
        pred = np.where(to_llm, llm_pred, base["pred"].values)
        rows.append({
            "threshold": t,
            "share_to_llm": float(to_llm.mean()),
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- gold set

GOLD_COLUMNS = ["human_queue"] + [f"human_{k}" for k in FLAG_DEFINITIONS] + ["notes"]


def make_gold_sample(test: pd.DataFrame, n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Stratified sample of the test set for human review, with blank label columns.

    Model predictions are deliberately left out so the reviewer is not anchored by them.
    """
    per = max(1, n // test["queue"].nunique())
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in test.groupby("queue")]
    s = pd.concat(parts).sample(frac=1, random_state=seed)
    out = s[["complaint_id", "product", "issue", "sub_issue", "queue", "text"]].rename(
        columns={"queue": "cfpb_mapped_queue", "issue": "cfpb_issue", "sub_issue": "cfpb_sub_issue"})
    for c in GOLD_COLUMNS:
        out[c] = ""
    return out


def _truthy(x) -> bool:
    return str(x).strip().lower() in {"1", "y", "yes", "true", "t", "x"}


def evaluate_gold(gold: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict:
    """Score CFPB labels, models, and flag detectors against human labels.

    `preds` maps a model name to a frame with complaint_id + pred (and flag_* for LLMs).
    """
    g = gold[gold["human_queue"].astype(str).str.strip().isin(LABELS)].copy()
    g["complaint_id"] = g["complaint_id"].astype(str)
    res = {"n_labeled": len(g),
           "cfpb_label_agreement": float((g["cfpb_mapped_queue"] == g["human_queue"]).mean()) if len(g) else None,
           "models": {}, "flags": {}}
    for name, p in preds.items():
        p = p.copy()
        p["complaint_id"] = p["complaint_id"].astype(str)
        m = g.merge(p, on="complaint_id", how="inner")
        if len(m):
            res["models"][name] = classification_metrics(m["human_queue"], m["pred"])

    rules = pd.DataFrame([rule_flags(t) for t in g["text"]], index=g.index)
    for k in FLAG_DEFINITIONS:
        truth = g[f"human_{k}"].map(_truthy)
        entry = {"prevalence": float(truth.mean()) if len(g) else None,
                 "rules": _prf(truth, rules[k])}
        for name, p in preds.items():
            col = f"flag_{k}"
            if col in p.columns:
                m = g[["complaint_id"]].merge(p[["complaint_id", col]].astype({"complaint_id": str}),
                                              on="complaint_id", how="left")
                entry[name] = _prf(truth.values, m[col].fillna(False).astype(bool).values)
        res["flags"][k] = entry
    return res


def _prf(truth, pred) -> dict:
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}
