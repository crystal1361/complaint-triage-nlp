"""Load and clean the CFPB Consumer Complaint export, and build train/test splits."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .taxonomy import map_to_queue

# CFPB's website CSV uses Title Case headers; the API JSON uses snake_case.
_COLUMN_MAP = {
    "complaint id": "complaint_id",
    "date received": "date_received",
    "product": "product",
    "sub-product": "sub_product",
    "issue": "issue",
    "sub-issue": "sub_issue",
    "consumer complaint narrative": "narrative",
    "complaint_what_happened": "narrative",
    "company": "company",
    "state": "state",
    "tags": "tags",
    "company response to consumer": "company_response",
    "submitted via": "submitted_via",
}

_REDACTION = re.compile(r"\b[Xx]{2,}(?:[/\-][Xx]{2,})*\b")
_WS = re.compile(r"\s+")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace("_", " ")
        key_snake = col.strip().lower()
        if key in _COLUMN_MAP:
            rename[col] = _COLUMN_MAP[key]
        elif key_snake in _COLUMN_MAP:
            rename[col] = _COLUMN_MAP[key_snake]
    return df.rename(columns=rename)


def clean_text(text: str) -> str:
    """Collapse CFPB's XXXX redactions into one token and normalize whitespace."""
    text = _REDACTION.sub("[REDACTED]", str(text))
    text = re.sub(r"(\[REDACTED\]\s*){2,}", "[REDACTED] ", text)
    return _WS.sub(" ", text).strip()


def load_complaints(path: str | Path, min_words: int = 20) -> pd.DataFrame:
    """Read a CFPB CSV export and return cleaned rows with a `queue` label.

    Drops rows without a narrative, very short narratives, and exact duplicate
    narratives (the same complaint is sometimes filed against several companies).
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = _normalize_columns(df)
    required = {"complaint_id", "date_received", "issue", "narrative"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {sorted(missing)}; got {list(df.columns)}")
    if "sub_issue" not in df.columns:
        df["sub_issue"] = ""

    n_raw = len(df)
    df = df.dropna(subset=["narrative", "issue"]).copy()
    df["text"] = df["narrative"].map(clean_text)
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"] >= min_words]
    df = df.drop_duplicates(subset="text")
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df = df.dropna(subset=["date_received"])
    df["queue"] = [map_to_queue(i, s) for i, s in zip(df["issue"], df["sub_issue"].fillna(""))]
    df = df.sort_values("date_received").reset_index(drop=True)
    df.attrs["n_raw"] = n_raw
    return df


def split(df: pd.DataFrame, how: str = "time", test_frac: float = 0.2, seed: int = 42):
    """Return (train, test).

    `time` (default) holds out the most recent complaints, which mirrors
    deployment: the model is trained on the past and scored on what comes next.
    `random` is a stratified random split, kept for comparison.
    """
    if how == "time":
        cut = int(len(df) * (1 - test_frac))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    if how == "random":
        from sklearn.model_selection import train_test_split

        return train_test_split(df, test_size=test_frac, stratify=df["queue"], random_state=seed)
    raise ValueError(f"unknown split: {how}")
