"""Command-line entry point.

    python -m complaint_triage.cli prepare --input data/raw/complaints.csv
    python -m complaint_triage.cli baseline
    python -m complaint_triage.cli llm --mode zeroshot --n 500
    python -m complaint_triage.cli llm --mode fewshot --n 500
    python -m complaint_triage.cli compare
    python -m complaint_triage.cli gold-sample --n 150
    python -m complaint_triage.cli gold-eval --gold gold/annotation_labeled.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import baseline, evaluate, llm, plots
from .evaluate import LABELS

ROOT = Path.cwd()
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def _load_split():
    train = pd.read_csv(PROC / "train.csv.gz", dtype={"complaint_id": str}, parse_dates=["date_received"])
    test = pd.read_csv(PROC / "test.csv.gz", dtype={"complaint_id": str}, parse_dates=["date_received"])
    return train, test


def cmd_prepare(a):
    from .data import load_complaints, split

    df = load_complaints(a.input)
    train, test = split(df, how=a.split)
    PROC.mkdir(parents=True, exist_ok=True)
    cols = ["complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
            "company", "text", "n_words", "queue"]
    cols = [c for c in cols if c in df.columns]
    train[cols].to_csv(PROC / "train.csv.gz", index=False)
    test[cols].to_csv(PROC / "test.csv.gz", index=False)
    profile = {
        "n_raw_rows": df.attrs.get("n_raw"),
        "n_clean": len(df), "n_train": len(train), "n_test": len(test), "split": a.split,
        "date_range": [str(df["date_received"].min().date()), str(df["date_received"].max().date())],
        "train_dates": [str(train["date_received"].min().date()), str(train["date_received"].max().date())],
        "test_dates": [str(test["date_received"].min().date()), str(test["date_received"].max().date())],
        "median_words": float(df["n_words"].median()),
        "queue_share": df["queue"].value_counts(normalize=True).round(4).to_dict(),
        "products": df["product"].value_counts().to_dict() if "product" in df else {},
        "issue_to_queue": (df.groupby(["issue", "queue"]).size().sort_values(ascending=False)
                           .reset_index(name="n").to_dict(orient="records")),
    }
    _save_json(profile, REPORTS / "data_profile.json")
    print(json.dumps({k: v for k, v in profile.items() if k != "issue_to_queue"}, indent=2))


def cmd_baseline(a):
    train, test = _load_split()
    model, preds, timing = baseline.fit_predict(train, test)
    preds.to_csv(REPORTS / "baseline_preds.csv", index=False)
    m = evaluate.classification_metrics(test["queue"], preds["pred"])
    m["timing"] = timing
    m["top_terms"] = baseline.top_terms(model)
    _save_json(m, REPORTS / "baseline_metrics.json")
    FIG.mkdir(parents=True, exist_ok=True)
    plots.confusion(test["queue"], preds["pred"], LABELS,
                    "TF-IDF + logistic regression, most recent 20% of complaints", FIG / "confusion_baseline.png")
    print(f"baseline: accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
          f"({timing['predict_ms_per_item']:.2f} ms/complaint)")


def _llm_sample(test: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """A fixed random sample of the test set, so every LLM run scores the same complaints."""
    return test.sample(min(n, len(test)), random_state=seed)


def cmd_llm(a):
    train, test = _load_split()
    sample = _llm_sample(test, a.n)
    client = llm.make_client(a.provider, a.model)
    examples = llm.pick_examples(train, per_class=a.shots) if a.mode == "fewshot" else None
    print(f"{client.model} {a.mode}: {len(sample)} complaints")
    out = llm.run(sample, client, REPORTS / "llm_cache.jsonl", examples=examples)
    tag = f"llm_{a.mode}_{client.model}".replace("/", "_")
    out.to_csv(REPORTS / f"{tag}_preds.csv", index=False)
    ok = out["parse_ok"].astype(bool)
    print(f"parsed {ok.mean():.1%}; accuracy on parsed = "
          f"{(out.loc[ok, 'queue'].values == sample.loc[ok.values, 'queue'].values).mean():.3f}")


def cmd_compare(a):
    _, test = _load_split()
    base = pd.read_csv(REPORTS / "baseline_preds.csv", dtype={"complaint_id": str})
    runs = sorted(REPORTS.glob("llm_*_preds.csv"))
    if not runs:
        raise SystemExit("No LLM predictions yet - run the `llm` command first.")
    results, table, sweeps = {}, [], {}
    for path in runs:
        name = path.stem.removesuffix("_preds")
        lp = pd.read_csv(path, dtype={"complaint_id": str})
        ids = lp["complaint_id"]
        truth = test.set_index("complaint_id").loc[ids, "queue"].values
        b = base.set_index("complaint_id").loc[ids].reset_index()
        lp = lp.reset_index(drop=True)
        if "TF-IDF baseline" not in results:
            results["TF-IDF baseline"] = evaluate.classification_metrics(truth, b["pred"])
            table.append({"model": "TF-IDF baseline", "n": len(ids),
                          **{k: results["TF-IDF baseline"][k] for k in ("accuracy", "macro_f1")},
                          "parse_fail_rate": 0.0})
        # A parse failure counts as wrong: in production it would need a fallback.
        pred = lp["queue"].where(lp["parse_ok"].astype(bool), "PARSE_FAIL")
        results[name] = evaluate.classification_metrics(truth, pred)
        cost = evaluate.cost_summary(lp)
        table.append({"model": name, "n": len(ids),
                      **{k: results[name][k] for k in ("accuracy", "macro_f1")}, **cost})
        sweeps[name] = evaluate.hybrid_sweep(truth, b, lp)
        sweeps[name].to_csv(REPORTS / f"hybrid_{name}.csv", index=False)
    pd.DataFrame(table).to_csv(REPORTS / "comparison.csv", index=False)
    _save_json(results, REPORTS / "comparison_metrics.json")
    FIG.mkdir(parents=True, exist_ok=True)
    plots.per_class_f1(results, LABELS, FIG / "f1_by_queue.png",
                       title="F1 by queue on the same held-out sample")
    best = max(sweeps, key=lambda k: sweeps[k]["accuracy"].max())
    plots.hybrid_curve(sweeps[best], FIG / "hybrid_tradeoff.png",
                       title=f"Confidence routing: baseline first, {best} when unsure")
    print(pd.DataFrame(table).to_string(index=False))


def cmd_gold_sample(a):
    _, test = _load_split()
    gold = evaluate.make_gold_sample(test, n=a.n)
    out = ROOT / "gold" / "annotation_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False)
    print(f"wrote {len(gold)} rows to {out}; fill human_* columns, save as annotation_labeled.csv")


def cmd_gold_eval(a):
    gold = pd.read_csv(a.gold, dtype=str).fillna("")
    preds = {"TF-IDF baseline": pd.read_csv(REPORTS / "baseline_preds.csv", dtype={"complaint_id": str})}
    for path in sorted(REPORTS.glob("llm_*_preds.csv")):
        lp = pd.read_csv(path, dtype={"complaint_id": str})
        lp["pred"] = lp["queue"]
        preds[path.stem.removesuffix("_preds")] = lp
    res = evaluate.evaluate_gold(gold, preds)
    _save_json(res, REPORTS / "gold_eval.json")
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=2, default=float))
    for name, m in res["models"].items():
        print(f"{name}: accuracy vs human = {m['accuracy']:.3f} (n={m['n']})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="complaint_triage")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("prepare"); s.add_argument("--input", required=True)
    s.add_argument("--split", default="time", choices=["time", "random"]); s.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("baseline"); s.set_defaults(fn=cmd_baseline)
    s = sub.add_parser("llm"); s.add_argument("--mode", default="zeroshot", choices=["zeroshot", "fewshot"])
    s.add_argument("--n", type=int, default=500); s.add_argument("--shots", type=int, default=2)
    s.add_argument("--provider"); s.add_argument("--model"); s.set_defaults(fn=cmd_llm)
    s = sub.add_parser("compare"); s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("gold-sample"); s.add_argument("--n", type=int, default=150); s.set_defaults(fn=cmd_gold_sample)
    s = sub.add_parser("gold-eval"); s.add_argument("--gold", required=True); s.set_defaults(fn=cmd_gold_eval)
    a = p.parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
