import json

import pandas as pd

from complaint_triage import baseline, evaluate, llm
from complaint_triage.data import clean_text, load_complaints, split
from complaint_triage.flags import rule_flags
from complaint_triage.taxonomy import QUEUES, map_to_queue


def test_taxonomy_priorities():
    assert map_to_queue("Fraud or scam") == "FRAUD_UNAUTHORIZED"
    # fraud wins over transactions when both appear
    assert map_to_queue("Unauthorized transactions or other transaction problem") == "FRAUD_UNAUTHORIZED"
    assert map_to_queue("Problem caused by your funds being low", "Overdraft, savings, or protection") == "FEES_OVERDRAFT"
    assert map_to_queue("Managing an account", "Fee problem") == "FEES_OVERDRAFT"
    assert map_to_queue("Closing an account", "Company closed your account") == "ACCESS_OPEN_CLOSE"
    assert map_to_queue("Managing an account", "Deposits and withdrawals") == "TRANSACTION_FUNDS"
    assert map_to_queue("Money was not available when promised") == "TRANSACTION_FUNDS"
    assert map_to_queue("Other service problem") == "SERVICE_OTHER"
    assert map_to_queue("Problem with a lender or other company charging your account",
                        "Transaction was not authorized") == "FRAUD_UNAUTHORIZED"
    assert map_to_queue("Opening an account", "Account opened without my consent or knowledge") == "FRAUD_UNAUTHORIZED"
    assert map_to_queue("Opening an account", "Didn't receive terms that were advertised") == "SERVICE_OTHER"
    assert map_to_queue("Trouble accessing funds in your mobile or digital wallet") == "TRANSACTION_FUNDS"


def test_clean_text_collapses_redactions():
    assert clean_text("On XX/XX/XXXX I called XXXX XXXX   again") == "On [REDACTED] I called [REDACTED] again"


def test_load_drops_short_and_duplicates(raw_csv):
    df = load_complaints(raw_csv)
    assert "short" not in set(df["complaint_id"])
    assert df["text"].is_unique
    assert set(df["queue"]) <= set(QUEUES)
    assert df["date_received"].is_monotonic_increasing


def test_time_split_holds_out_latest(raw_csv):
    df = load_complaints(raw_csv)
    train, test = split(df, "time")
    assert train["date_received"].max() <= test["date_received"].min()
    assert len(train) + len(test) == len(df)


def test_baseline_learns_templates(raw_csv):
    train, test = split(load_complaints(raw_csv), "random")
    _, preds, _ = baseline.fit_predict(train, test)
    assert (preds["pred"].values == test["queue"].values).mean() > 0.9
    assert preds["confidence"].between(0, 1).all()


def test_parse_response_validates():
    good = '```json\n{"queue": "FEES_OVERDRAFT", "confidence": 0.8, "amount_usd": "35", "flags": {"fraud_or_scam": false}}\n```'
    p = llm.parse_response(good)
    assert p["parse_ok"] and p["queue"] == "FEES_OVERDRAFT" and p["amount_usd"] == 35.0
    assert p["flag_legal_threat"] is False
    assert llm.parse_response('{"queue": "MADE_UP"}')["parse_ok"] is False
    assert llm.parse_response("not json")["parse_ok"] is False


def test_llm_run_caches(raw_csv, tmp_path):
    train, test = split(load_complaints(raw_csv), "time")
    sample = test.head(10)
    fake = llm.FakeLLM()
    cache = tmp_path / "cache.jsonl"
    out1 = llm.run(sample, fake, cache, log_every=0)
    assert len(fake.calls) == 10 and out1["parse_ok"].all()
    out2 = llm.run(sample, fake, cache, log_every=0)
    assert len(fake.calls) == 10  # second run served from cache
    pd.testing.assert_frame_equal(out1, out2)
    # few-shot uses a separate cache key and includes examples in the prompt
    ex = llm.pick_examples(train, per_class=1)
    llm.run(sample.head(2), fake, cache, examples=ex, log_every=0)
    assert len(fake.calls) == 12 and "Labeled examples" in fake.calls[-1]
    assert all(json.loads(l)["key"] for l in cache.read_text().splitlines())


def test_hybrid_sweep_endpoints():
    y = pd.Series(["A_", "B_", "A_", "B_"]).map({"A_": "FEES_OVERDRAFT", "B_": "SERVICE_OTHER"})
    base = pd.DataFrame({"pred": ["FEES_OVERDRAFT", "FEES_OVERDRAFT", "FEES_OVERDRAFT", "SERVICE_OTHER"],
                         "confidence": [0.9, 0.4, 0.95, 0.3]})
    llm_df = pd.DataFrame({"queue": list(y), "parse_ok": [True] * 4})
    s = evaluate.hybrid_sweep(y, base, llm_df, thresholds=[0, 0.5, 1.01])
    assert s.loc[0, "share_to_llm"] == 0 and s.loc[0, "accuracy"] == 0.75
    assert s.loc[1, "share_to_llm"] == 0.5 and s.loc[1, "accuracy"] == 1.0
    assert s.loc[2, "share_to_llm"] == 1.0


def test_rule_flags():
    f = rule_flags("My 82-year-old mother was scammed and we hired an attorney. She can't pay rent.")
    assert f == {"fraud_or_scam": True, "vulnerable_client": True, "legal_threat": True, "financial_hardship": True}
    assert not any(rule_flags("The app is slow when I log in.").values())


def test_gold_eval(raw_csv):
    _, test = split(load_complaints(raw_csv), "time")
    gold = evaluate.make_gold_sample(test, n=20)
    gold["human_queue"] = gold["cfpb_mapped_queue"]
    gold.loc[gold.index[0], "human_queue"] = "SERVICE_OTHER" if gold.iloc[0]["cfpb_mapped_queue"] != "SERVICE_OTHER" else "FEES_OVERDRAFT"
    gold["human_fraud_or_scam"] = gold["text"].str.contains("fraud").map({True: "y", False: ""})
    preds = {"m": pd.DataFrame({"complaint_id": gold["complaint_id"], "pred": gold["cfpb_mapped_queue"]})}
    res = evaluate.evaluate_gold(gold, preds)
    assert res["n_labeled"] == len(gold)
    assert res["cfpb_label_agreement"] < 1.0
    assert res["flags"]["fraud_or_scam"]["rules"]["recall"] == 1.0
