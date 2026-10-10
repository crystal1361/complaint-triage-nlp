"""Tiny synthetic CFPB-shaped CSV for tests only. Real results come from the CFPB export."""
import random

import pandas as pd
import pytest

TEMPLATES = {
    ("Fraud or scam", ""): "Someone committed fraud on my account and a scammer took {amt} without my permission. The bank refuses to reimburse me for this fraud claim even after I filed a police report and called many times.",
    ("Problem caused by your funds being low", "Overdraft, savings, or protection"): "The bank charged me an overdraft fee of {amt} three times in one day for a single pending transaction. I asked them to waive the fee but they said fee policy cannot change and refused my request.",
    ("Closing an account", "Company closed your account"): "My checking account was closed without any warning and they will not tell me why. My direct deposit of {amt} is stuck and I have been locked out of the closed account for weeks now.",
    ("Managing an account", "Deposits and withdrawals"): "I made a mobile deposit of {amt} and the bank placed a hold on the deposit for ten business days. The deposit still has not posted and customer service keeps giving me different dates every time.",
    ("Other service problem", ""): "The representative was rude and hung up on me twice. Nobody returns my calls or emails and the website gives conflicting information about what I need to do next with this request of {amt}.",
}


@pytest.fixture
def raw_csv(tmp_path):
    rng = random.Random(0)
    rows = []
    for i in range(300):
        (issue, sub), tpl = rng.choice(list(TEMPLATES.items()))
        rows.append({
            "Date received": f"2025-0{1 + i % 3}-{1 + i % 28:02d}",
            "Product": "Checking or savings account",
            "Sub-product": "Checking account",
            "Issue": issue, "Sub-issue": sub,
            "Consumer complaint narrative": tpl.format(amt=f"${rng.randint(10, 5000)}.00") + f" Ref XXXX/XXXX {i}.",
            "Company": "BANK", "Complaint ID": str(1000 + i),
        })
    rows.append({**rows[0], "Complaint ID": "dup"})  # exact duplicate narrative
    rows.append({**rows[1], "Complaint ID": "short", "Consumer complaint narrative": "too short"})
    path = tmp_path / "complaints.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
