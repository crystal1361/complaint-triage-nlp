"""Rule-based escalation flags: the transparent baseline the LLM flags are judged against.

These flags decide whether a complaint skips the normal queue. They are scored
on the human-annotated gold sample, never on CFPB labels (CFPB has no such field).
"""
from __future__ import annotations

import re

FLAG_DEFINITIONS: dict[str, str] = {
    "fraud_or_scam": "Client reports fraud, a scam, identity theft, or money taken without authorization.",
    "vulnerable_client": "Client is elderly, disabled, seriously ill, recently bereaved, or otherwise at heightened risk.",
    "legal_threat": "Client threatens a lawsuit, mentions an attorney, or says they will go to a regulator or court "
                    "(filing this CFPB complaint alone does not count).",
    "financial_hardship": "Client describes being unable to pay rent, bills, food, or medicine because of the problem.",
}

_PATTERNS: dict[str, re.Pattern] = {
    "fraud_or_scam": re.compile(
        r"\bfraud|\bscam|identity theft|stolen|unauthori[sz]ed|didn'?t authori[sz]e|"
        r"did not authori[sz]e|hacked|phishing", re.I),
    "vulnerable_client": re.compile(
        r"\belderly\b|senior citizen|\b(7|8|9)\d[- ]year[- ]old|disab(led|ility)|"
        r"social security (disability|income)|\bssi\b|cancer|hospital|passed away|"
        r"\bdeceased\b|widow", re.I),
    "legal_threat": re.compile(
        r"\blawyer|\battorney\b|\bsue\b|\bsuing\b|lawsuit|legal action|small claims|"
        r"attorney general|\bfdic\b|\bocc\b|\bfinra\b|\bsec\b|better business bureau|\bbbb\b", re.I),
    "financial_hardship": re.compile(
        r"(can'?t|cannot|unable to|couldn'?t) (pay|afford)|\brent\b|evict|groceries|"
        r"\bfood\b|medication|overdrawn|late fees? on my|behind on", re.I),
}


def rule_flags(text: str) -> dict[str, bool]:
    return {name: bool(p.search(text)) for name, p in _PATTERNS.items()}
