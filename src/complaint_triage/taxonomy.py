"""Routing taxonomy: maps CFPB's consumer-selected (issue, sub-issue) to service queues.

CFPB labels are chosen by the consumer from a dropdown when filing, so they are
noisy and sometimes more about how the consumer feels than what happened. We map
them into a small set of operational queues a client-service org would actually
route to. Rules are ordered: the first match wins, so fraud is checked first
(misrouting fraud is the costliest error).
"""
from __future__ import annotations

import re

QUEUES: dict[str, str] = {
    "FRAUD_UNAUTHORIZED": (
        "Fraud, scams, identity theft, or transactions/accounts the client says "
        "they did not authorize."
    ),
    "FEES_OVERDRAFT": (
        "Disputes about fees, overdraft or NSF charges, interest, or problems "
        "caused by low funds."
    ),
    "ACCESS_OPEN_CLOSE": (
        "Opening, closing, freezing, or being locked out of an account, card, "
        "or mobile wallet."
    ),
    "TRANSACTION_FUNDS": (
        "Deposits, withdrawals, transfers, or payments that were delayed, held, "
        "wrong, missing, or not processed as instructed (no fraud alleged)."
    ),
    "SERVICE_OTHER": (
        "Customer-service quality, misleading marketing, communication problems, "
        "or anything that fits none of the other queues."
    ),
}

# (pattern on lower-cased "issue | sub-issue", queue). First match wins.
_RULES: list[tuple[str, str]] = [
    (r"fraud|scam|unauthori[sz]ed|not authori[sz]ed|identity theft|"
     r"without (your |my )?consent|didn't authorize|did not authorize", "FRAUD_UNAUTHORIZED"),
    (r"\bfees?\b|funds being low|overdraft|non-sufficient|\binterest\b",
     "FEES_OVERDRAFT"),
    (r"advertis|disclosure|marketing|customer service|other service problem",
     "SERVICE_OTHER"),
    (r"closing|closed|opening an account|open an account|mobile wallet|"
     r"accessing (the |your )?account|\baccess\b|locked|frozen|freez",
     "ACCESS_OPEN_CLOSE"),
    (r"deposit|withdraw|not available when promised|transaction|wrong amount|"
     r"exchange rate|disbursed|cashing a check|banking error|trouble (getting|sending)|"
     r"transfer|payment|debit or atm card|charging your account|accessing funds|"
     r"adding money|lost or stolen (check|money order)",
     "TRANSACTION_FUNDS"),
]
_COMPILED = [(re.compile(p), q) for p, q in _RULES]


def map_to_queue(issue: str | None, sub_issue: str | None = None) -> str:
    """Map a CFPB issue/sub-issue pair to a routing queue."""
    text = f"{issue or ''} | {sub_issue or ''}".lower()
    for pattern, queue in _COMPILED:
        if pattern.search(text):
            return queue
    return "SERVICE_OTHER"


def queue_descriptions() -> str:
    return "\n".join(f"- {q}: {d}" for q, d in QUEUES.items())
