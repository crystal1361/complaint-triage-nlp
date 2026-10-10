"""LLM triage: zero-shot or few-shot classification plus structured extraction.

One call per complaint returns JSON with the queue, escalation flags, the dollar
amount in dispute, and a short rationale. Every response is cached to disk keyed
by (model, prompt version, complaint id), so re-running evaluation costs nothing
and results are reproducible.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from .flags import FLAG_DEFINITIONS
from .taxonomy import QUEUES, queue_descriptions

PROMPT_VERSION = "v1"
MAX_WORDS = 600  # truncate very long narratives to bound cost

SYSTEM = (
    "You triage consumer complaints for a bank and brokerage client-service team. "
    "Read the complaint and return only a JSON object, no prose."
)


def _schema_text() -> str:
    flags = "\n".join(f'    "{k}": true|false,   // {v}' for k, v in FLAG_DEFINITIONS.items())
    queues = " | ".join(f'"{q}"' for q in QUEUES)
    return (
        "{\n"
        f'  "queue": {queues},\n'
        '  "confidence": number between 0 and 1,\n'
        '  "amount_usd": number or null,   // main dollar amount in dispute, if stated\n'
        '  "flags": {\n'
        f"{flags}\n"
        "  },\n"
        '  "rationale": "one sentence"\n'
        "}"
    )


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " [...]"


def build_prompt(text: str, examples: list[tuple[str, str]] | None = None) -> str:
    parts = [
        "Queues:",
        queue_descriptions(),
        "",
        "If several queues apply, pick the one the client most needs resolved. "
        "Fraud or unauthorized activity always takes priority.",
        "",
        "Return JSON with exactly this shape:",
        _schema_text(),
    ]
    if examples:
        parts += ["", "Labeled examples:"]
        for ex_text, ex_queue in examples:
            parts += [f"<complaint>{truncate(ex_text, 150)}</complaint>", f"queue: {ex_queue}"]
    parts += ["", f"<complaint>{truncate(text)}</complaint>"]
    return "\n".join(parts)


def pick_examples(train: pd.DataFrame, per_class: int = 2, seed: int = 42) -> list[tuple[str, str]]:
    """Few-shot examples: short, typical complaints from each queue in the training set."""
    out = []
    for queue, grp in train.groupby("queue"):
        mid = grp[(grp["n_words"] >= 40) & (grp["n_words"] <= 120)]
        pool = mid if len(mid) >= per_class else grp
        for t in pool.sample(min(per_class, len(pool)), random_state=seed)["text"]:
            out.append((t, queue))
    return out


# ---------------------------------------------------------------- clients


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, prompt: str) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model, max_tokens=400, temperature=0, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens,
                           time.perf_counter() - t0)


class OpenAIClient:
    def __init__(self, model: str):
        import openai

        self.model = model
        self._client = openai.OpenAI()

    def complete(self, system: str, prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        r = self._client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return LLMResponse(r.choices[0].message.content or "", r.usage.prompt_tokens,
                           r.usage.completion_tokens, time.perf_counter() - t0)


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests: keyword routing, no network."""

    model: str = "fake"
    calls: list[str] = field(default_factory=list)

    def complete(self, system: str, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        body = prompt.rsplit("<complaint>", 1)[-1].lower()
        queue = "SERVICE_OTHER"
        for q, kw in [("FRAUD_UNAUTHORIZED", "fraud"), ("FEES_OVERDRAFT", "fee"),
                      ("ACCESS_OPEN_CLOSE", "closed"), ("TRANSACTION_FUNDS", "deposit")]:
            if kw in body:
                queue = q
                break
        flags = {k: (k == "fraud_or_scam" and "fraud" in body) for k in FLAG_DEFINITIONS}
        payload = {"queue": queue, "confidence": 0.9, "amount_usd": None,
                   "flags": flags, "rationale": "keyword stub"}
        return LLMResponse("```json\n" + json.dumps(payload) + "\n```", 100, 30, 0.0)


def make_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    model = model or os.getenv("LLM_MODEL")
    if provider == "anthropic":
        return AnthropicClient(model or "claude-haiku-5-5")
    if provider == "openai":
        if not model:
            raise ValueError("Set LLM_MODEL for the openai provider")
        return OpenAIClient(model)
    if provider == "fake":  # offline smoke test of the whole pipeline
        return FakeLLM()
    raise ValueError(f"unknown provider {provider}")


# ---------------------------------------------------------------- parsing + run

_JSON = re.compile(r"\{.*\}", re.S)


def parse_response(text: str) -> dict:
    """Extract and validate the JSON object. Invalid output becomes a parse failure, not a guess."""
    m = _JSON.search(text)
    if not m:
        return {"parse_ok": False}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"parse_ok": False}
    queue = obj.get("queue")
    if queue not in QUEUES:
        return {"parse_ok": False, "raw_queue": queue}
    flags = obj.get("flags") or {}
    out = {"parse_ok": True, "queue": queue,
           "confidence": _to_float(obj.get("confidence")),
           "amount_usd": _to_float(obj.get("amount_usd")),
           "rationale": str(obj.get("rationale", ""))[:300]}
    for k in FLAG_DEFINITIONS:
        out[f"flag_{k}"] = bool(flags.get(k, False))
    return out


def _to_float(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def run(df: pd.DataFrame, client: LLMClient, cache_path: str | Path,
        examples: list[tuple[str, str]] | None = None, retries: int = 3,
        log_every: int = 25) -> pd.DataFrame:
    """Classify each row, reading from and appending to a JSONL cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "fewshot" if examples else "zeroshot"
    key_prefix = f"{client.model}|{PROMPT_VERSION}|{mode}|"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec

    rows = []
    with cache_path.open("a") as fh:
        for i, (cid, text) in enumerate(zip(df["complaint_id"], df["text"])):
            key = key_prefix + str(cid)
            if key not in cache:
                prompt = build_prompt(text, examples)
                resp, err = None, None
                for attempt in range(retries):
                    try:
                        resp = client.complete(SYSTEM, prompt)
                        break
                    except Exception as e:  # network / rate limit: back off and retry
                        err = repr(e)
                        time.sleep(2 ** attempt)
                rec = {"key": key, "complaint_id": str(cid), "model": client.model, "mode": mode,
                       "error": None if resp else err}
                if resp:
                    rec.update(raw=resp.text, input_tokens=resp.input_tokens,
                               output_tokens=resp.output_tokens, latency_s=resp.latency_s)
                cache[key] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            rec = cache[key]
            parsed = parse_response(rec.get("raw", "")) if rec.get("raw") else {"parse_ok": False}
            rows.append({"complaint_id": str(cid), "model": rec["model"], "mode": mode,
                         "input_tokens": rec.get("input_tokens", 0),
                         "output_tokens": rec.get("output_tokens", 0),
                         "latency_s": rec.get("latency_s"), "error": rec.get("error"), **parsed})
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{len(df)}", flush=True)
    return pd.DataFrame(rows)
