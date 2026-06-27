"""
Secure OpenRouter integration for desk-style commentary.

Security model (the whole point):
  * The API key is read from the server environment / Streamlit secrets ONLY.
    It is never returned to the client, never logged, never embedded in code.
    Because this module runs server-side (inside Streamlit / FastAPI), the
    browser talks to *our* process and never to openrouter.ai directly.
  * A per-process token-bucket rate limiter caps how often the endpoint can be
    hit, so a leaked URL can't drain credits. (On multi-instance hosting, swap
    this for a shared Upstash/Redis counter -- noted in the README.)
  * Inputs are size-capped before they are forwarded, so nobody can push a
    huge prompt and run up the bill.
  * The model is constrained to PHRASE pre-computed metrics, never to compute
    or invent numbers -- the numerics stay in scipy where they belong.

If no key is configured the module degrades gracefully: callers get a clear
"commentary disabled" result and the rest of the app keeps working.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# COST SAFETY -----------------------------------------------------------------
# Default to OpenRouter's free router, which auto-selects a $0 ":free" model.
# With a $0 account balance and a free model there is nothing to bill against,
# so the app cannot incur a charge. The default below keeps the repo free-by-
# default even if someone forgets to configure anything.
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

# Free-only mode is ON unless the operator explicitly opts into paid models by
# setting OPENROUTER_ALLOW_PAID=1. While ON, a request for any non-free model is
# refused *before* any network call, so a misconfiguration can never spend money.
FREE_ONLY = os.environ.get("OPENROUTER_ALLOW_PAID", "0") != "1"

REQUEST_TIMEOUT_S = 30
MAX_INPUT_CHARS = 6000           # hard cap on the metrics payload we forward
MAX_OUTPUT_TOKENS = 600

# Conservative caps, deliberately *below* OpenRouter's free-tier limits
# (20 req/min, 50 req/day) so we never even reach them.
PER_MIN_LIMIT = int(os.environ.get("LLM_PER_MIN", "12"))
PER_DAY_LIMIT = int(os.environ.get("LLM_PER_DAY", "40"))


def _is_free_model(model: str) -> bool:
    """True for OpenRouter $0 routes: any ':free' slug or the free router."""
    m = (model or "").strip().lower()
    return m.endswith(":free") or m.startswith("openrouter/free")

SYSTEM_PROMPT = (
    "You are a sell-side volatility desk strategist writing a short internal "
    "note. You will be given a JSON object of ALREADY-COMPUTED metrics from a "
    "fitted implied-volatility surface. Your job is ONLY to phrase these numbers "
    "into a concise, professional desk commentary (about 120-180 words).\n"
    "STRICT RULES:\n"
    "- Never compute, estimate, infer, or invent any number that is not present "
    "in the provided JSON. If a figure is not given, do not mention it.\n"
    "- Quote the provided numbers faithfully (you may round for readability).\n"
    "- Interpret qualitatively: what the skew/term-structure/arbitrage status "
    "imply for positioning and risk, in the voice of a desk note.\n"
    "- No bullet lists, no headers; flowing prose only. No disclaimers."
)


@dataclass
class ReportResult:
    ok: bool
    text: str
    reason: str = ""


def get_api_key():
    """Resolve the key from env or Streamlit secrets, server-side only."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # optional dependency
        return st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        return None


class RateLimiter:
    """Simple in-process token bucket: `rate` requests per `per` seconds."""

    def __init__(self, rate=10, per=60.0):
        self.rate = rate
        self.per = per
        self.allowance = float(rate)
        self.last = time.monotonic()

    def allow(self):
        now = time.monotonic()
        self.allowance += (now - self.last) * (self.rate / self.per)
        self.last = now
        if self.allowance > self.rate:
            self.allowance = float(self.rate)
        if self.allowance < 1.0:
            return False
        self.allowance -= 1.0
        return True


class DailyCap:
    """Hard ceiling on calls per UTC day. Resets at midnight UTC."""

    def __init__(self, limit):
        self.limit = limit
        self.day = time.gmtime().tm_yday
        self.count = 0

    def allow(self):
        today = time.gmtime().tm_yday
        if today != self.day:
            self.day, self.count = today, 0
        if self.count >= self.limit:
            return False
        self.count += 1
        return True


# Process-wide caps, deliberately below OpenRouter's free-tier limits.
_limiter = RateLimiter(rate=PER_MIN_LIMIT, per=60.0)
_daily = DailyCap(PER_DAY_LIMIT)


def generate_desk_note(metrics: dict, model: str = DEFAULT_MODEL,
                       limiter: RateLimiter = None,
                       daily: DailyCap = None) -> ReportResult:
    """
    Turn a finished metrics dict into a desk note via OpenRouter. Safe to call
    unconditionally: returns ok=False with a reason rather than raising.

    Cost safety, in order:
      1. no key            -> no call.
      2. paid model blocked -> no call, unless OPENROUTER_ALLOW_PAID=1.
      3. per-minute cap     -> no call.
      4. per-day cap        -> no call.
    Combined with a $0 balance on a ':free' model, the app cannot be charged.
    """
    limiter = limiter or _limiter
    daily = daily or _daily
    key = get_api_key()
    if not key:
        return ReportResult(False, "", "no_api_key")
    if FREE_ONLY and not _is_free_model(model):
        # refuse to touch a paid model in free-only mode -> cannot spend money
        return ReportResult(False, "", "paid_model_blocked")
    if not limiter.allow():
        return ReportResult(False, "", "rate_limited")
    if not daily.allow():
        return ReportResult(False, "", "daily_limit")

    payload_str = json.dumps(metrics, default=str)
    if len(payload_str) > MAX_INPUT_CHARS:
        payload_str = payload_str[:MAX_INPUT_CHARS]  # bound the request size

    body = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Write the desk note from these metrics:\n" + payload_str},
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # optional attribution headers OpenRouter recognises
        "HTTP-Referer": os.environ.get("APP_PUBLIC_URL", "http://localhost"),
        "X-Title": "Tessera",
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body,
                             timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return ReportResult(True, text)
    except requests.Timeout:
        return ReportResult(False, "", "timeout")
    except requests.HTTPError as e:
        return ReportResult(False, "", f"http_{e.response.status_code}")
    except Exception as e:
        return ReportResult(False, "", f"error_{type(e).__name__}")
