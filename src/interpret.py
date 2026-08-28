"""Plain-language interpretation of a scoring result, via Gemini.

The risk model emits a probability, a tier and a list of SHAP-derived reasons.
That is precise and it is unreadable to anyone who does not already know what a
tier is. This module turns it into two or three sentences a merchant can act on.

Design rules, in order of importance:

1. **It cannot invent numbers.** The prompt supplies the probability, the rupee
   figure and the reasons, and asks for prose over exactly those. Nothing is left
   for the model to guess at.
2. **It cannot change the decision.** The tier is decided by the frozen cost
   thresholds. This layer explains that decision, it never revises it.
3. **It must fail open.** No key, no network, a timeout, a malformed response,
   any of them fall back to a deterministic template. The technical response is
   always returned in full.

The key is read from ``GEMINI_API_KEY`` in the environment, or from a local
``.env`` if one exists. It is never logged, never returned in a response and
never committed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Google retires model names on their own schedule, and a hard-coded one turns
# into a silent fallback to the template the day it goes. The name is overridable
# from the environment, and the first choice is tried before the next.
# "gemini-flash-latest" is an alias Google keeps pointed at a current model, so
# it does not rot when a specific version is retired, and it carries its own
# free-tier quota bucket separate from the pinned names.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-2.5-flash"]
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
TIMEOUT_SECONDS = 8.0


def _endpoint(model: str) -> str:
    return f"{API_ROOT}/{model}:generateContent"

ACTION_WORDS = {
    "allow_cod": "let the customer pay cash on delivery as normal",
    "charge_cod_fee": "offer cash on delivery but add a small fee to cover the risk",
    "disable_cod": "ask the customer to pay online instead of offering cash on delivery",
}


def _load_env_key() -> str | None:
    """Read the API key from the environment, falling back to a local .env file.

    Deliberately minimal, no dependency on python-dotenv, and it never returns or
    logs the value anywhere except into the outbound request.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip() or None

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                return value or None
    return None


def gemini_available() -> bool:
    """Whether a key is configured. Says nothing about whether the call will work."""
    return _load_env_key() is not None


def _fallback_summary(result: dict) -> str:
    """Deterministic plain-English summary used whenever Gemini is unavailable.

    This is not a degraded placeholder. It is a complete, readable answer built
    from the same three facts, so a demo with no API key still shows a sensible
    plain-language explanation rather than an error.
    """
    pct = result["rto_probability"] * 100
    action = ACTION_WORDS.get(result["action"], result["action"])
    loss = result["expected_loss_if_shipped_inr"]

    if not result.get("is_cod", True):
        return (
            f"This customer has already paid online, so there is nothing to decide "
            f"here. The risk of the parcel coming back is very low, around "
            f"{pct:.0f}%, and because the money is already collected you would not "
            "lose the sale even if it did. Ship it as normal."
        )

    if result["action"] == "allow_cod":
        risk = f"This order looks safe, with roughly a {pct:.0f}% chance of coming back."
    elif result["action"] == "charge_cod_fee":
        risk = (
            f"This order carries moderate risk, with roughly a {pct:.0f}% chance "
            "of being returned undelivered."
        )
    else:
        risk = (
            f"This order is high risk, with roughly a {pct:.0f}% chance of being "
            "returned undelivered."
        )

    reasons = result.get("reasons") or []
    because = f" The main factors are {', and '.join(reasons[:2])}." if reasons else ""
    return (
        f"{risk} On average that works out to about Rs{loss:,.0f} of wasted "
        f"shipping cost per order like this. The recommendation is to {action}.{because}"
    )


def _build_prompt(result: dict) -> str:
    """The instruction sent to Gemini. Every number it may use is supplied here."""
    reasons = result.get("reasons") or []
    reason_lines = "\n".join(f"- {r}" for r in reasons) or "- no single standout factor"

    if not result.get("is_cod", True):
        # A prepaid order has no cash-on-delivery decision to make. Left to its
        # own devices the model produces confident nonsense here, recommending
        # that you "let the customer pay cash on delivery" on an order that is
        # already paid for.
        return f"""You are explaining an automated risk check to a small online \
retailer in India who has no technical background.

Facts you must use, and you must not invent any others:
- This customer has ALREADY PAID ONLINE. There is no cash-on-delivery decision to make.
- Chance the parcel is returned undelivered anyway: {result['rto_probability'] * 100:.1f}%

Write 2 sentences telling the retailer that this order needs no action because it \
is already paid for, and that the risk of it coming back is low.

Rules:
- Never suggest offering, pricing or withdrawing cash on delivery. It does not apply here.
- No technical or statistical words.
- Do not invent any number beyond the one listed above.
- Write in the second person, addressed to the retailer.
- Return the sentences only, with no preamble or bullet points."""

    # On an allow, src.serving asks for the factors that held risk DOWN. Handing
    # those over unlabelled makes them read as risk factors and the explanation
    # comes back negated, turning "paying online up front" into "they aren't
    # paying online upfront". The prompt has to say which way they point.
    if result["action"] == "allow_cod":
        reason_header = (
            "Reassuring things that are TRUE of this order, each one a reason it is "
            "LOW risk. Repeat them as facts, never negate them"
        )
    else:
        reason_header = (
            "Concerning things that are TRUE of this order, each one a reason it is "
            "risky. Repeat them as facts, never negate them"
        )

    return f"""You are explaining an automated risk decision to a small online \
retailer in India who has no technical background.

Facts you must use, and you must not invent any others:
- Chance this order is returned undelivered: {result['rto_probability'] * 100:.1f}%
- Average money wasted on shipping for an order like this: Rs{result['expected_loss_if_shipped_inr']:,.0f}
- The system's recommendation: {ACTION_WORDS.get(result['action'], result['action'])}
- {reason_header}:
{reason_lines}

Write 2 to 3 sentences, in plain English, covering what the risk level means, \
what the retailer should do, and why.

Rules:
- No technical or statistical words. Never write model, probability, score, \
threshold, feature, SHAP, algorithm, prediction or percentage point.
- Do not invent any number that is not listed above.
- Do not contradict or second-guess the recommendation, explain it.
- Write in the second person, addressed to the retailer.
- Return the sentences only, with no preamble, heading or bullet points."""


def _looks_usable(text: str) -> bool:
    """Reject output that would embarrass us in front of a reader.

    A truncated or rambling completion is worse than the template, because the
    template is always coherent. Anything that fails these checks falls back
    rather than being shown. Cheap insurance against a model change upstream.
    """
    if not text or len(text) < 60 or len(text) > 900:
        return False
    if not text.rstrip().endswith((".", "!", "?")):
        return False                          # truncated mid-sentence
    lowered = text.lower()
    leaked_reasoning = ("better stay close", "let me", "i should", "wait,",
                        "hmm", "okay,", "first,i", "we need to")
    if any(marker in lowered for marker in leaked_reasoning):
        return False
    banned_jargon = ("shap", "probability", "threshold", "model output",
                     "algorithm", "feature importance")
    if any(word in lowered for word in banned_jargon):
        return False
    return True


def plain_language_summary(result: dict, timeout: float = TIMEOUT_SECONDS) -> dict:
    """Turn a :func:`src.serving.score_order` result into readable prose.

    Returns a dict with the text and its provenance, so a caller can always tell
    whether it came from Gemini or from the local template rather than having to
    guess from the wording.
    """
    key = _load_env_key()
    if not key:
        return {
            "text": _fallback_summary(result),
            "source": "template",
            "detail": "GEMINI_API_KEY is not set, using the built-in summary.",
        }

    body = {
        "contents": [{"parts": [{"text": _build_prompt(result)}]}],
        "generationConfig": {
            "temperature": 0.1,               # explanation, not creative writing
            "maxOutputTokens": 512,
            "topP": 0.9,
            # Current Gemini flash models reason before answering, and that
            # reasoning is billed against maxOutputTokens. Left on, it eats the
            # budget and the actual answer comes back truncated mid-sentence.
            # This is a two-sentence rewrite of facts already supplied, so there
            # is nothing to reason about.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    candidates = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS
                                   if m != GEMINI_MODEL]
    last_error = "no attempt made"
    try:
        import httpx
    except ImportError:
        return {"text": _fallback_summary(result), "source": "template",
                "detail": "httpx is not installed, using the built-in summary."}

    for model in candidates:
        try:
            response = httpx.post(_endpoint(model), params={"key": key}, json=body,
                                  timeout=timeout)
            if response.status_code == 404:   # retired model name, try the next
                last_error = f"{model} is unavailable"
                continue
            response.raise_for_status()
            candidate = response.json()["candidates"][0]
            if candidate.get("finishReason") not in (None, "STOP"):
                raise ValueError(f"finishReason={candidate.get('finishReason')}")
            parts = candidate["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            if not _looks_usable(text):
                raise ValueError("completion failed the quality check")
            return {"text": text, "source": "gemini", "detail": model}
        except Exception as exc:              # noqa: BLE001 - must fail open
            last_error = f"{type(exc).__name__}"
            continue

    return {
        "text": _fallback_summary(result),
        "source": "template",
        "detail": f"Gemini unavailable ({last_error}), using the built-in summary.",
    }
