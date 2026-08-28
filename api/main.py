"""FastAPI scoring service for the COD Return-Risk Scorer.

Thin by design. Every decision — feature engineering, the frozen threshold, the
tier cut points, the cost model, the SHAP reasons — lives in ``src/`` and is
imported. This file does request validation, HTTP status codes and JSON shape,
and nothing else. If a number appears here that is not in ``src/serving.py``,
that is a bug.

Run:

    uvicorn api.main:app --reload      # http://127.0.0.1:8000/docs

Defence-only: the service returns a probability, a recommended tier and reasons.
It has no code path that captures a payment, blocks an account, cancels an order
or contacts a customer. Acting on the recommendation is the merchant's decision.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from src.interpret import GEMINI_MODEL, gemini_available  # noqa: E402
from src.serving import (  # noqa: E402
    UnknownPincode,
    get_service,
    sample_orders,
    score_order,
)

VALID_CATEGORIES = [
    "fashion", "footwear", "beauty", "accessories", "home_kitchen", "electronics",
]


class CustomerHistoryIn(BaseModel):
    """The merchant's own record of this customer.

    A checkout payload does not carry these; they come from the merchant's order
    store. Omitting the block scores the order as a genuine first-time customer,
    which is a real and common case — 10% of the test split — and one the model
    handles measurably worse (see the exception list in `05`).
    """

    past_orders: int = Field(0, ge=0, le=10_000)
    past_rto_count: int = Field(0, ge=0, le=10_000)
    account_age_days: float = Field(0.0, ge=0.0, le=20_000.0)
    order_velocity_24h: int = Field(0, ge=0, le=100)

    @field_validator("past_rto_count")
    @classmethod
    def _rto_not_above_orders(cls, v: int, info) -> int:
        past = info.data.get("past_orders")
        if past is not None and v > past:
            raise ValueError(
                f"past_rto_count ({v}) cannot exceed past_orders ({past})"
            )
        return v


class OrderIn(BaseModel):
    """One order, as a checkout would present it.

    ``city``, ``state`` and ``pincode_tier`` are deliberately **not** accepted:
    they are derived from the pincode against the real India Post directory. A
    caller who could supply a state that disagrees with the pincode would be
    handing the model a combination it never saw in training.
    """

    pincode: str = Field(..., description="6-digit Indian pincode")
    address_line: str = Field(..., min_length=1, max_length=500)
    category: Literal[
        "fashion", "footwear", "beauty", "accessories", "home_kitchen", "electronics"
    ]
    order_value: float = Field(..., gt=0, le=1_000_000, description="INR")
    is_cod: bool

    order_id: str = Field("ord_adhoc", max_length=64)
    customer_id: str = Field("CUST_UNKNOWN", max_length=64)
    discount_pct: float = Field(0.0, ge=0.0, le=1.0, description="fraction, not %")
    is_festive: bool | None = Field(
        None, description="defaults to whether order_ts falls in a known sale window"
    )
    is_alternate_address: bool = False
    delivery_days_est: float | None = Field(
        None, gt=0, le=60, description="courier ETA; defaults to the tier baseline"
    )
    order_ts: datetime | None = None
    customer_history: CustomerHistoryIn = Field(default_factory=CustomerHistoryIn)

    @field_validator("pincode")
    @classmethod
    def _six_digits(cls, v: str) -> str:
        v = v.strip()
        if not (len(v) == 6 and v.isdigit()):
            raise ValueError("pincode must be exactly 6 digits")
        return v


class ScoreOut(BaseModel):
    """The decision record. Shape is fixed by `05`; do not reorder casually."""

    order_id: str
    rto_probability: float
    band: Literal["low", "medium", "high"]
    action: Literal["allow_cod", "charge_cod_fee", "disable_cod"]
    action_label: Literal["ALLOW", "REVIEW", "BLOCK"]
    reasons: list[str]
    expected_loss_if_shipped_inr: float
    flagged_at_binary_threshold: bool
    is_cod: bool
    resolved: dict[str, Any]
    plain_language_summary: str | None = None
    summary_source: Literal["gemini", "template"] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup — never per request."""
    get_service()
    yield


app = FastAPI(
    title="COD Return-Risk Scorer",
    description=(
        "Scores Cash-on-Delivery orders for Return-to-Origin risk and returns a "
        "graded recommendation. Razorpay AI Buildathon 2026, Track 02. "
        "Defence-only: it scores and explains, it never moves money."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    """The one-page demo. Same service, same URL, no second process to start."""
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(UnknownPincode)
async def _unknown_pincode(request, exc: UnknownPincode) -> JSONResponse:
    """A pincode outside the directory is the caller's error, not a crash."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Model identity and the frozen operating point currently loaded.

    Exists so a judge can confirm the demo is running the real shipped artifact —
    the SHA-256 here is of ``models/final_model.joblib`` itself, and the threshold
    and cut points are read out of that file rather than restated in code.
    """
    return {
        "status": "ok",
        **get_service().describe(),
        "plain_language_layer": {
            "configured": gemini_available(),
            "model": GEMINI_MODEL,
            "fallback": "built-in template when the key is absent or the call fails",
        },
    }


@app.get("/sample_orders", tags=["demo"])
def get_sample_orders(n_per_tier: int = 2) -> dict:
    """Ready-made orders spanning ALLOW, REVIEW and BLOCK.

    Drawn from the **validation** split, not the sealed test set — see the note
    in the response body. Saves a judge from inventing a plausible Indian address
    to see all three outcomes.
    """
    if not 1 <= n_per_tier <= 5:
        raise HTTPException(422, "n_per_tier must be between 1 and 5")
    return sample_orders(n_per_tier=n_per_tier)


@app.post("/score", response_model=ScoreOut, tags=["scoring"])
def score(order: OrderIn, explain_plainly: bool = True) -> dict:
    """Score one order: probability, expected rupee loss, tier, reasons.

    ``explain_plainly`` adds the Gemini plain-language summary. It costs a network
    round trip and degrades to a built-in template if the key is absent or the
    call fails, so it can be left on safely.
    """
    payload = order.model_dump()
    payload["customer_history"] = order.customer_history.model_dump()
    try:
        return score_order(payload, interpret=explain_plainly)
    except UnknownPincode as exc:
        raise HTTPException(422, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, f"could not score this order: {exc}") from exc
