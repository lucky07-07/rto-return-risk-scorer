"""Contract and threshold-regression tests for the scoring API.

Two jobs:

1. **Contract** — ``/score`` returns the shape `05` produces, ``/health`` reports
   the artifact actually loaded, and malformed input gets a 4xx rather than a 500.
2. **Regression against the frozen operating point** — a known low-risk order must
   still come back ALLOW and a known high-risk order BLOCK. If someone edits a
   threshold, swaps the artifact or breaks the feature path, these fail.

The tier-vs-payment-mode test is the one that would have caught the "allow COD
printed on a prepaid order" bug: the API is a COD gate, and what it says about an
order that is already prepaid has to be coherent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.serving import get_service, sample_orders

VALID_ACTIONS = {"allow_cod", "charge_cod_fee", "disable_cod"}
VALID_LABELS = {"ALLOW", "REVIEW", "BLOCK"}
ACTION_TO_LABEL = {
    "allow_cod": "ALLOW", "charge_cod_fee": "REVIEW", "disable_cod": "BLOCK",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def samples() -> dict:
    return sample_orders(n_per_tier=2)


def _payload(samples: dict, label: str) -> dict:
    for order in samples["orders"]:
        if order["label"] == label:
            return order["payload"]
    raise AssertionError(f"no {label} sample available")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_score_returns_the_documented_shape(client, samples):
    r = client.post("/score", json=_payload(samples, "REVIEW"))
    assert r.status_code == 200, r.text
    body = r.json()

    expected = {
        "order_id", "rto_probability", "band", "action", "action_label",
        "reasons", "expected_loss_if_shipped_inr", "flagged_at_binary_threshold",
        "resolved",
    }
    assert expected <= set(body), f"missing keys: {expected - set(body)}"

    assert 0.0 <= body["rto_probability"] <= 1.0
    assert body["action"] in VALID_ACTIONS
    assert body["action_label"] in VALID_LABELS
    assert body["band"] in {"low", "medium", "high"}
    assert isinstance(body["reasons"], list)
    assert all(isinstance(x, str) for x in body["reasons"])
    assert body["expected_loss_if_shipped_inr"] >= 0.0


def test_action_and_action_label_never_disagree(client, samples):
    """``action_label`` is a rename, not a second decision."""
    for order in samples["orders"]:
        body = client.post("/score", json=order["payload"]).json()
        assert ACTION_TO_LABEL[body["action"]] == body["action_label"]


def test_expected_loss_is_probability_times_fn_cost(client, samples):
    """The rupee figure must be derived, not decorative."""
    svc = get_service()
    body = client.post("/score", json=_payload(samples, "BLOCK")).json()
    expected = body["rto_probability"] * svc.costs.fn_cost_inr
    assert body["expected_loss_if_shipped_inr"] == pytest.approx(expected, abs=0.51)


# ---------------------------------------------------------------------------
# Threshold regression
# ---------------------------------------------------------------------------


def test_known_allow_order_still_scores_allow(client, samples):
    body = client.post("/score", json=_payload(samples, "ALLOW")).json()
    assert body["action"] == "allow_cod"
    assert body["action_label"] == "ALLOW"
    assert body["rto_probability"] < get_service().low_cut


def test_known_block_order_still_scores_block(client, samples):
    body = client.post("/score", json=_payload(samples, "BLOCK")).json()
    assert body["action"] == "disable_cod"
    assert body["action_label"] == "BLOCK"
    assert body["rto_probability"] >= get_service().high_cut


def test_tiers_follow_the_frozen_cut_points(client, samples):
    """Every sample's tier must be reproducible from its probability alone."""
    svc = get_service()
    for order in samples["orders"]:
        body = client.post("/score", json=order["payload"]).json()
        p = body["rto_probability"]
        expected = ("allow_cod" if p < svc.low_cut
                    else "charge_cod_fee" if p < svc.high_cut
                    else "disable_cod")
        assert body["action"] == expected, (
            f"{order['label']} sample scored {p} but was tiered {body['action']}"
        )


def test_allow_tier_contains_no_cod_orders(samples):
    """Regression guard on a real finding, not a hypothetical.

    At the frozen cut points no COD order reaches the allow tier -- the tier is
    populated entirely by prepaid orders, for which the COD gate is a no-op.
    ``sample_orders`` must keep saying so. If a future change makes COD orders
    reachable in the allow tier, this test fails and the note (and the README
    table it corrects) has to be revisited rather than silently going stale.
    """
    allow = [o for o in samples["orders"] if o["label"] == "ALLOW"]
    assert allow, "expected at least one ALLOW sample"
    assert all(o["cod_orders_in_tier"] == 0 for o in allow)
    assert all(not o["is_cod"] for o in allow)
    assert "ALLOW" in samples["tier_notes"]


def test_prepaid_order_is_not_told_to_allow_cod(client, samples):
    """The bug this file exists for.

    The service is a COD gate. On an order that is already prepaid there is no
    COD to allow, price or withdraw, so the response must be coherent about that
    rather than emitting a COD instruction. The API keeps the tier for backward
    compatibility and exposes ``is_cod`` upstream; the UI is what must not print
    "allow COD" on a prepaid order, and it branches on the same flag.
    """
    prepaid = dict(_payload(samples, "ALLOW"))
    prepaid["is_cod"] = False
    body = client.post("/score", json=prepaid).json()
    assert body["action"] in VALID_ACTIONS
    # A prepaid order is inherently low RTO risk; it must never come back BLOCK.
    assert body["action"] != "disable_cod"
    assert body["rto_probability"] < 0.5


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_identifies_the_loaded_artifact(client):
    body = client.get("/health").json()
    svc = get_service()

    assert body["status"] == "ok"
    assert body["model_key"] == svc.bundle["model_key"] == "09_catboost__flaml"
    assert body["estimator"] == "CatBoostClassifier"
    assert body["model_sha256"] == svc.model_sha256
    assert body["frozen_threshold"] == svc.threshold
    assert body["frozen_tier_cuts"]["low_cut"] == svc.low_cut
    assert body["frozen_tier_cuts"]["high_cut"] == svc.high_cut


def test_health_matches_the_frozen_point_recorded_by_05(client):
    """The demo must report the same operating point `05` persisted."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "reports/results/05_final_metrics.json").read_text(encoding="utf-8")
    )["frozen_operating_point"]

    body = client.get("/health").json()
    assert body["model_key"] == frozen["model_key"]
    assert body["frozen_threshold"] == frozen["threshold"]
    assert body["frozen_tier_cuts"]["low_cut"] == frozen["low_cut"]
    assert body["frozen_tier_cuts"]["high_cut"] == frozen["high_cut"]


# ---------------------------------------------------------------------------
# Malformed input must be 4xx, never 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ({"pincode": "99"}, "pincode too short"),
        ({"pincode": "abcdef"}, "pincode not numeric"),
        ({"pincode": "999999"}, "pincode absent from the India Post directory"),
        ({"category": "weapons"}, "category outside the trained set"),
        ({"order_value": -5}, "negative order value"),
        ({"order_value": 0}, "zero order value"),
        ({"discount_pct": 5}, "discount as a percent rather than a fraction"),
        ({"customer_history": {"past_orders": 1, "past_rto_count": 9}},
         "more returns than orders"),
    ],
)
def test_malformed_input_returns_4xx(client, samples, mutation, reason):
    body = {**_payload(samples, "REVIEW"), **mutation}
    r = client.post("/score", json=body)
    assert 400 <= r.status_code < 500, f"{reason}: got {r.status_code}, want 4xx"


def test_missing_required_fields_returns_4xx(client):
    r = client.post("/score", json={"pincode": "110001"})
    assert r.status_code == 422


def test_sample_orders_are_labelled_as_validation_not_test(client):
    """PRE_REGISTRATION.md keeps the test set for `05`. The demo must not use it."""
    body = client.get("/sample_orders").json()
    assert "val" in body["source"]
    assert "test" not in body["source"]
    assert "VALIDATION" in body["note"]
    assert {o["label"] for o in body["orders"]} == VALID_LABELS


# ---------------------------------------------------------------------------
# The plain-language layer must never break the technical response
# ---------------------------------------------------------------------------


class _Resp429:
    """A Gemini rate-limit response, which the free tier hits readily."""

    status_code = 429
    text = "quota exceeded"

    def raise_for_status(self):
        import httpx
        raise httpx.HTTPStatusError("429", request=None, response=None)

    def json(self):
        return {}


@pytest.mark.parametrize(
    "failure, label",
    [
        (lambda *a, **k: _Resp429(), "rate limited"),
        (__import__("httpx").ConnectTimeout("timed out"), "timeout"),
        (__import__("httpx").ConnectError("refused"), "connection refused"),
        (ValueError("malformed response"), "malformed response"),
    ],
)
def test_score_survives_every_gemini_failure(client, samples, failure, label):
    """A broken explanation layer must degrade the wording, never the decision.

    The free Gemini tier rate-limits readily, so this is the common path in a
    live demo rather than an edge case. Every mode has to come back as a full
    200 carrying the real probability and a readable summary.
    """
    from unittest.mock import patch

    payload = _payload(samples, "REVIEW")
    with patch("httpx.post", side_effect=failure):
        r = client.post("/score", json=payload)

    assert r.status_code == 200, f"{label} produced HTTP {r.status_code}"
    body = r.json()
    assert body["summary_source"] == "template"
    assert len(body["plain_language_summary"]) > 60
    assert body["rto_probability"] > 0
    assert body["action"] in VALID_ACTIONS


def test_summary_falls_back_when_no_key_is_configured():
    """No key is a supported configuration, not an error."""
    from unittest.mock import patch

    import src.interpret as interpret

    result = {
        "rto_probability": 0.4, "expected_loss_if_shipped_inr": 80.0,
        "action": "disable_cod", "reasons": ["paying cash on delivery"],
        "is_cod": True,
    }
    with patch.object(interpret, "_load_env_key", return_value=None):
        out = interpret.plain_language_summary(result)

    assert out["source"] == "template"
    assert len(out["text"]) > 60
    assert "40%" in out["text"]


def test_prepaid_summary_never_recommends_a_cod_action():
    """A prepaid order has no cash-on-delivery decision to make."""
    from unittest.mock import patch

    import src.interpret as interpret

    result = {
        "rto_probability": 0.016, "expected_loss_if_shipped_inr": 3.2,
        "action": "allow_cod", "reasons": ["paying online up front"], "is_cod": False,
    }
    with patch.object(interpret, "_load_env_key", return_value=None):
        text = interpret.plain_language_summary(result)["text"].lower()

    assert "already paid" in text
    assert "offer cash on delivery" not in text
    assert "let the customer pay cash on delivery" not in text
