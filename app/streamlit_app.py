"""Streamlit demo for the COD Return-Risk Scorer.

One page, four steps, in the order a reviewer is told to expect:

    risk probability -> expected rupee loss -> ALLOW / REVIEW / BLOCK -> reasons

Imports :func:`src.serving.score_order` directly rather than calling the FastAPI
service over HTTP. One process instead of two, nothing to mis-configure in a live
demo, and — the part that matters — it is provably the *same* scoring path the
API uses, because it is the same function. `api/main.py` exists and is tested;
this simply does not need it standing up to be useful.

Run:

    streamlit run app/streamlit_app.py

Defence-only: this displays a score, a recommendation and reasons. It takes no
action on any order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving import (  # noqa: E402
    SAMPLE_SOURCE_NOTE,
    UnknownPincode,
    get_service,
    sample_orders,
    score_order,
)

CATEGORIES = [
    "fashion", "footwear", "beauty", "accessories", "home_kitchen", "electronics",
]

st.set_page_config(page_title="COD Return-Risk Scorer", page_icon="📦",
                   layout="centered")


@st.cache_resource(show_spinner="Loading the shipped model…")
def _service():
    return get_service()


@st.cache_data(show_spinner=False)
def _samples() -> dict:
    return sample_orders(n_per_tier=2)


service = _service()
info = service.describe()
samples = _samples()

st.title("📦 COD Return-Risk Scorer")
st.caption(
    f"**{info['estimator']}** · frozen threshold **{info['frozen_threshold']}** · "
    f"tier cuts **{info['frozen_tier_cuts']['low_cut']} / "
    f"{info['frozen_tier_cuts']['high_cut']}** · artifact "
    f"`{info['model_sha256'][:12]}` — the exact model and operating point "
    "`05_final_evaluation.ipynb` shipped. Nothing here is retrained or re-tuned."
)

# ---------------------------------------------------------------------------
# 1. Order input
# ---------------------------------------------------------------------------
st.subheader("1 · Order")

choices = {
    f"{o['label']} example — {o['payload']['category']}, "
    f"₹{o['payload']['order_value']:,.0f}, "
    f"{'COD' if o['payload']['is_cod'] else 'prepaid'}": o
    for o in samples["orders"]
}
picked_name = st.selectbox(
    "Start from a real order (swap to see all three tiers)",
    list(choices),
    help=SAMPLE_SOURCE_NOTE,
)
picked = choices[picked_name]
sample = picked["payload"]

# Surfaced rather than left for a reviewer to notice: at the frozen cut points
# the ALLOW tier contains no COD orders at all, so "allow COD" is only ever
# reached by orders that were already prepaid.
if note := samples.get("tier_notes", {}).get(picked["label"]):
    st.warning(f"**Worth knowing about the {picked['label']} tier.** {note}")

with st.form("order"):
    c1, c2 = st.columns(2)
    with c1:
        pincode = st.text_input("Pincode", sample["pincode"], max_chars=6)
        category = st.selectbox("Category", CATEGORIES,
                                index=CATEGORIES.index(sample["category"]))
        order_value = st.number_input("Order value (₹)", min_value=1.0,
                                      value=float(sample["order_value"]), step=50.0)
        discount_pct = st.slider("Discount", 0.0, 0.75,
                                 float(sample["discount_pct"]), 0.01)
    with c2:
        is_cod = st.checkbox("Cash on delivery", value=bool(sample["is_cod"]))
        eta = st.number_input("Courier ETA (days)", min_value=1.0, max_value=30.0,
                              value=float(sample["delivery_days_est"]), step=0.5)
        past_orders = st.number_input(
            "Customer's prior orders", min_value=0, max_value=500,
            value=int(sample["customer_history"]["past_orders"]))
        past_rto = st.number_input(
            "…of which returned", min_value=0, max_value=500,
            value=int(sample["customer_history"]["past_rto_count"]))

    address_line = st.text_area("Delivery address", sample["address_line"], height=68)
    submitted = st.form_submit_button("Score this order", type="primary",
                                      use_container_width=True)

if past_rto > past_orders:
    st.error("A customer cannot have returned more orders than they have placed.")
    st.stop()

payload = {
    **sample,
    "pincode": pincode.strip(),
    "category": category,
    "order_value": float(order_value),
    "discount_pct": float(discount_pct),
    "is_cod": bool(is_cod),
    "delivery_days_est": float(eta),
    "address_line": address_line,
    "customer_history": {
        **sample["customer_history"],
        "past_orders": int(past_orders),
        "past_rto_count": int(past_rto),
    },
}

try:
    result = score_order(payload, service)
except UnknownPincode as exc:
    st.error(f"{exc}. Try a pincode from the India Post directory, e.g. 400001.")
    st.stop()
except (KeyError, ValueError) as exc:
    st.error(f"Could not score this order: {exc}")
    st.stop()

probability = result["rto_probability"]

# ---------------------------------------------------------------------------
# 2. Risk probability
# ---------------------------------------------------------------------------
st.subheader("2 · Risk probability")
m1, m2 = st.columns([2, 1])
m1.progress(min(probability, 1.0))
m1.caption(
    f"{result['resolved']['city']}, {result['resolved']['state']} · "
    f"{result['resolved']['pincode_tier']} · {result['resolved']['order_value_band']}"
)
m2.metric("P(return to origin)", f"{probability:.1%}", help=f"band: {result['band']}")

# ---------------------------------------------------------------------------
# 3. Expected rupee loss
# ---------------------------------------------------------------------------
st.subheader("3 · Expected loss if shipped")
l1, l2 = st.columns([1, 2])
l1.metric("Expected loss", f"₹{result['expected_loss_if_shipped_inr']:,.2f}")
l2.caption(
    f"P(RTO) × ₹{info['cost_params_inr']['fn_cost_inr']:,.0f} forward + reverse "
    "shipping, packaging and handling. Not the order value — an RTO burns "
    "fulfilment cost, it does not lose the sale price. The ₹200 figure is marked "
    "*assumed* in `config/evidence.yaml` and sensitivity-tested across ₹150–250 "
    "in `05`."
)

# ---------------------------------------------------------------------------
# 4. Decision
# ---------------------------------------------------------------------------
st.subheader("4 · Decision")
label = result["action_label"]
detail = (
    f"**{label}** — `{result['action']}` "
    f"(cut points {info['frozen_tier_cuts']['low_cut']} / "
    f"{info['frozen_tier_cuts']['high_cut']}, chosen on validation at the rupee "
    "cost minimum)"
)
if not is_cod:
    st.info(
        f"{detail}\n\nThis order is **already prepaid**, so there is no COD to "
        "allow, price or withdraw. The scorer is a COD gate; the recommendation "
        "does not apply here."
    )
elif label == "ALLOW":
    st.success(f"{detail}\n\nShip on COD. No friction on a good customer.")
elif label == "REVIEW":
    st.warning(f"{detail}\n\nOffer COD **with a fee**. Price the risk instead of "
               "losing the sale.")
else:
    st.error(f"{detail}\n\nWithdraw COD and offer prepaid. Avoid the shipping "
             "loss entirely.")

# ---------------------------------------------------------------------------
# 5. Reasons
# ---------------------------------------------------------------------------
st.subheader("5 · Why")
if result["reasons"]:
    direction = "held this order down" if label == "ALLOW" else "pushed this order up"
    st.caption(f"Top SHAP contributors that {direction}, in plain language.")
    for reason in result["reasons"]:
        st.markdown(f"- {reason}")
else:
    st.caption("No single factor stood out on this order.")

st.divider()
st.caption(
    "Advisory only — this system scores and explains, it never captures, refunds, "
    "blocks or contacts anyone. Trained on synthetic orders built on the real "
    "India Post pincode directory and calibrated against published Indian RTO "
    "statistics; **no claim of validation on real merchant data**. "
    f"Sample orders: {SAMPLE_SOURCE_NOTE}"
)
