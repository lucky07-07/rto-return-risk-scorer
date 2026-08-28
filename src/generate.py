"""Synthetic COD order generation on the real India Post pincode skeleton.

Everything that determines the RTO label is driven by one seeded ``numpy``
Generator. Faker supplies only cosmetic strings (names, phones, street text).

Design contract
---------------
1. **Marginals are calibrated, conditionals are free.** A small set of log-odds
   offsets (payment mode x order-value band) is *solved* so that the generated
   data reproduces the published Indian statistics in ``config/evidence.yaml``.
   Everything else -- city prior, customer latent, address quality, category,
   discount, velocity, account age, delivery ETA -- contributes freely.
2. **Labels are Bernoulli draws** from the calibrated probability plus an
   idiosyncratic logit noise term, so the classes overlap irreducibly. There is
   no deterministic rule for a model to memorise.
3. **The order log contains only what is knowable at checkout.** Latents
   (customer reliability, true address quality, the per-pincode prior) are
   written to ``data/interim/`` for auditing and are never used as features.
4. **Customer identity persists.** ``past_rto_rate`` is a real expanding history
   over that customer's earlier orders, not a random column. The label depends
   on the *latent* reliability, never on the observed history, so there is no
   circularity.

Defence-only: this module fabricates data for model development. It contains no
capability to act on a real order, a real customer or real money.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEED = 20260101

# Window over which orders are generated; the chronological split runs along it.
WINDOW_START = pd.Timestamp("2024-01-01")
WINDOW_END = pd.Timestamp("2025-06-30")

N_ORDERS = 50_000
N_CUSTOMERS = 20_000

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

# Curated tier lists. Real, checkable geography -- deliberately *not* inferred
# from the CSV, because post-office density in the India Post directory tracks
# rural spread rather than urbanisation (Thrissur has 370 pincodes, Pune 125).
METRO_CITIES = {
    "Mumbai", "Delhi", "New Delhi", "Bangalore", "Bengaluru", "Chennai",
    "Kolkata", "Calcutta", "Hyderabad", "Pune", "Ahmedabad",
}

TIER1_CITIES = {
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
    "Vadodara", "Surat", "Ludhiana", "Agra", "Nashik", "Coimbatore",
    "Visakhapatnam", "Chandigarh", "Guwahati", "Bhubaneswar", "Raipur",
    "Ranchi", "Jodhpur", "Amritsar", "Varanasi", "Meerut", "Rajkot",
    "Madurai", "Jabalpur", "Srinagar", "Aurangabad", "Dhanbad", "Faridabad",
    "Ghaziabad", "Gurgaon", "Noida", "Thane", "Trivandrum", "Kochi",
    "Ernakulam", "Mysore", "Mangalore", "Vijayawada", "Kozhikode", "Solapur",
    "Gwalior", "Jalandhar", "Bareilly", "Allahabad", "Salem", "Warangal",
}

TIER_ORDER = ["metro", "tier_1", "tier_2", "tier_3"]

# Where each tier sits inside the published 18-35% city RTO range.
# The *endpoints* are grounded (evidence.yaml); this positioning is assumed.
TIER_POSITION = {"metro": 0.28, "tier_1": 0.45, "tier_2": 0.63, "tier_3": 0.80}

# Baseline courier ETA in days by tier (assumed; see DATA_CARD.md).
TIER_ETA_DAYS = {"metro": 2.0, "tier_1": 3.0, "tier_2": 4.5, "tier_3": 6.5}

# Share of order volume by tier. Metro / tier-1 dominate D2C volume.
TIER_VOLUME_WEIGHT = {"metro": 0.34, "tier_1": 0.30, "tier_2": 0.22, "tier_3": 0.14}

# Cities named in evidence.yaml as the endpoints of the published range. Their
# priors are pinned exactly rather than sampled.
ANCHOR_CITIES = {"Vadodara": "min", "Patna": "max"}

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Category:
    """One merchandise category: volume share, price distribution, COD lean."""

    name: str
    share: float
    log_value_mu: float      # lognormal parameters for order value, INR
    log_value_sigma: float
    cod_logit: float         # nudge towards / away from COD at checkout


CATEGORIES = [
    Category("fashion",      0.30, float(np.log(680)),  0.62,  0.25),
    Category("footwear",     0.11, float(np.log(980)),  0.55,  0.15),
    Category("beauty",       0.13, float(np.log(520)),  0.58,  0.10),
    Category("accessories",  0.14, float(np.log(460)),  0.70,  0.20),
    Category("home_kitchen", 0.17, float(np.log(880)),  0.75, -0.05),
    Category("electronics",  0.15, float(np.log(1750)), 0.85, -0.55),
]

VALUE_BANDS = ["under_500", "500_1000", "1000_plus"]

# Festive windows: (start, end, order-volume multiplier, extra discount).
FESTIVE_WINDOWS = [
    ("2024-01-18", "2024-01-28", 1.7, 0.10),   # Republic Day sale
    ("2024-07-08", "2024-07-20", 1.5, 0.12),   # end-of-season sale
    ("2024-09-27", "2024-11-05", 2.6, 0.22),   # Big Billion Days / Diwali
    ("2025-01-18", "2025-01-28", 1.7, 0.10),
    ("2025-05-05", "2025-05-18", 1.4, 0.10),   # summer sale
]

# ---------------------------------------------------------------------------
# Label model: log-odds coefficients (assumed; sensitivity-tested in 05)
# ---------------------------------------------------------------------------

COEF = {
    "city": 1.00,            # x (logit(city_prior) - logit(mean city prior))
    "reliability": 0.85,     # per SD of the negated customer reliability latent
    "address": 1.30,         # per unit of (1 - true address quality)
    "category": 1.00,        # x log(category RTO multiplier)
    "discount": 0.90,        # per unit discount fraction, centred at 0.20
    "festive": 0.28,
    "delivery_days": 0.075,  # per day of courier ETA beyond the mean
    "velocity": 0.30,        # per prior order by the same customer in 24h
    "account_age": 0.32,     # per unit of -log1p(days)/log1p(365), centred
    "repeat_buyer": -0.45,   # having any prior order at all
    "noise_sd": 0.55,        # idiosyncratic logit noise (irreducible)
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def load_evidence(path: str | Path = "config/evidence.yaml") -> dict:
    """Load the grounding constants. Every generator target comes from here."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def solve_offset(base_logit: np.ndarray, target_rate: float) -> float:
    """Find ``c`` such that ``mean(sigmoid(base_logit + c)) == target_rate``.

    Deterministic bisection -- consumes no RNG draws, so it cannot perturb
    reproducibility. ``mean(sigmoid(z + c))`` is strictly increasing in ``c``.
    """
    base_logit = np.asarray(base_logit, dtype=float)
    if base_logit.size == 0:
        return 0.0
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sigmoid(base_logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def value_band(value) -> np.ndarray:
    """Order-value band. The RTO curve across these bands is non-monotonic."""
    value = np.asarray(value, dtype=float)
    return np.select(
        [value < 500.0, value < 1000.0],
        ["under_500", "500_1000"],
        default="1000_plus",
    )


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_manifest(
    paths,
    out_path: str | Path,
    extra: dict | None = None,
    root: str | Path | None = None,
) -> dict:
    """SHA-256 manifest of every output, so a rerun is provably identical.

    Paths are recorded relative to ``root`` (the repo root by default) so two
    checkouts on different machines produce comparable manifests.
    """
    root = Path(root) if root is not None else Path(out_path).resolve().parents[2]

    def _key(p: Path) -> str:
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    manifest = {
        "seed": SEED,
        "files": {
            _key(Path(p)): {
                "sha256": sha256_file(p),
                "bytes": Path(p).stat().st_size,
            }
            for p in sorted(map(str, paths))
        },
    }
    if extra:
        manifest.update(extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


# ---------------------------------------------------------------------------
# Step 1 -- the real geographic skeleton
# ---------------------------------------------------------------------------


def load_pincode_skeleton(
    path: str | Path = "data/external/india_pincodes.csv",
) -> pd.DataFrame:
    """Collapse the India Post directory to one row per real pincode.

    Returns columns: pincode, city, district, state, n_post_offices, tier.
    Nothing here is invented -- pincode, city, district and state come straight
    from the directory. ``tier`` is a rule-based assignment (see DATA_CARD.md).
    """
    raw = pd.read_csv(path, dtype=str)
    raw = raw.dropna(subset=["Pincode", "City", "DistrictsName", "State"])
    raw["Pincode"] = raw["Pincode"].str.strip()
    raw = raw[raw["Pincode"].str.fullmatch(r"\d{6}")]

    n_po = raw.groupby("Pincode")["PostOfficeName"].size().rename("n_post_offices")
    skel = (
        raw.drop_duplicates(subset=["Pincode"])
        .set_index("Pincode")
        .join(n_po)
        .reset_index()
        .rename(
            columns={
                "Pincode": "pincode",
                "City": "city",
                "DistrictsName": "district",
                "State": "state",
            }
        )[["pincode", "city", "district", "state", "n_post_offices"]]
    )

    skel["city"] = skel["city"].str.strip()
    skel["district"] = skel["district"].str.strip()
    skel["state"] = skel["state"].str.strip()

    # Tier: curated metro / tier-1 lists, then district-headquarters towns as
    # tier-2, everything else tier-3.
    is_metro = skel["city"].isin(METRO_CITIES)
    is_t1 = skel["city"].isin(TIER1_CITIES)
    is_hq = skel["city"].str.casefold() == skel["district"].str.casefold()
    skel["tier"] = np.select(
        [is_metro, is_t1, is_hq], ["metro", "tier_1", "tier_2"], default="tier_3"
    )
    return skel.sort_values("pincode").reset_index(drop=True)


def assign_pincode_priors(
    skeleton: pd.DataFrame, evidence: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Give every city -- and then every pincode -- a latent RTO propensity.

    City priors are drawn inside the published 18-35% range, positioned by tier.
    The two cities named in ``evidence.yaml`` are pinned to their published
    values rather than sampled. Pincode priors jitter around their city.
    """
    lo = float(evidence["city_rto_range"]["min"]["value"])
    hi = float(evidence["city_rto_range"]["max"]["value"])
    span = hi - lo

    cities = (
        skeleton.groupby(["state", "city"], as_index=False)["tier"]
        .agg(lambda s: s.mode().iat[0])
        .sort_values(["state", "city"])
        .reset_index(drop=True)
    )

    # Beta draw centred on the tier position, then mapped into [lo, hi].
    pos = cities["tier"].map(TIER_POSITION).to_numpy(dtype=float)
    conc = 12.0
    a = np.clip(pos * conc, 0.2, None)
    b = np.clip((1.0 - pos) * conc, 0.2, None)
    u = rng.beta(a, b)
    cities["city_rto_prior"] = lo + span * u

    # Pin the published anchors.
    for city, which in ANCHOR_CITIES.items():
        mask = cities["city"] == city
        if mask.any():
            cities.loc[mask, "city_rto_prior"] = (
                lo if which == "min" else hi
            )

    out = skeleton.merge(cities[["state", "city", "city_rto_prior"]],
                         on=["state", "city"], how="left")

    # Pincode-level jitter around the city prior, clipped to the published range.
    jitter = rng.normal(0.0, 0.022, size=len(out))
    out["pincode_rto_prior"] = np.clip(out["city_rto_prior"] + jitter, lo, hi)
    return out


def sample_serviced_pincodes(
    priors: pd.DataFrame, rng: np.random.Generator, n_pincodes: int = 3500
) -> pd.DataFrame:
    """Pick the pincode footprint a single mid-size D2C merchant actually ships to.

    A real merchant does not see uniform draws from all 23,916 pincodes; volume
    concentrates in metros and tier-1. We sample a footprint, then attach a
    volume weight so order counts per pincode are heavy-tailed.
    """
    tier_counts = {
        t: int(round(n_pincodes * w)) for t, w in TIER_VOLUME_WEIGHT.items()
    }
    chosen = []
    for tier in TIER_ORDER:
        pool = priors.index[priors["tier"] == tier].to_numpy()
        take = min(tier_counts[tier], pool.size)
        chosen.append(rng.choice(pool, size=take, replace=False))
    idx = np.sort(np.concatenate(chosen))
    foot = priors.loc[idx].reset_index(drop=True)

    # Heavy-tailed volume weights within tier.
    tier_w = foot["tier"].map(TIER_VOLUME_WEIGHT).to_numpy(dtype=float)
    gamma = rng.gamma(shape=1.1, scale=1.0, size=len(foot))
    w = tier_w * gamma
    foot["volume_weight"] = w / w.sum()
    return foot


# ---------------------------------------------------------------------------
# Step 2 -- the customer population
# ---------------------------------------------------------------------------


def build_customers(
    footprint: pd.DataFrame,
    rng: np.random.Generator,
    faker,
    n_customers: int = N_CUSTOMERS,
) -> pd.DataFrame:
    """Build a persistent customer population.

    Each customer carries two latents that drive labels but are never features:

    ``reliability_z``     higher = more likely to actually take delivery
    ``address_quality``   truth about the address; the model only sees the
                          *string*, and the extracted features are a noisy read
                          of this latent.
    """
    pin_idx = rng.choice(
        len(footprint), size=n_customers, replace=True,
        p=footprint["volume_weight"].to_numpy()
    )
    home = footprint.iloc[pin_idx].reset_index(drop=True)

    reliability = rng.normal(0.0, 1.0, size=n_customers)
    # Address quality is worse on average outside metros.
    tier_shift = home["tier"].map(
        {"metro": 0.55, "tier_1": 0.35, "tier_2": 0.05, "tier_3": -0.25}
    ).to_numpy(dtype=float)
    aq = sigmoid(rng.normal(0.55, 0.95, size=n_customers) + tier_shift)

    # Purchase propensity -> heavy-tailed orders per customer.
    propensity = rng.gamma(shape=1.15, scale=1.0, size=n_customers)

    cust = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:06d}" for i in range(n_customers)],
            "customer_name": [faker.name() for _ in range(n_customers)],
            "phone": [faker.msisdn()[-10:] for _ in range(n_customers)],
            "home_pincode": home["pincode"].to_numpy(),
            "home_city": home["city"].to_numpy(),
            "home_district": home["district"].to_numpy(),
            "home_state": home["state"].to_numpy(),
            "home_tier": home["tier"].to_numpy(),
            "pincode_rto_prior": home["pincode_rto_prior"].to_numpy(),
            "reliability_z": reliability,
            "address_quality": aq,
            "propensity": propensity / propensity.sum(),
        }
    )

    # Signup date: uniform over the two years preceding the window end, so that
    # account age varies. Clipped later against each customer's first order.
    days_back = rng.integers(0, 730, size=n_customers)
    cust["signup_date"] = WINDOW_END - pd.to_timedelta(days_back, unit="D")

    # Persistent primary address string, degraded according to address_quality.
    cust["address_line"] = [
        _make_address(faker, rng, q, city)
        for q, city in zip(cust["address_quality"], cust["home_city"])
    ]
    return cust


_GIBBERISH_CHARS = list("bcdfghjklmnpqrstvwxz")


def _make_address(faker, rng: np.random.Generator, quality: float, city: str) -> str:
    """Render an address string whose *observable* quality tracks the latent.

    High quality  -> house number, street, landmark, locality, city.
    Low quality   -> short, no house number, no landmark, sometimes a mangled
                     token typed by a customer in a hurry.
    """
    parts: list[str] = []

    if rng.random() < 0.15 + 0.80 * quality:
        parts.append(f"H No {faker.building_number()}")
    if rng.random() < 0.10 + 0.85 * quality:
        parts.append(faker.street_name())
    else:
        parts.append(rng.choice(["gali no 4", "main road", "bazar", "colony"]))
    if rng.random() < 0.05 + 0.75 * quality:
        parts.append(
            "Near " + rng.choice(
                ["City Hospital", "Bus Stand", "SBI Bank", "Govt School",
                 "Water Tank", "Post Office", "Petrol Pump", "Shiv Mandir"]
            )
        )
    # Locality built from the *real* city, not a Faker city -- a fabricated city
    # name inside a real pincode's address would be an obvious tell.
    if rng.random() < 0.30 + 0.60 * quality:
        parts.append(
            f"{city} " + rng.choice(
                ["Nagar", "Colony", "East", "West", "Extension", "Sector 5",
                 "Puram", "Layout"]
            )
        )

    # A hurried / mangled token appears mostly at the low-quality end.
    if rng.random() < 0.32 * (1.0 - quality):
        n = int(rng.integers(5, 9))
        parts.insert(
            int(rng.integers(0, len(parts) + 1)),
            "".join(rng.choice(_GIBBERISH_CHARS, size=n)),
        )

    parts.append(city)
    return ", ".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Step 3 -- the order stream
# ---------------------------------------------------------------------------


def _festive_frames() -> pd.DataFrame:
    rows = [
        {
            "start": pd.Timestamp(s),
            "end": pd.Timestamp(e),
            "volume_mult": vm,
            "discount_boost": db,
        }
        for s, e, vm, db in FESTIVE_WINDOWS
    ]
    return pd.DataFrame(rows)


def _day_weights(days: pd.DatetimeIndex) -> np.ndarray:
    """Daily order-volume weights: weekly seasonality x festive multipliers."""
    w = np.ones(len(days), dtype=float)
    # Weekend lift, Monday dip.
    dow = days.dayofweek.to_numpy()
    w *= np.select([dow >= 5, dow == 0], [1.25, 0.92], default=1.0)
    # Slow secular growth over the window.
    w *= np.linspace(0.85, 1.20, len(days))
    fest = _festive_frames()
    for _, row in fest.iterrows():
        m = np.asarray((days >= row["start"]) & (days <= row["end"]))
        w[m] *= row["volume_mult"]
    return w / w.sum()


def generate_orders(
    customers: pd.DataFrame,
    footprint: pd.DataFrame,
    rng: np.random.Generator,
    faker,
    n_orders: int = N_ORDERS,
) -> pd.DataFrame:
    """Generate the raw order stream (no labels yet).

    Only checkout-time attributes are produced here. Labels are assigned in
    :func:`assign_labels` after the causal features exist, because some of them
    (velocity, account age, ETA) are genuine label drivers.
    """
    days = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    day_p = _day_weights(days)

    day_idx = rng.choice(len(days), size=n_orders, replace=True, p=day_p)
    # Intraday time: evening-heavy.
    secs = np.clip(rng.normal(19.0, 4.2, size=n_orders), 0.0, 23.999) * 3600.0
    order_ts = days.to_numpy()[day_idx] + pd.to_timedelta(secs, unit="s")

    cust_idx = rng.choice(
        len(customers), size=n_orders, replace=True,
        p=customers["propensity"].to_numpy()
    )

    cat_share = np.array([c.share for c in CATEGORIES], dtype=float)
    cat_share = cat_share / cat_share.sum()
    cat_idx = rng.choice(len(CATEGORIES), size=n_orders, replace=True, p=cat_share)
    cat_mu = np.array([c.log_value_mu for c in CATEGORIES])[cat_idx]
    cat_sigma = np.array([c.log_value_sigma for c in CATEGORIES])[cat_idx]
    cat_cod = np.array([c.cod_logit for c in CATEGORIES])[cat_idx]
    cat_name = np.array([c.name for c in CATEGORIES])[cat_idx]

    order_value = np.round(np.exp(rng.normal(cat_mu, cat_sigma)), 0)
    order_value = np.clip(order_value, 99.0, 60_000.0)

    df = pd.DataFrame(
        {
            "order_ts": order_ts,
            "customer_id": customers["customer_id"].to_numpy()[cust_idx],
            "category": cat_name,
            "order_value": order_value,
        }
    )
    df["_cust_idx"] = cust_idx

    # Festive flag + discount.
    fest = _festive_frames()
    festive = np.zeros(n_orders, dtype=bool)
    disc_boost = np.zeros(n_orders, dtype=float)
    ts = df["order_ts"].to_numpy()
    for _, row in fest.iterrows():
        m = (ts >= row["start"].to_datetime64()) & (ts <= row["end"].to_datetime64())
        festive |= m
        disc_boost = np.where(m, np.maximum(disc_boost, row["discount_boost"]), disc_boost)
    df["is_festive"] = festive
    df["discount_pct"] = np.clip(
        rng.beta(2.0, 6.0, size=n_orders) * 0.60 + disc_boost, 0.0, 0.75
    ).round(3)

    # Shipping pincode: usually home, sometimes an alternate address.
    alt = rng.random(n_orders) < 0.12
    alt_pin_idx = rng.choice(
        len(footprint), size=n_orders, replace=True,
        p=footprint["volume_weight"].to_numpy()
    )
    home = customers.iloc[cust_idx].reset_index(drop=True)
    altrow = footprint.iloc[alt_pin_idx].reset_index(drop=True)

    def _pick(alt_col: str, home_col: str) -> np.ndarray:
        return np.where(
            alt,
            altrow[alt_col].to_numpy(dtype=object),
            home[home_col].to_numpy(dtype=object),
        )

    df["pincode"] = _pick("pincode", "home_pincode")
    df["city"] = _pick("city", "home_city")
    df["district"] = _pick("district", "home_district")
    df["state"] = _pick("state", "home_state")
    df["pincode_tier"] = _pick("tier", "home_tier")
    df["_pincode_rto_prior"] = np.where(
        alt, altrow["pincode_rto_prior"], home["pincode_rto_prior"]
    )
    df["is_alternate_address"] = alt

    # Alternate addresses get their own (independently drawn) quality.
    alt_quality = sigmoid(rng.normal(0.30, 1.05, size=n_orders))
    df["_address_quality"] = np.where(
        alt, alt_quality, home["address_quality"].to_numpy()
    )
    alt_addr = np.asarray(
        [
            _make_address(faker, rng, q, c)
            for q, c in zip(alt_quality, df["city"].to_numpy())
        ],
        dtype=object,
    )
    df["address_line"] = np.where(
        alt, alt_addr, home["address_line"].to_numpy(dtype=object)
    )

    # Payment mode. Calibrated to the published COD share.
    cod_logit_raw = (
        cat_cod
        + df["pincode_tier"].map(
            {"metro": -0.45, "tier_1": -0.10, "tier_2": 0.30, "tier_3": 0.65}
        ).to_numpy(dtype=float)
        - 0.55 * (np.log(df["order_value"].to_numpy()) - np.log(800.0))
        + rng.normal(0.0, 0.70, size=n_orders)
    )
    return df, cod_logit_raw


def _finalise_payment_mode(
    df: pd.DataFrame, cod_logit_raw: np.ndarray, evidence: dict,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Draw COD vs prepaid, calibrated to the published COD share."""
    target = float(evidence["cod_share"]["value"])
    offset = solve_offset(cod_logit_raw, target)
    p_cod = sigmoid(cod_logit_raw + offset)
    df = df.copy()
    df["is_cod"] = rng.random(len(df)) < p_cod
    df["payment_mode"] = np.where(df["is_cod"], "COD", "PREPAID")
    return df


# ---------------------------------------------------------------------------
# Step 4 -- checkout-time features that are also label drivers
# ---------------------------------------------------------------------------


def add_checkout_features(
    df: pd.DataFrame, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Add the pre-label features: ETA, velocity, account age, value band.

    All of these are knowable at checkout. ``order_velocity_24h`` counts only
    *strictly earlier* orders by the same customer -- never the order itself.
    """
    df = df.sort_values(["order_ts", "customer_id"], kind="mergesort").reset_index(
        drop=True
    )
    df["order_id"] = [f"ORD{i:06d}" for i in range(len(df))]

    # Courier ETA: tier baseline, festive congestion, noise.
    base = df["pincode_tier"].map(TIER_ETA_DAYS).to_numpy(dtype=float)
    eta = base + df["is_festive"].to_numpy() * 1.1 + rng.gamma(1.6, 0.75, size=len(df))
    df["delivery_days_est"] = np.round(np.clip(eta, 1.0, 21.0), 1)

    # Orders by the same customer in the preceding 24 hours (strictly prior).
    ts = df["order_ts"].to_numpy().astype("datetime64[ns]")
    vel = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("customer_id", sort=False).indices.items():
        t = ts[idx]
        lower = np.searchsorted(t, t - np.timedelta64(24, "h"), side="left")
        vel[idx] = np.arange(len(t)) - lower
    df["order_velocity_24h"] = vel

    # Account age. Signup is clipped to at least one day before the first order.
    sign = customers.set_index("customer_id")["signup_date"]
    first_order = df.groupby("customer_id")["order_ts"].min()
    clip_to = first_order.dt.normalize() - pd.Timedelta(days=1)
    sign = pd.to_datetime(
        sign.combine(clip_to, lambda a, b: min(a, b) if pd.notna(b) else a)
    )
    df["signup_date"] = pd.to_datetime(df["customer_id"].map(sign))
    df["account_age_days"] = (
        (df["order_ts"] - df["signup_date"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0).round(2)

    df["order_value_band"] = value_band(df["order_value"].to_numpy())
    return df


# ---------------------------------------------------------------------------
# Step 5 -- calibrated labels
# ---------------------------------------------------------------------------


def build_risk_logit(
    df: pd.DataFrame, evidence: dict, rng: np.random.Generator
) -> np.ndarray:
    """Uncalibrated log-odds of RTO from the grounded drivers.

    The intercept is meaningless at this stage -- :func:`assign_labels` solves
    the per-cell offsets that pin the marginals to the published rates.
    """
    cat_mult = {
        k: float(v["value"]) for k, v in evidence["category_multipliers"].items()
        if k != "other"
    }
    default_mult = float(evidence["category_multipliers"]["other"]["value"])

    prior = df["_pincode_rto_prior"].to_numpy(dtype=float)
    city_term = COEF["city"] * (logit(prior) - logit(prior.mean()))

    rel = df["_reliability_z"].to_numpy(dtype=float)
    aq = df["_address_quality"].to_numpy(dtype=float)

    mult = df["category"].map(cat_mult).fillna(default_mult).to_numpy(dtype=float)

    eta = df["delivery_days_est"].to_numpy(dtype=float)
    age = df["account_age_days"].to_numpy(dtype=float)
    age_term = -np.log1p(age) / np.log1p(365.0)

    z = (
        city_term
        + COEF["reliability"] * (-rel)
        + COEF["address"] * (1.0 - aq)
        + COEF["category"] * np.log(mult)
        + COEF["discount"] * (df["discount_pct"].to_numpy(dtype=float) - 0.20)
        + COEF["festive"] * df["is_festive"].to_numpy(dtype=float)
        + COEF["delivery_days"] * (eta - eta.mean())
        + COEF["velocity"] * df["order_velocity_24h"].to_numpy(dtype=float)
        + COEF["account_age"] * (age_term - age_term.mean())
        + COEF["repeat_buyer"] * (df["order_velocity_24h"].to_numpy() > 0)
        + rng.normal(0.0, COEF["noise_sd"], size=len(df))
    )
    return z


def assign_labels(
    df: pd.DataFrame, evidence: dict, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """Calibrate the marginals to published rates, then draw Bernoulli labels.

    Twelve cells are solved independently: {COD, prepaid} x three value bands x
    {fashion, non-fashion}. Within each cell a single log-odds intercept is
    found by bisection, so the cell's expected rate lands on its published
    target while every other driver keeps contributing freely.

    Targets:

    * COD band rates come straight from ``evidence.yaml`` (0.25 / 0.28 / 0.24).
      This is what makes the order-value curve non-monotonic.
    * Fashion carries the published 40% rate as a ratio to the 26% COD rate;
      the non-fashion target in each cell is then whatever makes the cell's
      volume-weighted mean equal its band target. Nothing is double-counted.
    * Prepaid keeps the same relative shape, scaled to the published sub-2%.
    """
    z = build_risk_logit(df, evidence, rng)

    bands = evidence["order_value_bands"]
    cod_band_target = {
        "under_500": float(bands["under_500"]["value"]),
        "500_1000": float(bands["mid_500_1000"]["value"]),
        "1000_plus": float(bands["over_1000"]["value"]),
    }
    cod_overall = float(evidence["cod_rto_rate"]["value"])
    prepaid_overall = float(evidence["prepaid_rto_rate"]["value"])
    # Aim comfortably inside the published "<2%" bound.
    prepaid_target_level = 0.75 * prepaid_overall
    prepaid_band_target = {
        b: prepaid_target_level * (cod_band_target[b] / cod_overall)
        for b in VALUE_BANDS
    }

    # Published fashion rate, carried as a ratio so it composes with the bands.
    fashion_ratio = float(evidence["fashion_rto_rate"]["value"]) / cod_overall

    is_cod = df["is_cod"].to_numpy()
    band = df["order_value_band"].to_numpy()
    is_fashion = (df["category"].to_numpy() == "fashion")

    offsets: dict[str, float] = {}
    cell_targets: dict[str, float] = {}
    z_cal = z.copy()
    for cod_flag, targets, tag in (
        (True, cod_band_target, "COD"),
        (False, prepaid_band_target, "PREPAID"),
    ):
        for b in VALUE_BANDS:
            cell = (is_cod == cod_flag) & (band == b)
            t_band = targets[b]
            w_f = float(is_fashion[cell].mean()) if cell.any() else 0.0
            t_fashion = min(t_band * fashion_ratio, 0.95)
            # Whatever makes the cell's volume-weighted mean equal the band
            # target once fashion is pinned. No rate is counted twice.
            t_other = (
                (t_band - w_f * t_fashion) / (1.0 - w_f) if w_f < 1.0 else t_band
            )
            assert t_other > 0.0, (
                f"fashion share {w_f:.3f} in cell {tag}|{b} leaves no room for "
                "the published band rate; check evidence.yaml"
            )
            for label, m, t in (
                ("fashion", cell & is_fashion, t_fashion),
                ("other", cell & ~is_fashion, t_other),
            ):
                c = solve_offset(z[m], t)
                key = f"{tag}|{b}|{label}"
                offsets[key] = c
                cell_targets[key] = t
                z_cal[m] = z[m] + c

    p = sigmoid(z_cal)
    out = df.copy()
    out["_p_rto_true"] = p
    out["rto"] = (rng.random(len(out)) < p).astype(int)

    diagnostics = {
        "offsets": offsets,
        "cell_targets": cell_targets,
        "targets": {"cod": cod_band_target, "prepaid": prepaid_band_target},
        "fashion_ratio": fashion_ratio,
        "expected_cod_rate": float(p[is_cod].mean()),
        "expected_prepaid_rate": float(p[~is_cod].mean()),
        "bayes_optimal_mean_p": float(p.mean()),
    }
    return out, diagnostics


# ---------------------------------------------------------------------------
# Step 6 -- causal customer history (uses labels, strictly from the past)
# ---------------------------------------------------------------------------


def add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding per-customer history over *strictly earlier* orders.

    ``past_orders`` / ``past_rto_count`` / ``past_rto_rate`` are shifted by one
    within each customer, so the current order never contributes to its own
    features. First-ever orders get ``past_rto_rate = NaN`` and
    ``has_history = 0``; imputation is a modelling decision, made in 02/03.
    """
    df = df.sort_values(["order_ts", "order_id"], kind="mergesort").reset_index(
        drop=True
    )
    g = df.groupby("customer_id", sort=False)["rto"]
    past_orders = g.cumcount().to_numpy()
    past_rto = (g.cumsum() - df["rto"]).to_numpy()

    df["past_orders"] = past_orders
    df["past_rto_count"] = past_rto
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(past_orders > 0, past_rto / np.maximum(past_orders, 1), np.nan)
    df["past_rto_rate"] = rate
    df["has_history"] = (past_orders > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 7 -- chronological split
# ---------------------------------------------------------------------------


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, pd.DataFrame]:
    """Split along time. Never random -- a random split leaks the future."""
    df = df.sort_values(["order_ts", "order_id"], kind="mergesort").reset_index(
        drop=True
    )
    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {
        "train": df.iloc[:n_train].copy(),
        "val": df.iloc[n_train : n_train + n_val].copy(),
        "test": df.iloc[n_train + n_val :].copy(),
    }


# ---------------------------------------------------------------------------
# Calibration check -- the assertions, in one place
# ---------------------------------------------------------------------------


def calibration_report(orders: pd.DataFrame) -> dict:
    """Realised rates that must match the published Indian statistics."""
    cod = orders[orders["is_cod"]]
    prepaid = orders[~orders["is_cod"]]
    band_rate = cod.groupby("order_value_band")["rto"].mean()
    return {
        "n_orders": int(len(orders)),
        "cod_share": float(orders["is_cod"].mean()),
        "cod_rto_rate": float(cod["rto"].mean()),
        "prepaid_rto_rate": float(prepaid["rto"].mean()),
        "overall_rto_rate": float(orders["rto"].mean()),
        "cod_rto_by_band": {b: float(band_rate.get(b, np.nan)) for b in VALUE_BANDS},
        "cod_rto_by_category": {
            str(k): float(v) for k, v in cod.groupby("category")["rto"].mean().items()
        },
        "cod_rto_fashion": float(cod.loc[cod["category"] == "fashion", "rto"].mean()),
        **_city_spread(cod, min_orders=min(100, max(20, len(cod) // 200))),
    }


def _city_spread(cod: pd.DataFrame, min_orders: int) -> dict:
    """COD RTO rate by city, over cities with enough volume to be readable.

    Published city figures are COD rates, so this is measured on COD only.
    """
    g = cod.groupby("city")["rto"]
    rates = g.mean()[g.size() >= min_orders]
    if rates.empty:
        return {"city_rto_min": np.nan, "city_rto_max": np.nan, "n_cities_scored": 0}
    return {
        "city_rto_min": float(rates.min()),
        "city_rto_max": float(rates.max()),
        "n_cities_scored": int(len(rates)),
        "city_min_name": str(rates.idxmin()),
        "city_max_name": str(rates.idxmax()),
    }


def assert_calibrated(orders: pd.DataFrame) -> dict:
    """Hard assertions. These fail the build, they do not print a warning."""
    rep = calibration_report(orders)

    assert 0.24 <= rep["cod_rto_rate"] <= 0.28, (
        f"COD RTO rate {rep['cod_rto_rate']:.4f} outside published 26% +/- 2pp"
    )
    assert rep["prepaid_rto_rate"] < 0.02, (
        f"Prepaid RTO rate {rep['prepaid_rto_rate']:.4f} not below published 2%"
    )
    assert rep["cod_rto_by_band"]["500_1000"] > rep["cod_rto_by_band"]["1000_plus"], (
        "Order-value RTO curve is monotonic; published data peaks in the "
        f"Rs500-1000 impulse band ({rep['cod_rto_by_band']})"
    )
    assert rep["cod_rto_by_band"]["500_1000"] > rep["cod_rto_by_band"]["under_500"], (
        f"Rs500-1000 band is not the peak ({rep['cod_rto_by_band']})"
    )
    assert 0.58 <= rep["cod_share"] <= 0.66, (
        f"COD share {rep['cod_share']:.4f} outside the published 60-65% band"
    )
    assert rep["cod_rto_fashion"] >= 0.38, (
        f"Fashion COD RTO {rep['cod_rto_fashion']:.4f} below the published 40%"
    )
    return rep


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

# Written to data/raw -- what a merchant's order log would plausibly contain.
RAW_COLUMNS = [
    "order_id", "order_ts", "customer_id", "customer_name", "phone",
    "address_line", "pincode", "city", "district", "state", "pincode_tier",
    "category", "order_value", "order_value_band", "discount_pct",
    "payment_mode", "is_cod", "is_festive", "is_alternate_address",
    "delivery_days_est", "signup_date", "account_age_days",
    "order_velocity_24h", "rto",
]

# Latents: audit only. Using any of these as a feature is leakage by definition.
LATENT_COLUMNS = [
    "order_id", "_pincode_rto_prior", "_reliability_z", "_address_quality",
    "_p_rto_true",
]

# Written to data/processed -- raw plus causal history and address quality.
HISTORY_COLUMNS = [
    "past_orders", "past_rto_count", "past_rto_rate", "has_history",
]


def build_dataset(
    evidence_path: str | Path = "config/evidence.yaml",
    pincode_path: str | Path = "data/external/india_pincodes.csv",
    seed: int = SEED,
    n_orders: int = N_ORDERS,
    n_customers: int = N_CUSTOMERS,
) -> dict:
    """Run the whole generator end to end. Returns every intermediate artefact.

    Deterministic given ``seed``: one RNG, drawn in a fixed order, plus a Faker
    instance seeded from the same value.
    """
    from faker import Faker

    evidence = load_evidence(evidence_path)
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    faker = Faker("en_IN")
    faker.seed_instance(seed)

    skeleton = load_pincode_skeleton(pincode_path)
    priors = assign_pincode_priors(skeleton, evidence, rng)
    footprint = sample_serviced_pincodes(priors, rng)
    customers = build_customers(footprint, rng, faker, n_customers=n_customers)

    orders, cod_logit_raw = generate_orders(
        customers, footprint, rng, faker, n_orders=n_orders
    )
    orders = _finalise_payment_mode(orders, cod_logit_raw, evidence, rng)

    # Attach the customer latents that drive labels (dropped before writing raw).
    cust_by_id = customers.set_index("customer_id")
    orders["_reliability_z"] = orders["customer_id"].map(cust_by_id["reliability_z"])
    orders["customer_name"] = orders["customer_id"].map(cust_by_id["customer_name"])
    orders["phone"] = orders["customer_id"].map(cust_by_id["phone"])

    orders = add_checkout_features(orders, customers, rng)
    orders, diagnostics = assign_labels(orders, evidence, rng)
    orders = add_customer_history(orders)
    orders = orders.drop(columns=["_cust_idx"])

    report = assert_calibrated(orders)
    splits = chronological_split(orders)

    return {
        "evidence": evidence,
        "skeleton": skeleton,
        "priors": priors,
        "footprint": footprint,
        "customers": customers,
        "orders": orders,
        "splits": splits,
        "calibration": report,
        "diagnostics": diagnostics,
    }
