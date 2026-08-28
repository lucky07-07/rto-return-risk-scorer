# Return-Risk Scorer

**Stops Indian online sellers losing money on cash-on-delivery orders that come back undelivered.**

Razorpay AI Buildathon 2026 · Track 02, AI Risk Manager

> 🔗 **Live demo:** _deploying, URL will be added here_
> 📄 [How it was built and tested](#for-the-technically-minded) · [What it gets wrong](#what-it-gets-wrong)

---

## The problem, in one paragraph

About 6 in 10 online orders in India are paid cash on delivery. Roughly **1 in 4 of those
never gets delivered**. The customer isn't home, changes their mind, or refuses at the
door. The seller has already paid to ship it out and now pays to ship it back, plus
packaging and handling. Nothing is collected. Across Indian e-commerce that's an estimated
₹8,000 crore a year.

This tool looks at an order **before it ships** and answers one question. Is this one
likely to come back?

## What it does

You give it an order. It gives you four things, in plain English.

| | |
|---|---|
| **How risky** | A percentage chance this order comes back |
| **What it costs** | The rupees you'd expect to waste shipping it |
| **What to do** | Allow cash on delivery, add a small fee, or ask for online payment |
| **Why** | The actual reasons, like a vague address or a customer who has returned before |

It only ever advises. It never blocks a customer, takes money, or contacts anyone.

## What it's worth

Tested on 10,000 orders the model had never seen.

| | |
|---|---|
| **Money saved** | **₹29,481** against the best you could do with no model at all |
| **Cost of a lazy cut-off** | **₹20,835** — what you'd waste flagging at 50% out of habit instead of at the point where money is actually minimised |
| **Catches** | About 1 in 4 of the orders that would have come back |
| **When it flags an order** | It's right a little under half the time |

That last row matters and it's stated deliberately. This is not a tool that is right every
time. It's a tool that, once you add up the wins and the mistakes in rupees, leaves the
seller better off.

---

## Demo

Three real orders, scored by the live model. Switch between them in the dropdown.

**A risky order, so the tool says ask for online payment**

![The tool showing a BLOCK decision on a high-risk fashion order](docs/screenshots/block.png)

**A middling order, so the tool says allow cash on delivery but add a fee**

![The tool showing a REVIEW decision on a moderate-risk order](docs/screenshots/review.png)

**A safe order, so the tool lets it through**

![The tool showing an ALLOW decision on a low-risk order](docs/screenshots/allow.png)

The blue box at the top of each is written by Google Gemini, which turns the raw numbers
into something a shop owner can read. If the Gemini key is missing or its free quota is used
up, the app writes the same summary itself and carries on working.

## How it fits together

![Architecture, from generated data through to the merchant-facing page](docs/architecture_diagram.png)

Editable source, [`docs/architecture_diagram.drawio`](docs/architecture_diagram.drawio)

---

## Getting started

Everything below is a real command. Nothing to fill in except your own API key, which is
optional.

**1. Clone and enter the project**

```bash
git clone https://github.com/lucky07-07/rto-return-risk-scorer.git
cd rto-return-risk-scorer
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

On macOS or Linux use `source .venv/bin/activate` instead.

**3. Install the dependencies**

```bash
pip install --only-binary=:all: -r requirements.txt
```

The `--only-binary` flag matters. Without it pip tries to compile CatBoost from source and
fails after several minutes.

**4. Add a Gemini key, optional**

```bash
cp .env.example .env
```

Open `.env` and paste a free key from [Google AI Studio](https://aistudio.google.com/apikey).
Skip this and everything still runs, you just get the built-in summaries instead of
Gemini-written ones.

**5. Run the demo**

```bash
uvicorn api.main:app --reload
```

Open <http://127.0.0.1:8000>. The page loads with a real order already filled in, so a
decision appears immediately. Use the dropdown to switch between a safe order, a middling
one and a risky one. Interactive API documentation is at <http://127.0.0.1:8000/docs>.

**6. Run the tests**

```bash
pytest -q
```

49 tests covering data calibration, leakage guards and the API.

**7. Rebuild the model from scratch, optional**

The trained model ships with the repository, so this is only needed to reproduce the whole
pipeline. Run the notebooks in order, `01` through `05`. Notebook `01` generates the data
everything else depends on.

```bash
jupyter lab
```

---

## Project structure

```
api/            Web service and the one-page demo UI
app/            Alternative Streamlit interface, optional
config/         Published statistics the generated data is calibrated against
data/external/  Real India Post pincode directory, 39,736 post offices
docs/           Architecture diagram and screenshots
models/         The trained model plus its decision cut-offs, one file
notebooks/      The full build, 01 to 05, run in order
reports/        Every chart and every metric, written by the notebooks
src/            The reusable code the notebooks and the web service both import
tests/          Calibration, leakage and API tests
```

---

## What it gets wrong

Nothing here is hidden. Each limitation gets a plain-English version first.

**The data is generated, not real.**
There is no public dataset of Indian cash-on-delivery orders with return outcomes, so the
orders were built to match published Indian statistics rather than taken from a real shop.
The geography is real, the India Post directory of 39,736 post offices. The statistics it is
tuned to are real and published. The orders themselves are not. **No claim is made that this
has been validated on real merchant data.**

**It helps least the sellers who need it most.**
A seller whose orders are almost all cash on delivery gets the weakest performance, because
the single most useful clue, whether the customer chose to pay cash, is then the same for
every order they have. It still beats doing nothing, but such a seller should expect the
lower end of the range.

**The "allow" tier never actually fires for a cash order.**
Every cash-on-delivery order gets either a fee or a request for online payment. Nothing is
waved straight through. That falls out of the economics assumed for the fee, and it is
reported rather than tidied away.

**Two numbers in the cost model are guesses.**
The fee amount, and how many customers walk away when shown a fee, have no published source.
The main threshold does not depend on them and is solid. The three-tier split does depend on
them and should be treated as provisional.

**A simple model did just as well.**
Plain logistic regression matched everything more sophisticated. Tuning made the model
slightly worse on unseen orders. Both findings are reported in full rather than buried.

---

## For the technically minded

The section above is deliberately free of jargon. The rigour is all still here.

| Document | What's in it |
|---|---|
| [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) | Metrics, cost model and thresholds fixed **before** any model was trained, then git-tagged |
| [`DATA_CARD.md`](DATA_CARD.md) | What is real, what is generated, the full schema, and every limitation |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design, feature groups and the reproducibility contract |
| [`WHAT_BROKE.md`](WHAT_BROKE.md) | 21 entries on what went wrong and how it was fixed, including the embarrassing ones |

**Headline numbers**, all backed by files in `reports/results/`.

| | Test set | Validation |
|---|---|---|
| PR-AUC | 0.386, which is 2.62× the no-skill floor | 0.397 |
| ROC-AUC | 0.806 | 0.810 |
| Brier score | 0.106 | 0.109 |
| Expected calibration error | 0.012 | 0.013 |
| Precision, recall, F1 | 0.458, 0.269, 0.339 | — |

The Bayes ceiling for this data is a PR-AUC of 0.555, so the model captures about 62% of
what is achievable. A score above the ceiling would be evidence of a leak rather than of
skill, and the notebooks assert on it.

**Method in brief.**

- 50,000 orders split by time, 70/10/20. Never randomly, which would leak the future.
- Generated data calibrated against published Indian statistics and enforced by test rather
  than asserted in prose. `pytest tests/test_calibration.py` fails the build if the return
  rates drift.
- The order-value curve is deliberately non-monotonic. Risk peaks in the ₹500 to ₹1,000 band
  and falls above ₹1,000, which is what the published data shows and the opposite of the
  intuitive assumption.
- Pincode history is target-encoded out of fold. Doing it the naive way inflates the
  apparent score by 0.170 AUC, which notebook `02` measures rather than assumes.
- 11 models benchmarked on identical folds, then Optuna and FLAML compared head to head on
  an identical wall-clock budget.
- The test set was opened once, at settings frozen on validation beforehand.
- Performance is reported across the full 18% to 35% return range seen across Indian cities,
  so behaviour under a shifted base rate is measured rather than assumed.

---

## Defence only

This system reads order details and returns a number, a recommendation and an explanation.

It has no code path that captures a payment, issues a refund, blocks an account, cancels an
order or contacts a customer. The recommendation is advice. Acting on it is the seller's
decision. There is nothing here that could be repurposed to commit fraud.

---

## Licence

[MIT](LICENSE)

## Author

Anil Kumar
