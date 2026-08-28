---
title: COD Return-Risk Scorer
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predicts which cash-on-delivery orders come back undelivered
---

# COD Return-Risk Scorer

Scores an Indian cash-on-delivery order before it ships and says whether to allow
cash on delivery, add a fee, or ask for online payment. Shows the risk, the rupees
at stake, the decision and the reasons, with a plain-English summary on top.

Source and full methodology: https://github.com/lucky07-07/rto-return-risk-scorer

**This runs the real trained model**, not a simplified stand-in. Check `/health` for
the model's SHA-256 and its frozen decision cut-offs.

Trained on synthetic orders built on the real India Post pincode directory and
calibrated against published Indian return statistics. No claim is made that it has
been validated on real merchant data.
