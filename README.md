# Calibration Decomposition — Measurable Reliability for Probabilistic Classifiers

> *Reporting a single ECE number after temperature scaling is not a calibration diagnosis. It is one measurement at one distribution. This artifact does the diagnosis.*

---

## What this is

A systematic decomposition of probabilistic miscalibration into three distinct regimes — overconfidence, underconfidence, and class imbalance — with optimised temperature scaling applied to each and bin-level reliability tables showing exactly where and why calibration succeeds or fails.

Part of a series on measurable reliability in ML. See also: [Split Conformal Prediction](https://github.com/mariam-elelidy/Mathematical-Reliability-for-ML-Predictions) · [Assumption Stress Harness](https://github.com/mariam-elelidy/Assumption-Stress-Harness) · [Influence & Stability Analysis](https://github.com/mariam-elelidy/Influence-Stability-Analysis-for-ML-Predictions).

---

## Core findings (n=5000, bins=10, seed=42)

| Scenario | T* | ECE before | ECE after | Δ ECE | Works? |
|---|---|---|---|---|---|
| Overconfidence | 0.1933 | 0.159 | **0.004** | −97.4% | ✓ fully |
| Underconfidence | 4.3911 | 0.169 | **0.010** | −94.1% | ✓ fully |
| Class Imbalance | 0.5875 | 0.193 | **0.151** | −22.0% | ⚠ partial |

**The imbalance scenario is the critical one.** Global ECE improves by 22% while bins 0.5–1.0 become *more* overconfident after calibration. Minority-class ECE remains 0.208. A single global ECE number would not expose this.

---

## Quick start

```bash
pip install numpy scipy

# Defaults: n=5000, bins=10, seed=42, positive-class rate=0.15
python Calibration_decomposition.py

# Stronger imbalance
python Calibration_decomposition.py --pi 0.05 --n 10000 --bins 15
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 5000 | Samples per scenario |
| `--bins` | 10 | Reliability diagram bins |
| `--seed` | 42 | Random seed |
| `--pi` | 0.15 | Positive-class prevalence (imbalance scenario) |

---

## How it works

```
For each scenario (overconf / underconf / imbalance):
  1. Generate logits with controlled mismatch
  2. Compute p_raw = sigmoid(logits)
  3. Measure ECE, NLL, Brier (before)
  4. Optimise T* = argmin_T NLL(temperature_scale(p, T))   ← not hardcoded
  5. Apply p_cal = sigmoid(logits / T*)
  6. Measure ECE, NLL, Brier (after)
  7. Build bin-level reliability table with gap direction
  8. For imbalance: compute class-conditional ECE separately
```

T* is **optimised via scalar minimisation**, not hardcoded. A hardcoded T in the wrong direction (e.g. T=1.7 on an overconfident model) increases ECE. The sign of the correction matters as much as the magnitude.

---

## Key findings

**T* direction encodes regime.** T*=0.193 (overconfident), T*=4.391 (underconfident), T*=0.588 (imbalanced). Before inspecting any bins, T* alone identifies which regime the model is in and which direction the fix should go.

**ECE and NLL diverge under underconfidence.** ECE drops 94%; NLL drops only 16%. This is not a metric inconsistency. ECE measures confidence–frequency alignment; NLL measures full probabilistic quality. A model that was hedging at 0.5 passes the ECE test but cannot provide strong probabilistic guidance — the logit signal never separated the classes. Reporting only ECE improvement overstates the fix.

**Class imbalance creates a structural failure in global ECE.** Temperature scaling minimises global NLL, which is dominated by the majority class (85% of data). T* is chosen to correct majority-class predictions, which partially miscorrects minority-class predictions. Bins 0.5–1.0 are *more* overconfident after calibration. This is not a tuning error — T* is globally optimal. The failure is structural.

**A hardcoded T worsens overconfidence.** The original version of this artifact used T=1.7. For the overconfidence scenario (which needs T<1), this produced ΔECE = +0.056 — ECE increased. Calibration without regime identification is worse than no calibration.

---

## Outputs

| Output | What it answers |
|---|---|
| T* per scenario | "Which regime is this? Which direction?" |
| Metrics tensor [ECE, NLL, Brier] before/after | "How much did each metric improve?" |
| ECE–NLL divergence | "Did the logits have discriminative content to begin with?" |
| Bin reliability table + direction | "Where is the miscalibration? Overconf or underconf per bin?" |
| Class-conditional ECE | "Is global calibration hiding minority-class failure?" |
| Summary tensor [3 scenarios × 6 metrics] | "How does calibration repair compare across regimes?" |

---

## Relation to the reliability series

| Artifact | Reliability question |
|---|---|
| Split conformal prediction | Does this interval contain the truth ≥90% of the time? |
| Assumption stress harness | Does coverage hold under distribution shift? |
| Influence & stability | Which training points drive predictions? |
| **This artifact** | Are predicted probabilities trustworthy for decisions? |

Each layer addresses a different dimension. A model can be well-calibrated yet have unstable predictions under training perturbation — and vice versa.

---

## Repository layout

```
├── README.md                     ← this file
├── Calibration_decomposition.py  ← implementation
├── output.txt                    ← annotated run output with observations
└── writeup.md                    ← full technical writeup
```

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Kull, M., et al. (2019). Beyond temperature scaling. *NeurIPS*.

---
