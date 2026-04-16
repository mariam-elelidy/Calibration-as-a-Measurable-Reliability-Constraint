# Calibration as a Measurable Reliability Constraint

**Author:** Mariam Mohamed Elelidy  
**Topic:** Calibration · Decision Reliability · Probabilistic Classification

---

## TL;DR

Calibration is not a cosmetic fix applied after training. It is a constraint on the relationship between predicted probabilities and empirical frequencies — one that can be satisfied, violated, or partially repaired depending on the structure of the miscalibration.

This artifact decomposes miscalibration into three distinct regimes (overconfidence, underconfidence, class imbalance), applies optimal temperature scaling to each, and measures the resulting bin-level reliability. The key finding: temperature scaling fully repairs symmetric miscalibration but fails structurally under class imbalance — a silent failure that global ECE conceals.

---

## 1. Problem

Given a probabilistic classifier $f_\theta(x) = \hat{p}(y=1 \mid x) \in [0,1]$, perfect calibration requires:

$$\mathbb{P}(y=1 \mid f_\theta(x) = p) = p \quad \forall p \in [0,1]$$

Most evaluations report accuracy $\mathbb{E}[\mathbf{1}(\hat{p} > 0.5 = y)]$. This ignores the question that matters for decisions: when the model says 70%, how often is it actually right?

A classifier can achieve high accuracy while being systematically overconfident (it says 90%, truth is 60%) or underconfident (it says 55%, truth is 85%). Both are unacceptable failure modes in clinical or high-stakes decision-making — and they require opposite corrections.

Most calibration pipelines report a single ECE number and a reliability plot, then apply a post-hoc fix. This is not a diagnosis. It answers "is the model calibrated?" without answering "why not?" and "what will the fix actually do?"

---

## 2. Testable Claims

**Primary:** Miscalibration decomposes into at least three measurable regimes that require different corrections.

**Secondary:** Temperature scaling (with optimised T*) fully repairs symmetric miscalibration (overconfidence, underconfidence) but fails structurally under class imbalance — global ECE improves while minority-class ECE remains elevated.

**Diagnostic claim:** T* direction ($< 1$ vs $> 1$) is a sufficient regime indicator; ECE and NLL diverge under underconfidence in a way that reveals whether the logit signal had discriminative content to begin with.

---

## 3. Method

### Data generation

For each scenario, generate synthetic logits:

$$z_i \sim \mathcal{N}(\mu_{\text{regime}}, \sigma^2_{\text{regime}})$$

Convert to probabilities: $\hat{p}_i = \sigma(z_i)$, generate labels: $y_i \sim \text{Bernoulli}(p_i^{\text{true}})$.

The three regimes differ in how $p_i^{\text{true}}$ relates to $\hat{p}_i$:

| Regime | Mechanism | T* direction |
|---|---|---|
| **Overconfidence** | Logits scaled 3.5×; model too sharp | T* < 1 (soften) |
| **Underconfidence** | Logits scaled 0.5×; model hedges near 0.5 | T* > 1 (sharpen) |
| **Class imbalance** | $\mathbb{P}(y=1) = 0.15$; base rate mismatch | T* < 1 but partial |

### Calibration metrics

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

$$\text{NLL} = -\frac{1}{n} \sum_i \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

$$\text{Brier} = \frac{1}{n} \sum_i (\hat{p}_i - y_i)^2$$

### Temperature scaling with optimised T*

$$\hat{p}_i^{(T)} = \sigma\!\left(\frac{z_i}{T}\right), \quad T^* = \arg\min_T \text{NLL}(T)$$

T* is found via scalar minimisation (bounded, $T \in [0.1, 10]$). **Not hardcoded.** Applying a hardcoded T in the wrong direction actively worsens calibration — it is worse than no correction.

### Reliability table with bin-level direction

Each bin is classified as:
- **OVERCONF** if gap = acc − conf < −0.05
- **UNDERCONF** if gap > +0.05
- **calibrated** otherwise

The signed decomposition is more informative than the unsigned ECE per bin.

---

## 4. Results

### Scenario 1 — Overconfidence (T* = 0.1933)

| Metric | Before | After | Δ | Improvement |
|---|---|---|---|---|
| ECE | 0.15901 | 0.00409 | −0.15491 | **97.4%** |
| NLL | 0.20371 | 0.04381 | −0.15990 | 78.5% |
| Brier | 0.04336 | 0.01205 | −0.03131 | 72.2% |

Bin-level pattern: classic S-curve. All bins below 0.5 are OVERCONF (model says 0.3, truth is 0.05); all bins above 0.5 are UNDERCONF (model says 0.7, truth is ~1.0). This is characteristic of logit-scale overconfidence — a global fix is appropriate and effective. ECE drops from 0.159 to 0.004.

### Scenario 2 — Underconfidence (T* = 4.3911)

| Metric | Before | After | Δ | Improvement |
|---|---|---|---|---|
| ECE | 0.16869 | 0.01001 | −0.15867 | 94.1% |
| NLL | 0.80433 | 0.67866 | −0.12567 | **15.6%** |
| Brier | 0.27951 | 0.24284 | −0.03667 | 13.1% |

ECE improves by 94%; NLL improves by only 16%. This divergence is diagnostic: ECE measures calibration (probability–frequency alignment); NLL measures the full quality of the probabilistic prediction. A model that was hedging near 0.5 has a logit signal that never encoded discriminative information — temperature scaling cannot recover it. Lower ECE does not mean higher predictive quality.

### Scenario 3 — Class Imbalance (T* = 0.5875, P(y=1) = 0.15)

| Metric | Before | After | Δ | Improvement |
|---|---|---|---|---|
| ECE | 0.19312 | 0.15063 | −0.04249 | **22.0%** |
| NLL | 0.37352 | 0.34163 | −0.03190 | 8.5% |
| Brier | 0.11454 | 0.10701 | −0.00753 | 6.6% |

Class-conditional breakdown:

| | Before | After | Δ |
|---|---|---|---|
| Global ECE | 0.19312 | 0.15063 | −0.04249 |
| Minority ECE | 0.26608 | 0.20810 | −0.05798 |
| Majority ECE | 0.27394 | 0.21542 | −0.05851 |

Global ECE improves but minority-class ECE remains 0.208. Bins 0.5–1.0 are **more** overconfident after calibration than before. This is not a tuning failure — T* is globally optimal. The failure is structural: a global scalar correction cannot resolve a base-rate mismatch that operates differently across confidence bins.

### Summary tensor

```
[ECE_before, NLL_before, Brier_before, ECE_after, NLL_after, Brier_after]
Rows: [Overconfidence, Underconfidence, Class Imbalance]

[[0.15901  0.20371  0.04336  0.00409  0.04381  0.01205],
 [0.16869  0.80433  0.27951  0.01001  0.67866  0.24284],
 [0.19312  0.37352  0.11454  0.15063  0.34163  0.10701]]

Δ tensor [ΔECE, ΔNLL, ΔBrier]:
[[-0.15491  -0.15990  -0.03131],   # Overconfidence: T* fully repairs
 [-0.15867  -0.12567  -0.03667],   # Underconfidence: T* fully repairs
 [-0.04249  -0.03190  -0.00753]]   # Class imbalance: structural failure
```

---

## 5. Analysis

### T* direction is a regime indicator

T* = 0.193 (overconfident), T* = 4.391 (underconfident), T* = 0.588 (imbalanced + overconfident). The optimal temperature encodes the regime without requiring bin-level inspection first. In practice, computing T* before the reliability table is a useful first pass to understand what kind of correction the model needs.

### ECE–NLL divergence diagnoses logit quality

In Scenario 2, ECE drops 94% while NLL drops only 16%. This divergence is not a metric inconsistency — it reveals something about the model. ECE and NLL measure different things: ECE tests whether confidence is aligned with frequency; NLL tests the full probabilistic quality of the prediction. A model that was hedging at 0.5 passes the ECE test after calibration but still provides weak probabilistic guidance because its logits never separated the classes well. Reporting only ECE improvement after calibration would overstate the fix.

### Class imbalance creates a structural blind spot in global ECE

Temperature scaling minimises global NLL. Under class imbalance, the majority class (85% of data) dominates the optimisation objective. T* is chosen to best align majority-class predictions, which partially misaligns minority-class predictions. Bins 0.5–1.0 (where positive examples concentrate) become more overconfident after the correction. A model deployed for minority-class risk screening would be less reliable after calibration than before — while reporting a lower global ECE.

### Why hardcoded T fails

The original artifact used T = 1.7. For the overconfidence scenario, this produced ΔECE = +0.056 (ECE increased). T = 1.7 pushes probabilities toward 0.5 — the correct direction for underconfidence, the wrong direction for overconfidence. The error is not numerical; it is directional. Calibration applied without knowing the regime can actively harm reliability.

---

## 6. Failure Modes of Calibration Itself

| Failure mode | Mechanism | Detection |
|---|---|---|
| **Distribution shift** | $\mathcal{D}_{\text{cal}} \neq \mathcal{D}_{\text{deploy}}$ → T* suboptimal at deployment | Monitor ECE on deployment data, not just cal set |
| **Label noise** | Noisy labels lower ECE without improving truth fidelity | ECE decreases while decision reliability degrades |
| **Class imbalance** | Global ECE improves while minority ECE worsens | Class-conditional ECE required |
| **Wrong direction T** | T hardcoded or estimated on wrong regime | T* must be optimised, not assumed |
| **ECE ≠ NLL** | ECE can improve while NLL stays flat (underconfident logits) | Report both metrics; divergence is a signal |

---

## 7. Reliability as a Design Constraint

The memo's central claim is that calibration should be treated as a constraint, not a badge:

$$\text{model is reliability-valid} \iff \text{ECE}(f_\theta) \leq \varepsilon \;\text{ and }\; \Delta\text{ECE}_{\text{shift}} \leq \delta$$

This means:
1. Setting an explicit ECE threshold before deployment, not just reporting the number
2. Requiring class-conditional ECE for imbalanced problems, not just global ECE
3. Stress-testing T* stability across distribution shifts (connecting to the [assumption stress harness](assumption_stress_harness.py))
4. Treating ECE–NLL divergence as a discriminative quality flag, not a metric inconsistency

A model that reports ECE = 0.04 after calibration has not been verified reliable. It has passed one measurement at one distribution. Calibration without stability testing is a false sense of safety.

---

## 8. Connections to the Series

| Reliability question | Artifact |
|---|---|
| Does the interval contain the truth ≥90% of the time? | Split conformal prediction |
| Does coverage hold when assumptions break? | Assumption stress harness |
| Which training points drive predictions? | Influence & stability analysis |
| **Are predicted probabilities trustworthy for decisions?** | **This artifact** |

---

## 9. Reproducibility

```bash
pip install numpy scipy

# Defaults: n=5000, bins=10, seed=42, pi=0.15
python Calibration_decomposition.py

# Vary imbalance ratio
python Calibration_decomposition.py --pi 0.05 --n 10000
```

All outputs are deterministic given `--seed`. No plotting libraries required.

---

## 10. Takeaways

> **Lower ECE does not mean safer decisions. It means the confidence is better aligned with frequency — on the calibration set, at calibration time. These are not the same thing as decision reliability.**

Three shifts:

1. **Calibration is a direction problem before it is a magnitude problem.** T* = 0.193 and T* = 4.391 require opposite corrections. Applying the wrong direction (as a hardcoded T does) makes calibration worse. Regime identification — overconfident vs underconfident — is a prerequisite for applying any fix.

2. **Global ECE is insufficient for imbalanced problems.** The scenario where calibration matters most — high-stakes minority class prediction — is exactly the scenario where global ECE gives the most misleading signal. Class-conditional ECE is not a nice-to-have; it is the minimum measurement for imbalanced deployment.

3. **ECE and NLL measure different things; their divergence is information.** A 94% ECE reduction with a 16% NLL reduction is not a measurement inconsistency. It means the calibration fix aligned confidence to frequency but could not recover discriminative content the model never had. Both metrics belong in any calibration report.

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Kull, M., Nieto, M. P., Kängsepp, M., Flach, P., et al. (2019). Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration. *NeurIPS*.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
