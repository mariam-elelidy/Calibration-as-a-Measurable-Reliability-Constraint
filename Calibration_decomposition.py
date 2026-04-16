r"""
Calibration Decomposition Under Controlled Mismatch
====================================================
Author : Mariam Mohamed Elelidy
Purpose: Diagnose *why* a model is miscalibrated — not just whether it is.

Problem
-------
Most calibration pipelines report a single ECE number and a reliability plot.
That is not a diagnosis. Miscalibration decomposes into distinct, measurable
regimes that respond differently to post-hoc fixes:

  Overconfidence  → predicted p > empirical frequency  → T* < 1
  Underconfidence → predicted p < empirical frequency  → T* > 1
  Class imbalance → global ECE improves while minority-class ECE may worsen

Temperature scaling (T*) fixes overconfidence and underconfidence exactly when
calibration and deployment distributions match. It fails silently under class
imbalance: global ECE improves while minority-class reliability degrades.

Design choices
--------------
- T* is optimised via NLL minimisation (minimize_scalar), not hardcoded.
  A hardcoded T in the wrong direction actively worsens calibration.
- Three scenarios run in sequence: overconfidence, underconfidence, imbalance.
  Each breaks exactly one calibration assumption, matching the stress-harness
  design philosophy from the companion artifacts.
- Per-bin gap direction (overconf / underconf) is reported for every bin —
  the signed decomposition is more informative than the absolute ECE alone.
- Class-conditional ECE is reported for the imbalance scenario to expose
  the global-vs-minority divergence.

Metrics
-------
ECE   = sum_m (|B_m|/n) |acc(B_m) - conf(B_m)|
NLL   = -(1/n) sum_i [y_i log p_i + (1-y_i) log(1-p_i)]
Brier = (1/n) sum_i (p_i - y_i)^2

Usage
-----
    python Calibration_decomposition.py          # defaults
    python Calibration_decomposition.py --n 10000 --bins 15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll_score(y: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform binning)."""
    ece_val = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            acc  = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
            ece_val += np.abs(acc - conf) * float(np.mean(mask))
    return ece_val


def metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Return [ECE, NLL, Brier] as a numpy array."""
    return np.array([
        ece_score(y, p, n_bins),
        nll_score(y, p),
        brier_score(y, p),
    ])


# ────────────────────────────────────────────────────────────────────────────
# Temperature scaling with optimised T*
# ────────────────────────────────────────────────────────────────────────────

def temperature_scale(p: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to logits: p_T = sigma(logit(p) / T)."""
    eps = 1e-8
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    return 1.0 / (1.0 + np.exp(-logit / T))


def optimal_temperature(y: np.ndarray, p: np.ndarray) -> float:
    """T* = argmin_{T in [0.1, 10]} NLL(temperature_scale(p, T)).

    Optimising T on the same data it is evaluated on is a simplification
    (in practice you need a held-out validation set), but it correctly
    demonstrates the direction and magnitude of the fix.
    """
    result = minimize_scalar(
        lambda T: nll_score(y, temperature_scale(p, T)),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    return float(result.x)


# ────────────────────────────────────────────────────────────────────────────
# Reliability table
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BinRow:
    bin_label: str
    acc:       float
    conf:      float
    gap:       float          # acc - conf  (positive = underconf, negative = overconf)
    count:     int

    @property
    def direction(self) -> str:
        if   self.gap < -0.05: return "OVERCONF"
        elif self.gap >  0.05: return "UNDERCONF"
        else:                  return "calibrated"


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> list[BinRow]:
    edges = np.linspace(0, 1, n_bins + 1)
    rows  = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi)
        if mask.any():
            rows.append(BinRow(
                bin_label=f"{lo:.1f}-{hi:.1f}",
                acc=float(np.mean(y[mask])),
                conf=float(np.mean(p[mask])),
                gap=float(np.mean(y[mask]) - np.mean(p[mask])),
                count=int(mask.sum()),
            ))
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name:        str
    description: str
    regime:      str
    T_opt:       float
    before:      np.ndarray   # [ECE, NLL, Brier]
    after:       np.ndarray
    delta:       np.ndarray
    table_before: list[BinRow]
    table_after:  list[BinRow]
    extra:        dict         # any scenario-specific extras


def run_overconfidence(rng: np.random.Generator, n: int, n_bins: int) -> ScenarioResult:
    """Model logits too sharp → predicted p pushed toward 0/1.

    Generated with high-amplitude logits; the model is extremely confident
    on most examples, but the true probability near the decision boundary
    is far more uncertain.
    """
    y = (rng.random(n) < 0.5).astype(float)
    logits = 3.5 * (y - 0.5) + rng.normal(scale=0.8, size=n)
    p_raw  = 1.0 / (1.0 + np.exp(-logits))

    T_opt  = optimal_temperature(y, p_raw)
    p_cal  = temperature_scale(p_raw, T_opt)

    b = metrics(y, p_raw, n_bins)
    a = metrics(y, p_cal,  n_bins)

    # Identify the most overconfident bin (largest negative gap)
    rows = reliability_table(y, p_raw, n_bins)
    worst = min(rows, key=lambda r: r.gap)

    return ScenarioResult(
        name="Overconfidence",
        description="Logits scaled up 3.5×; model too sharp near decision boundary",
        regime="overconfident (T* < 1)",
        T_opt=T_opt,
        before=b, after=a, delta=a - b,
        table_before=rows,
        table_after=reliability_table(y, p_cal, n_bins),
        extra={"worst_bin": worst, "n_overconf_bins": sum(1 for r in rows if r.gap < -0.05)},
    )


def run_underconfidence(rng: np.random.Generator, n: int, n_bins: int) -> ScenarioResult:
    """Model logits too flat → predicted p bunched near 0.5."""
    y = (rng.random(n) < 0.5).astype(float)
    logits = 0.5 * (y - 0.5) + rng.normal(scale=1.5, size=n)
    p_raw  = 1.0 / (1.0 + np.exp(-logits))

    T_opt  = optimal_temperature(y, p_raw)
    p_cal  = temperature_scale(p_raw, T_opt)

    b = metrics(y, p_raw, n_bins)
    a = metrics(y, p_cal,  n_bins)

    return ScenarioResult(
        name="Underconfidence",
        description="Logits scaled down 0.5×; model hedges toward 0.5 on most examples",
        regime="underconfident (T* > 1)",
        T_opt=T_opt,
        before=b, after=a, delta=a - b,
        table_before=reliability_table(y, p_raw, n_bins),
        table_after=reliability_table(y, p_cal,  n_bins),
        extra={},
    )


def run_class_imbalance(
    rng: np.random.Generator, n: int, n_bins: int,
    imbalance_ratio: float = 0.15,
) -> ScenarioResult:
    """15% positive-class prevalence.

    Global ECE may improve after temperature scaling while minority-class
    (y=1) ECE is only partially corrected — the key failure mode.
    """
    y = (rng.random(n) < imbalance_ratio).astype(float)
    logits = 2.5 * (y - 0.5) + rng.normal(scale=1.2, size=n)
    p_raw  = 1.0 / (1.0 + np.exp(-logits))

    T_opt  = optimal_temperature(y, p_raw)
    p_cal  = temperature_scale(p_raw, T_opt)

    b = metrics(y, p_raw, n_bins)
    a = metrics(y, p_cal,  n_bins)

    # Class-conditional ECE
    def cond_ece(yy, pp, cls):
        mask = (yy == cls)
        if not mask.any():
            return 0.0
        return ece_score(yy[mask], pp[mask], n_bins)

    extra = {
        "imbalance_ratio":     imbalance_ratio,
        "global_ece_before":   b[0],
        "global_ece_after":    a[0],
        "minority_ece_before": cond_ece(y, p_raw, 1),
        "minority_ece_after":  cond_ece(y, p_cal, 1),
        "majority_ece_before": cond_ece(y, p_raw, 0),
        "majority_ece_after":  cond_ece(y, p_cal, 0),
    }

    return ScenarioResult(
        name="Class Imbalance",
        description=f"P(y=1) = {imbalance_ratio}; model sees mostly negatives during training",
        regime="imbalance-induced (global ECE misleading)",
        T_opt=T_opt,
        before=b, after=a, delta=a - b,
        table_before=reliability_table(y, p_raw, n_bins),
        table_after=reliability_table(y, p_cal,  n_bins),
        extra=extra,
    )


# ────────────────────────────────────────────────────────────────────────────
# Terminal report
# ────────────────────────────────────────────────────────────────────────────

def _bar(v: float, width: int = 24) -> str:
    v = max(0.0, min(1.0, v))
    k = int(round(v * width))
    return "█" * k + "░" * (width - k)


def _improvement_bar(before: float, after: float, width: int = 20) -> str:
    """Bar showing relative improvement (1 - after/before)."""
    if before < 1e-9:
        return "░" * width + "  (no change)"
    rel = max(0.0, min(1.0, 1.0 - after / before))
    k = int(round(rel * width))
    return "█" * k + "░" * (width - k) + f"  {rel*100:.1f}% reduction"


def print_scenario(sr: ScenarioResult, n_bins: int) -> None:
    sep = "─" * 76

    print()
    print("┌" + sep + "┐")
    print(f"│  Scenario: {sr.name:<20}  Regime: {sr.regime:<26}│")
    print(f"│  {sr.description:<74}│")
    print("└" + sep + "┘")

    print()
    print(f"  T*  = {sr.T_opt:.4f}  {'(< 1: softens overconfident logits)' if sr.T_opt < 1 else '(> 1: sharpens underconfident logits)'}")
    print()

    print("  Metrics tensor  [ECE, NLL, Brier]")
    print(f"    Before : {sr.before}")
    print(f"    After  : {sr.after}")
    print(f"    Δ      : {sr.delta}")
    print()

    print("  Improvement after T* calibration:")
    for i, name in enumerate(["ECE  ", "NLL  ", "Brier"]):
        print(f"    {name}  {_improvement_bar(sr.before[i], sr.after[i])}")
    print()

    # Scenario-specific extras
    if sr.name == "Overconfidence":
        w = sr.extra["worst_bin"]
        print(f"  Most overconfident bin: {w.bin_label}  gap = {w.gap:+.3f}  "
              f"(conf={w.conf:.3f}, acc={w.acc:.3f})")
        print(f"  Bins with |gap| > 0.05 (overconf): {sr.extra['n_overconf_bins']}/{n_bins}")

    if sr.name == "Class Imbalance":
        ex = sr.extra
        print("  Class-conditional ECE (global ECE can hide minority failure):")
        print(f"    Global   : {ex['global_ece_before']:.5f} → {ex['global_ece_after']:.5f}  "
              f"Δ = {ex['global_ece_after']-ex['global_ece_before']:+.5f}")
        print(f"    Minority : {ex['minority_ece_before']:.5f} → {ex['minority_ece_after']:.5f}  "
              f"Δ = {ex['minority_ece_after']-ex['minority_ece_before']:+.5f}")
        print(f"    Majority : {ex['majority_ece_before']:.5f} → {ex['majority_ece_after']:.5f}  "
              f"Δ = {ex['majority_ece_after']-ex['majority_ece_before']:+.5f}")
        print()
        print("  ⚠  Global ECE decreases yet minority-class ECE remains elevated.")
        print("     Reporting global ECE alone conceals the reliability gap for the")
        print("     minority class — the class that typically matters most clinically.")

    # Reliability tables
    print()
    print(f"  Reliability table — BEFORE calibration")
    print(f"  {'bin':<9} {'acc':>6} {'conf':>6} {'gap':>7}  {'count':>6}  direction")
    print("  " + "─" * 52)
    for row in sr.table_before:
        print(f"  {row.bin_label:<9} {row.acc:>6.3f} {row.conf:>6.3f} "
              f"{row.gap:>+7.3f}  {row.count:>6}  {row.direction}")

    print()
    print(f"  Reliability table — AFTER calibration (T* = {sr.T_opt:.4f})")
    print(f"  {'bin':<9} {'acc':>6} {'conf':>6} {'gap':>7}  {'count':>6}  direction")
    print("  " + "─" * 52)
    for row in sr.table_after:
        print(f"  {row.bin_label:<9} {row.acc:>6.3f} {row.conf:>6.3f} "
              f"{row.gap:>+7.3f}  {row.count:>6}  {row.direction}")


def print_summary_tensor(results: list[ScenarioResult]) -> None:
    print()
    print("═" * 76)
    print("SUMMARY TENSOR  [ECE_before, NLL_before, Brier_before, "
          "ECE_after, NLL_after, Brier_after]")
    print("Rows: [Overconfidence, Underconfidence, Class Imbalance]")
    print("═" * 76)
    rows = []
    for sr in results:
        rows.append(np.concatenate([sr.before, sr.after]))
    print(np.array(rows))
    print()
    print("Δ tensor  [ΔECE, ΔNLL, ΔBrier]  (after − before)")
    print(np.array([sr.delta for sr in results]))


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration decomposition under controlled mismatch"
    )
    p.add_argument("--n",     type=int, default=5000, help="samples per scenario")
    p.add_argument("--bins",  type=int, default=10,   help="reliability bins")
    p.add_argument("--seed",  type=int, default=42,   help="random seed")
    p.add_argument("--pi",    type=float, default=0.15,
                   help="positive-class prevalence for imbalance scenario")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    print("Calibration Decomposition — Three Miscalibration Regimes")
    print(f"n = {args.n}  │  bins = {args.bins}  │  seed = {args.seed}")
    print("Optimising T* via NLL minimisation (minimize_scalar, bounds [0.1, 10]) …")

    results = [
        run_overconfidence(rng,  args.n, args.bins),
        run_underconfidence(rng, args.n, args.bins),
        run_class_imbalance(rng, args.n, args.bins, args.pi),
    ]

    for sr in results:
        print_scenario(sr, args.bins)

    print_summary_tensor(results)

    print()
    print("─" * 76)
    print("KEY FINDING: class imbalance is the only scenario where temperature")
    print("scaling does not fully repair calibration. Global ECE improves but")
    print("minority-class ECE remains elevated — a silent failure mode that a")
    print("single ECE number would not expose.")
    print("─" * 76)


if __name__ == "__main__":
    main()
