"""Golden vectors for HaircutSharpe (Harvey-Liu 2015) and FisherZTransform.

HaircutSharpe — derived from the paper, not from the C# source
---------------------------------------------------------------
Harvey, C. R. & Liu, Y. (2015), "Backtesting", Journal of Portfolio
Management 42(1). Every step below is a paper equation evaluated with
scipy.stats.t; nothing here mirrors the library's arithmetic.

  Eq. 1-2   SR = mu / sigma = t-ratio / sqrt(T), with SR and T at the SAME
            sampling frequency. The caller supplies an annualized SR, the
            backtest length in years and the observations per year, so
              T          = years * periods_per_year
              SR_period  = SR_annual / sqrt(periods_per_year)
              t-ratio    = SR_period * sqrt(T)
  Eq. 3     p_S = Pr(|r| > t-ratio), r ~ Student t with T - 1 degrees of
            freedom (two-sided).
  Bonf.     p_M = min(1, M * p_S)                        (p. 16)
  Holm      step-down over the ordered p-values; the tested strategy is the
            best of the M trials and the other M - 1 carry no evidence
            (p = 1), so it is rank 1.
  BHY       step-up recursion p_(M) = p_(M),
            p_(i) = min[p_(i+1), M c(M) / i * p_(i)], c(M) = sum_{j<=M} 1/j
            (p. 16), evaluated over the same ordered vector.
  Eq. 5     p_M = Pr(|r| > HSR_period * sqrt(T)) solved for the haircut
            Sharpe from the Bonferroni p_M (Exhibit 1 reports HSR as the
            Bonferroni-adjusted Sharpe), then re-annualized. The sign of the
            observed Sharpe is preserved (the test is two-sided).

The paper's worked example (T = 240 months, annual SR 0.75, N = 200; prints
p_S = 0.0008, p_M = 0.15, HSR = 0.32) is emitted as case_g.

FisherZTransform
----------------
  Forward(r) = 0.5 * ln((1+r) / (1-r))            (= atanh(r))
  Inverse(z) = tanh(z)
  CI(r, n, cl):
    z        = Forward(r)
    se       = 1 / sqrt(n - 3)
    tail     = (1 - cl) / 2
    critical = Phi^{-1}(1 - tail)
    lower    = tanh(z - critical * se)
    upper    = tanh(z + critical * se)

Library pins:
  numpy>=2  scipy>=1.13
Fixed seed — regenerating produces bit-identical JSON.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from conftest import save_vector


# =========================================================================
#  HaircutSharpe reference (Harvey-Liu 2015)
# =========================================================================


def _ppf(p: float) -> float:
    return float(stats.norm.ppf(p))


def _holm(p_sorted: list[float]) -> list[float]:
    """Holm step-down adjusted p-values for an ascending p-value vector."""
    m = len(p_sorted)
    adjusted, running = [], 0.0
    for i, p in enumerate(p_sorted, start=1):
        running = max(running, min(1.0, (m - i + 1) * p))
        adjusted.append(running)
    return adjusted


def _bhy(p_sorted: list[float]) -> list[float]:
    """BHY step-up adjusted p-values for an ascending p-value vector (HL p. 16)."""
    m = len(p_sorted)
    c_m = sum(1.0 / j for j in range(1, m + 1))
    adjusted = [0.0] * m
    adjusted[m - 1] = p_sorted[m - 1]
    for i in range(m - 1, 0, -1):  # i is the 1-based rank of p_sorted[i - 1]
        adjusted[i - 1] = min(adjusted[i], m * c_m / i * p_sorted[i - 1])
    return [min(1.0, a) for a in adjusted]


def haircut_sharpe(
    observed_sharpe: float,
    num_trials: int,
    backtest_years: float,
    trading_days: int = 252,
) -> dict:
    """Harvey-Liu (2015) haircut from the paper's equations 1-5."""

    n_obs = backtest_years * trading_days
    dof = n_obs - 1.0
    sr_period = observed_sharpe / math.sqrt(trading_days)
    t_ratio = sr_period * math.sqrt(n_obs)
    p_single = float(2.0 * stats.t.sf(abs(t_ratio), dof))

    ordered = [p_single] + [1.0] * (num_trials - 1)
    p_bonferroni = min(1.0, num_trials * p_single)
    p_holm = _holm(ordered)[0]
    p_bhy = _bhy(ordered)[0]

    if p_bonferroni <= 0.0:
        haircut_sr = observed_sharpe
    else:
        t_haircut = float(stats.t.isf(p_bonferroni / 2.0, dof))
        haircut_sr = math.copysign(t_haircut / math.sqrt(n_obs) * math.sqrt(trading_days), observed_sharpe)
        if abs(haircut_sr) > abs(observed_sharpe):
            haircut_sr = observed_sharpe
    haircut_amount = observed_sharpe - haircut_sr

    return {
        "observed_sharpe": observed_sharpe,
        "num_trials": num_trials,
        "backtest_years": backtest_years,
        "trading_days_per_year": trading_days,
        "t_ratio": t_ratio,
        "p_single": p_single,
        "p_bonferroni": p_bonferroni,
        "p_holm": p_holm,
        "p_bhy": p_bhy,
        "haircut_sharpe": haircut_sr,
        "haircut_amount": haircut_amount,
    }


# =========================================================================
#  FisherZTransform reference
# =========================================================================


def fisher_forward(r: float) -> float:
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def fisher_inverse(z: float) -> float:
    return math.tanh(z)


def fisher_ci(r: float, n: int, confidence_level: float) -> dict:
    z = fisher_forward(r)
    se = 1.0 / math.sqrt(n - 3)
    tail = (1.0 - confidence_level) / 2.0
    critical = _ppf(1.0 - tail)
    lo = fisher_inverse(z - critical * se)
    hi = fisher_inverse(z + critical * se)
    return {
        "r": r,
        "n": n,
        "confidence_level": confidence_level,
        "z": z,
        "se": se,
        "critical": critical,
        "lower": lo,
        "upper": hi,
    }


# =========================================================================
#  Orchestrator
# =========================================================================


def generate() -> None:
    # ---- HaircutSharpe cases ----------------------------------------
    # Case A: single trial — p_M = p_S, so the haircut Sharpe equals the observed one.
    case_a = haircut_sharpe(observed_sharpe=1.2, num_trials=1, backtest_years=5.0)

    # Case B: moderate trials, moderate Sharpe, 5 years daily — BHY not saturated.
    case_b = haircut_sharpe(observed_sharpe=1.0, num_trials=10, backtest_years=5.0)

    # Case C: many trials, 3 years daily.
    case_c = haircut_sharpe(observed_sharpe=1.5, num_trials=100, backtest_years=3.0)

    # Case D: many trials, short backtest, low Sharpe — p_M saturates at 1, HSR = 0.
    case_d = haircut_sharpe(observed_sharpe=0.5, num_trials=1000, backtest_years=2.0)

    # Case E: negative observed Sharpe — the two-sided test is symmetric in the
    # sign, so the haircut shrinks the magnitude toward zero.
    case_e = haircut_sharpe(observed_sharpe=-0.3, num_trials=50, backtest_years=3.0)

    # Case F: very high Sharpe, many trials — small p_S, modest haircut.
    case_f = haircut_sharpe(observed_sharpe=3.0, num_trials=500, backtest_years=10.0)

    # Case G: Harvey-Liu (2015) worked example — 20 years of monthly returns
    # (T = 240), annual Sharpe 0.75, 200 trials.
    case_g = haircut_sharpe(observed_sharpe=0.75, num_trials=200, backtest_years=20.0, trading_days=12)

    # ---- FisherZTransform scalar cases ------------------------------
    # Forward / Inverse at a range of r values (validated against np.arctanh).
    r_values = [-0.9, -0.5, 0.0, 0.3, 0.7, 0.95]
    forward_cases = [
        {"r": r, "z": fisher_forward(r), "r_roundtrip": fisher_inverse(fisher_forward(r))}
        for r in r_values
    ]

    # z → r roundtrip at a range of z values.
    z_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    inverse_cases = [
        {"z": z, "r": fisher_inverse(z), "z_roundtrip": fisher_forward(fisher_inverse(z))}
        for z in z_values
    ]

    # ---- FisherZTransform CI cases ----------------------------------
    ci_cases = [
        fisher_ci(r=0.5, n=50, confidence_level=0.95),
        fisher_ci(r=0.5, n=50, confidence_level=0.99),
        fisher_ci(r=0.8, n=30, confidence_level=0.95),
        fisher_ci(r=-0.4, n=100, confidence_level=0.95),
        fisher_ci(r=0.0, n=20, confidence_level=0.95),    # r=0 → symmetric CI
        fisher_ci(r=0.9, n=200, confidence_level=0.99),   # near-boundary, large n
    ]

    save_vector(
        "haircut_fisher",
        {
            "library_pins": {"numpy": "2", "scipy": "1.13+"},
            "haircut_sharpe": {
                "case_a_single_trial": case_a,
                "case_b_moderate_trials": case_b,
                "case_c_many_trials": case_c,
                "case_d_low_sharpe_many_trials": case_d,
                "case_e_negative_sharpe": case_e,
                "case_f_high_sharpe": case_f,
                "case_g_harvey_liu_example": case_g,
            },
            "fisher_z": {
                "forward_cases": forward_cases,
                "inverse_cases": inverse_cases,
                "ci_cases": ci_cases,
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote haircut_fisher.json")
