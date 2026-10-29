"""DSR / MinTRL reference values, written from the papers.

Every function below transcribes an equation of Bailey & López de Prado:

* Probabilistic / Deflated Sharpe Ratio — "The Deflated Sharpe Ratio" (2014),
  eq. 1 (the rejection threshold SR̂₀) and eq. 2 (DSR, a PSR at SR̂₀).
* Minimum Track Record Length — "The Sharpe Ratio Efficient Frontier" (2012),
  eq. 13.

Both papers take the Sharpe ratio and the sample length T at the same,
non-annualized sampling frequency, and use raw (Pearson) kurtosis. The
annualized values the C# API accepts are converted here with √(periods/year)
before any equation is applied. The paper's own worked examples are stored
alongside the synthetic series as independent oracles.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from conftest import save_vector

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe(sr: float, sr_threshold: float, t: float,
                         skew: float, kurt: float) -> float:
    """PSR: Z[(SR̂ − SR*)·√(T−1) / √(1 − γ̂₃·SR̂ + (γ̂₄−1)/4·SR̂²)], per-period SR."""

    non_normality = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    z = (sr - sr_threshold) * math.sqrt(t - 1.0) / math.sqrt(non_normality)
    return float(stats.norm.cdf(z))


def dsr_threshold(trial_variance: float, num_trials: int) -> float:
    """Eq. 1: SR̂₀ = √V[{SR̂ₙ}]·((1−γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − e⁻¹/N])."""

    if num_trials == 1:
        return 0.0
    max_z = ((1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / num_trials)
             + EULER_MASCHERONI * stats.norm.ppf(1.0 - math.exp(-1.0) / num_trials))
    return math.sqrt(trial_variance) * float(max_z)


def minimum_track_record_length(sr: float, sr_bench: float, skew: float,
                                kurt: float, alpha: float) -> float:
    """Eq. 13, per-period SR and SR*; result in observations."""

    if sr <= sr_bench:
        return math.inf
    z = float(stats.norm.ppf(1.0 - alpha))
    non_normality = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return 1.0 + non_normality * (z / (sr - sr_bench)) ** 2


def generate() -> None:
    rng = np.random.default_rng(seed=2026)
    n = 252
    periods_per_year = 252
    num_trials = 100
    returns = rng.normal(loc=0.0012, scale=0.01, size=n)

    # Per-period Sharpe from N-divisor moments (the population moments the C#
    # ComputeFromReturns documents).
    sr = float(returns.mean()) / float(returns.std(ddof=0))
    sr_annual = sr * math.sqrt(periods_per_year)
    skew = float(stats.skew(returns, bias=True))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=True))

    # Without a disclosed trials' variance, V[{SR̂ₙ}] takes its null value: the
    # sampling variance 1/(T−1) of a per-period Sharpe estimate with no skill.
    null_threshold = dsr_threshold(1.0 / (n - 1.0), num_trials)
    dsr_null = probabilistic_sharpe(sr, null_threshold, n, skew, kurt)

    # A disclosed trials' variance of 0.5 in annualized units (the paper's value).
    trial_variance_annual = 0.5
    disclosed_threshold = dsr_threshold(trial_variance_annual / periods_per_year, num_trials)
    dsr_disclosed = probabilistic_sharpe(sr, disclosed_threshold, n, skew, kurt)

    # BLdP (2014) p. 10 worked example: annualized SR 2.5, five years of daily
    # data at 250 observations per year, N = 100, V = 1/2, skew −3, kurt 10.
    paper_ppy = 250
    paper_sr = 2.5 / math.sqrt(paper_ppy)
    paper_threshold = dsr_threshold(0.5 / paper_ppy, 100)
    paper_dsr = probabilistic_sharpe(paper_sr, paper_threshold, 5 * paper_ppy, -3.0, 10.0)

    # BLdP (2012) worked example: annualized SR 2 vs benchmark 1, daily (252),
    # Gaussian, 95% — the paper prints 688.2 observations (2.73 years).
    paper_min_trl = minimum_track_record_length(
        2.0 / math.sqrt(252), 1.0 / math.sqrt(252), 0.0, 3.0, 0.05)

    save_vector(
        "dsr",
        {
            "returns": returns.tolist(),
            "sharpe": sr_annual,
            "sharpe_per_period": sr,
            "skew": skew,
            "kurtosis_pearson": kurt,
            "probabilistic_sharpe": probabilistic_sharpe(sr, 0.0, n, skew, kurt),
            "deflated_sharpe_statistic_100_trials": 1.0 - dsr_null,
            "deflated_sharpe_scalar_100_trials": sr_annual - null_threshold * math.sqrt(periods_per_year),
            "expected_max_sharpe_100_trials": null_threshold * math.sqrt(periods_per_year),
            "trial_sharpe_variance_annualized": trial_variance_annual,
            "deflated_sharpe_statistic_100_trials_disclosed_variance": 1.0 - dsr_disclosed,
            "expected_max_sharpe_100_trials_disclosed_variance": disclosed_threshold * math.sqrt(periods_per_year),
            "min_trl_sr0_conf95": minimum_track_record_length(sr, 0.0, skew, kurt, 0.05),
            "num_trials": num_trials,
            "trading_days_per_year": periods_per_year,
            "paper_dsr_threshold_per_period": paper_threshold,
            "paper_dsr": paper_dsr,
            "paper_min_trl_observations": paper_min_trl,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote dsr.json")
