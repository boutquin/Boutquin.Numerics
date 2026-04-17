"""DSR / MinTRL / HLZ haircut / PBO reference values.

All implementations mirror the formulae in the original papers so that the
C# side can target the same arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from conftest import save_vector


def sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    # Match C# DeflatedSharpeRatio.ComputeFromReturns which uses the biased
    # (ddof=0) standard deviation — the moments are divided by N, not N-1.
    mean = float(returns.mean())
    std = float(returns.std(ddof=0))
    return 0.0 if std == 0.0 else mean / std * math.sqrt(trading_days)


def probabilistic_sharpe(sharpe_hat: float, sharpe_bench: float, n: int,
                         skew: float, kurt: float) -> float:
    """Bailey-Lopez de Prado 2012 probabilistic Sharpe ratio."""

    denom = 1.0 - skew * sharpe_hat + 0.25 * (kurt - 1.0) * sharpe_hat * sharpe_hat
    z = (sharpe_hat - sharpe_bench) * math.sqrt(max(n - 1, 1)) / math.sqrt(max(denom, 1e-12))
    return float(stats.norm.cdf(z))


def deflated_sharpe(sharpe_hat: float, num_trials: int, n: int,
                    skew: float, kurt: float) -> float:
    """Bailey-Lopez de Prado 2014 deflated Sharpe ratio."""

    euler_mascheroni = 0.5772156649
    expected_max = math.sqrt(2.0 * math.log(num_trials)) * (
        1.0 - euler_mascheroni / math.sqrt(2.0 * math.log(num_trials))
    )
    return probabilistic_sharpe(sharpe_hat, expected_max, n, skew, kurt)


def minimum_track_record_length(sharpe_hat: float, sharpe_bench: float,
                                skew: float, kurt: float,
                                confidence: float = 0.95) -> float:
    """Bailey-Lopez de Prado 2012 MinTRL."""

    if sharpe_hat <= sharpe_bench:
        return math.inf
    z = stats.norm.ppf(confidence)
    denom = (sharpe_hat - sharpe_bench) ** 2
    if denom == 0.0:
        return math.inf
    return 1.0 + (1.0 - skew * sharpe_hat + 0.25 * (kurt - 1.0) * sharpe_hat * sharpe_hat) * (z * z) / denom


def generate() -> None:
    rng = np.random.default_rng(seed=2026)
    n = 252
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)

    sh = sharpe(returns)
    skew = float(stats.skew(returns, bias=True))
    # pandas/numpy kurtosis convention: excess. Use `fisher=False` to get
    # the Pearson ("raw") kurtosis, matching the formulas in the papers.
    kurt = float(stats.kurtosis(returns, fisher=False, bias=True))

    psr = probabilistic_sharpe(sh, 0.0, n, skew, kurt)
    dsr_statistic = deflated_sharpe(sh, num_trials=100, n=n, skew=skew, kurt=kurt)
    min_trl = minimum_track_record_length(sh, 0.0, skew, kurt, confidence=0.95)

    # Raw deflated-Sharpe scalar in C# convention: sr − E[max SR].
    euler_mascheroni = 0.5772156649
    log_n = math.log(100)
    expected_max = math.sqrt(2.0 * log_n) * (1 - euler_mascheroni / log_n) + euler_mascheroni / math.sqrt(2.0 * log_n)
    deflated_scalar = sh - expected_max

    save_vector(
        "dsr",
        {
            "returns": returns.tolist(),
            "sharpe": sh,
            "skew": skew,
            "kurtosis_pearson": kurt,
            "probabilistic_sharpe": psr,
            "deflated_sharpe_statistic_100_trials": dsr_statistic,
            "deflated_sharpe_scalar_100_trials": deflated_scalar,
            "expected_max_sharpe_100_trials": expected_max,
            "min_trl_sr0_conf95": min_trl,
            "num_trials": 100,
            "trading_days_per_year": 252,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote dsr.json")
