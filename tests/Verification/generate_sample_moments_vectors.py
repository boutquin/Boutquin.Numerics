"""Sample skewness and excess kurtosis reference vectors.

Both statistics are verified against scipy.stats independent references:

* ``scipy.stats.skew(data, bias=False)`` — adjusted Fisher-Pearson skewness
* ``scipy.stats.kurtosis(data, fisher=True, bias=False)`` — Fisher G₂ excess kurtosis

scipy.stats uses a fully independent C/Cython codebase and different
computational layout from the Boutquin.Numerics two-pass implementation,
so shared algorithmic bugs are not possible.

NIST StRD univariate datasets certify mean and standard deviation only;
no NIST reference exists for skewness or kurtosis. Synthetic data is used
here; scipy.stats is the authoritative independent reference.

Library pins:

* numpy==2.x
* scipy==1.13+

Fixed seed; regenerating produces bit-identical JSON.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from conftest import save_vector


SEED = 20260415


def reference_skewness(data: np.ndarray) -> float:
    """Adjusted Fisher-Pearson skewness — scipy.stats independent reference."""
    return float(stats.skew(data, bias=False))


def reference_excess_kurtosis(data: np.ndarray) -> float:
    """Fisher excess kurtosis G₂ — scipy.stats independent reference."""
    return float(stats.kurtosis(data, fisher=True, bias=False))


def generate() -> None:
    rng = np.random.default_rng(SEED)
    cases = []

    # Case 1: Normal distribution — symmetric, near-zero skewness and kurtosis.
    normal_data = rng.normal(loc=0.0, scale=1.0, size=200)
    cases.append({
        "name": "normal_n200",
        "data": normal_data.tolist(),
        "skewness": reference_skewness(normal_data),
        "excess_kurtosis": reference_excess_kurtosis(normal_data),
    })

    # Case 2: Gamma(2,1) — positive skew, moderate leptokurtosis.
    gamma_data = rng.gamma(shape=2.0, scale=1.0, size=300)
    cases.append({
        "name": "gamma_positive_skew_n300",
        "data": gamma_data.tolist(),
        "skewness": reference_skewness(gamma_data),
        "excess_kurtosis": reference_excess_kurtosis(gamma_data),
    })

    # Case 3: Negated Gamma(2,1) — negative skew.
    neg_gamma_data = -rng.gamma(shape=2.0, scale=1.0, size=300)
    cases.append({
        "name": "neg_gamma_negative_skew_n300",
        "data": neg_gamma_data.tolist(),
        "skewness": reference_skewness(neg_gamma_data),
        "excess_kurtosis": reference_excess_kurtosis(neg_gamma_data),
    })

    # Case 4: Uniform(0,1) — symmetric, platykurtic (excess kurtosis near −1.2).
    uniform_data = rng.uniform(low=0.0, high=1.0, size=500)
    cases.append({
        "name": "uniform_n500",
        "data": uniform_data.tolist(),
        "skewness": reference_skewness(uniform_data),
        "excess_kurtosis": reference_excess_kurtosis(uniform_data),
    })

    # Case 5: t(3) — symmetric but heavy-tailed (high positive kurtosis).
    t3_data = rng.standard_t(df=3, size=400)
    cases.append({
        "name": "t3_heavy_tailed_n400",
        "data": t3_data.tolist(),
        "skewness": reference_skewness(t3_data),
        "excess_kurtosis": reference_excess_kurtosis(t3_data),
    })

    # Case 6: Small sample n=12 — exercises low-n correction factors.
    small_data = rng.normal(loc=5.0, scale=2.0, size=12)
    cases.append({
        "name": "small_n12",
        "data": small_data.tolist(),
        "skewness": reference_skewness(small_data),
        "excess_kurtosis": reference_excess_kurtosis(small_data),
    })

    save_vector(
        "sample_moments",
        {
            "seed": SEED,
            "library_pins": {"numpy": "2", "scipy": "1.13+"},
            "cases": cases,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote sample_moments.json")
