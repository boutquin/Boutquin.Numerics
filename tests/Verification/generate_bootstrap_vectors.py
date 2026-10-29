"""Bootstrap resampler reference vectors (Phase 3).

Spec §2.3 coverage of six bootstrap resamplers plus the Politis-White
automatic block-length selector. Each estimator is verified against one
of the following Python references:

* ``stationary`` — ``arch.bootstrap.StationaryBootstrap`` 95% band for
  the resampled-mean distribution.
* ``moving_block`` — ``arch.bootstrap.MovingBlockBootstrap`` 95% band
  for the resampled-mean distribution.
* ``wild_mammen`` — hand-coded Mammen weight distribution: a weighted
  two-point distribution with atoms at ``−(√5−1)/2`` and ``(√5+1)/2``.
* ``wild_rademacher`` — ±1 Rademacher weights via scipy's equivalent.
* ``fast_double_bootstrap`` — Davidson-MacKinnon 2007 two-stage
  reference p-value, written in numpy from the FDB definition (right
  tail uses the ``(1 - p1)``-quantile of the inner distribution).
* ``subsampler`` — direct numpy port of the Politis-Romano-Wolf
  subsampling distribution; the C# ``Subsampler`` is deterministic so
  the comparison is bit-identical.
* ``politis_white`` — ``arch.bootstrap.optimal_block_length``, compared
  against the C# ``PolitisWhiteBlockLength`` estimate within ±2
  (per spec §2.3).

The spec's §6 Decision Rule applies: statistical-envelope assertions
(mean within 95% band, weight-distribution KS) are the intended shape
for these tests — bit-identical RNG matching is not attempted because
the Numerics Pcg64 and numpy's BitGenerator streams do not share an
initialisation convention.

Library pins:
* numpy==2.x
* scipy==1.13+
* arch==8.0+
"""

from __future__ import annotations

import numpy as np
from arch.bootstrap import (
    MovingBlockBootstrap,
    StationaryBootstrap,
    optimal_block_length,
)

from conftest import save_vector


SEED = 20260415


# =====================================================================
#  Synthetic series — stationary AR(1) with mild persistence so the
#  block bootstraps have meaningful serial correlation to preserve.
# =====================================================================


def _build_ar1_series(seed: int, n: int, rho: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    for i in range(1, n):
        x[i] += rho * x[i - 1]
    return x


# =====================================================================
#  Stationary bootstrap — arch.bootstrap.StationaryBootstrap.
# =====================================================================


def reference_stationary_bootstrap_band(
    series: np.ndarray, mean_block_length: float, trials: int, seed: int
) -> dict:
    """95% band for the mean-of-resample distribution under arch's
    Stationary bootstrap.

    Reference: Politis & Romano (1994), via arch.bootstrap.
    """

    rng = np.random.default_rng(seed)
    bs = StationaryBootstrap(mean_block_length, series, seed=rng)
    means = np.empty(trials)
    for b, (pos, _) in enumerate(bs.bootstrap(trials)):
        resample = pos[0]
        means[b] = float(resample.mean())

    return {
        "mean_block_length": mean_block_length,
        "trials": trials,
        "mean_of_means": float(means.mean()),
        "mean_std": float(means.std(ddof=1)),
        "band_lower_2_5": float(np.percentile(means, 2.5)),
        "band_upper_97_5": float(np.percentile(means, 97.5)),
        "sample_mean": float(series.mean()),
    }


# =====================================================================
#  Moving block bootstrap — arch.bootstrap.MovingBlockBootstrap.
# =====================================================================


def reference_moving_block_band(
    series: np.ndarray, block_size: int, trials: int, seed: int
) -> dict:
    """95% band for the mean-of-resample distribution under arch's MBB."""

    rng = np.random.default_rng(seed)
    bs = MovingBlockBootstrap(block_size, series, seed=rng)
    means = np.empty(trials)
    for b, (pos, _) in enumerate(bs.bootstrap(trials)):
        resample = pos[0]
        means[b] = float(resample.mean())

    return {
        "block_size": block_size,
        "trials": trials,
        "mean_of_means": float(means.mean()),
        "mean_std": float(means.std(ddof=1)),
        "band_lower_2_5": float(np.percentile(means, 2.5)),
        "band_upper_97_5": float(np.percentile(means, 97.5)),
        "sample_mean": float(series.mean()),
    }


# =====================================================================
#  Wild bootstrap — Mammen + Rademacher weight distributions.
# =====================================================================


def reference_wild_mammen_moments() -> dict:
    """First two moments of the Mammen two-point weight distribution.

    Atoms at ``a = −(√5−1)/2`` (prob ``p = (√5+1)/(2√5)``) and
    ``b = (√5+1)/2`` (prob ``1 − p``). Mean 0, variance 1, third moment 1
    — the defining properties Mammen (1993) used for moment-matching.
    """

    p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
    a = -(np.sqrt(5) - 1) / 2
    b = (np.sqrt(5) + 1) / 2
    mean = p * a + (1 - p) * b
    var = p * a * a + (1 - p) * b * b - mean * mean
    third = p * (a - mean) ** 3 + (1 - p) * (b - mean) ** 3
    return {
        "probability_of_negative": float(p),
        "negative_atom": float(a),
        "positive_atom": float(b),
        "expected_mean": float(mean),
        "expected_variance": float(var),
        "expected_third_central_moment": float(third),
    }


def reference_wild_rademacher_moments() -> dict:
    """First two moments of the Rademacher ±1 distribution."""

    return {
        "probability_of_negative": 0.5,
        "negative_atom": -1.0,
        "positive_atom": 1.0,
        "expected_mean": 0.0,
        "expected_variance": 1.0,
        "expected_third_central_moment": 0.0,
    }


# =====================================================================
#  Fast Double Bootstrap — Davidson-MacKinnon 2007 reference p-value.
# =====================================================================


def reference_fast_double_bootstrap(
    observations: np.ndarray,
    observed: float,
    outer_count: int,
    seed: int,
    tail: str = "right",
    null_mean: float = 0.0,
) -> float:
    """Fast-double-bootstrap p-value of the signed sample mean under
    H0: mean = ``null_mean``.

    Written from the FDB definition (Davidson & MacKinnon 2007, CSDA
    51(7), 3259-3281, as restated in Davidson 2023 §4), not ported from
    the C# engine:

    1. The bootstrap DGP must respect the null. For resampling a location
       statistic, the null-imposed DGP of a sample is its empirical law
       shifted to mean ``null_mean``.
    2. For b = 1..B draw an outer sample from the null-imposed DGP of the
       observations and compute ``tau_star[b]``; draw ONE inner sample from the
       null-imposed DGP of that outer sample and compute ``tau_2star[b]``.
    3. ``p1`` = proportion of ``tau_star`` strictly beyond ``observed`` in the chosen tail.
    4. Right-rejecting test: ``Q`` = the ``(1 - p1)``-quantile of
       ``tau_2star``; left-rejecting test: ``Q`` = the ``p1``-quantile.
    5. FDB p-value = proportion of ``tau_star`` strictly beyond ``Q`` in the chosen tail.

    Strict comparisons (``>`` / ``<``) exclude ties at both stages, following
    Davidson 2023 §4, algorithm steps 4/6 and its right-tail remark.
    Quantiles use ``np.quantile``'s default linear method.
    ``tail="two"`` folds the two one-sided FDB p-values:
    ``min(2 * min(p_right, p_left), 1)``.
    """

    if tail not in ("right", "left", "two"):
        raise ValueError(f"Unknown tail '{tail}'.")

    rng = np.random.default_rng(seed)
    n = observations.size

    def null_dgp(sample: np.ndarray) -> np.ndarray:
        return sample - sample.mean() + null_mean

    population = null_dgp(observations)
    tau_star = np.empty(outer_count)
    tau_2star = np.empty(outer_count)
    for b in range(outer_count):
        outer = population[rng.integers(0, n, size=n)]
        tau_star[b] = float(outer.mean())
        inner = null_dgp(outer)[rng.integers(0, n, size=n)]
        tau_2star[b] = float(inner.mean())

    def right_fdb() -> float:
        p1 = float(np.mean(tau_star > observed))
        q = float(np.quantile(tau_2star, 1.0 - p1))
        return float(np.mean(tau_star > q))

    def left_fdb() -> float:
        p1 = float(np.mean(tau_star < observed))
        q = float(np.quantile(tau_2star, p1))
        return float(np.mean(tau_star < q))

    if tail == "right":
        return right_fdb()
    if tail == "left":
        return left_fdb()
    return min(2.0 * min(right_fdb(), left_fdb()), 1.0)


# =====================================================================
#  Subsampler — direct numpy port (deterministic).
# =====================================================================


def reference_subsampler_mean(series: np.ndarray, subsample_length: int) -> np.ndarray:
    """Return the sorted-ascending distribution of subsample means over
    the T − b + 1 contiguous subsamples of length b.

    Reference: Politis, Romano & Wolf (1999), *Subsampling*.
    The C# ``Subsampler.Run`` is deterministic; the output is sorted
    and directly comparable bit-for-bit.
    """

    t = series.size
    assert subsample_length < t
    means = np.array(
        [float(series[i : i + subsample_length].mean()) for i in range(t - subsample_length + 1)]
    )
    means.sort()
    return means


# =====================================================================
#  Politis-White block length — arch.bootstrap.optimal_block_length.
# =====================================================================


def reference_politis_white_block_length(series: np.ndarray) -> dict:
    """Optimal block length via Politis-White 2004 (Patton 2009
    correction), computed through ``arch.bootstrap.optimal_block_length``.

    Returns both the Stationary and Circular block-length choices. The
    C# ``PolitisWhiteBlockLength.Estimate`` returns the Stationary
    variant (matching the ``arch`` default for the "stationary" row).
    """

    import pandas as pd

    df = optimal_block_length(pd.Series(series))
    # ``optimal_block_length`` returns a 1-row DataFrame indexed by the
    # input Series' name (None here); iloc-0 is the stable access path.
    return {
        "stationary_block_length": float(df["stationary"].iloc[0]),
        "circular_block_length": float(df["circular"].iloc[0]),
    }


def reference_politis_white_paper(series: np.ndarray) -> float:
    """Complete corrected author listing; field name retained for compatibility."""
    from generate_ppw2009_vectors import author_estimate
    return author_estimate(series)


# =====================================================================
#  Orchestrator.
# =====================================================================


def generate() -> None:
    # ==================================================================
    # Legacy block — keep the IID mean-band test passing for the previous
    # ship without churn.
    # ==================================================================
    rng = np.random.default_rng(seed=42)
    legacy_n = 252
    legacy_series = rng.normal(loc=0.0, scale=1.0, size=legacy_n)
    legacy_mean = float(legacy_series.mean())
    legacy_std = float(legacy_series.std(ddof=1))
    legacy_se = legacy_std / np.sqrt(legacy_n)

    # ==================================================================
    # Phase 3 additions — rich block-bootstrap + wild + FDB + Subsampler
    # + PolitisWhite references, each with the assertion contract the
    # spec's §2.3 calls for.
    # ==================================================================

    # Shared AR(1) series with ρ=0.3 for block-bootstrap tests.
    ar_n = 400
    ar_rho = 0.3
    ar_series = _build_ar1_series(seed=20260501, n=ar_n, rho=ar_rho)

    stationary = reference_stationary_bootstrap_band(
        ar_series, mean_block_length=8.0, trials=5000, seed=20260502
    )
    moving_block = reference_moving_block_band(
        ar_series, block_size=8, trials=5000, seed=20260503
    )

    wild_mammen = reference_wild_mammen_moments()
    wild_rademacher = reference_wild_rademacher_moments()

    # FDB — 100-obs series, 2000 outer replications (spec's suggested scale).
    # Statistic is the SIGNED sample mean so that left and two-sided p-values
    # are both non-trivial (not identically 1 or 0).
    fdb_series = _build_ar1_series(seed=20260504, n=100, rho=0.0)
    fdb_observed = float(fdb_series.mean())
    fdb_pvalue_right = reference_fast_double_bootstrap(
        fdb_series, observed=fdb_observed, outer_count=2000, seed=20260505, tail="right"
    )
    fdb_pvalue_left = reference_fast_double_bootstrap(
        fdb_series, observed=fdb_observed, outer_count=2000, seed=20260505, tail="left"
    )
    fdb_pvalue_two = reference_fast_double_bootstrap(
        fdb_series, observed=fdb_observed, outer_count=2000, seed=20260505, tail="two"
    )

    # FDB at an upper-tail observed statistic (sample mean + 2 standard errors).
    # At the centre the right- and left-tail quantiles coincide near the median,
    # so only an off-centre observation distinguishes the (1 - p1)-quantile from
    # the p1-quantile.
    fdb_observed_upper = float(
        fdb_series.mean() + 2.0 * fdb_series.std(ddof=1) / np.sqrt(fdb_series.size)
    )
    fdb_upper = {
        f"pvalue_{key}_upper": reference_fast_double_bootstrap(
            fdb_series, observed=fdb_observed_upper, outer_count=2000, seed=20260505, tail=tail
        )
        for key, tail in (("right_tail", "right"), ("left_tail", "left"), ("two_sided", "two"))
    }

    # Subsampler — deterministic; 200-obs series, b = ⌈T^(2/3)⌉ ≈ 35.
    sub_series = _build_ar1_series(seed=20260506, n=200, rho=0.2)
    sub_b = int(np.ceil(sub_series.size ** (2.0 / 3.0)))
    sub_distribution = reference_subsampler_mean(sub_series, subsample_length=sub_b)

    # Politis-White — 512-obs AR(1) with ρ=0.4 so the optimal block is
    # comfortably above the trivial value of 1.
    pw_series = _build_ar1_series(seed=20260507, n=512, rho=0.4)
    pw_result = reference_politis_white_block_length(pw_series)
    pw_result["paper_stationary_block_length"] = reference_politis_white_paper(pw_series)

    # Negatively autocorrelated companion: odd-lag autocovariances are negative,
    # so the signed G of PW 2004 eq. 8 differs from a sum of |rho(k)|.
    pw_negative_series = _build_ar1_series(seed=20260509, n=512, rho=-0.4)
    pw_negative_block = reference_politis_white_paper(pw_negative_series)

    save_vector(
        "bootstrap",
        {
            "library_pins": {
                "numpy": "2",
                "scipy": "1.13+",
                "arch": "8.0+",
            },
            # Legacy top-level keys (unchanged semantics).
            "series": legacy_series.tolist(),
            "sample_mean": legacy_mean,
            "sample_std": legacy_std,
            "mean_standard_error_iid": legacy_se,
            "mean_95_ci_half_width": 1.96 * legacy_se,
            "lag1_acf": float(np.corrcoef(legacy_series[:-1], legacy_series[1:])[0, 1]),
            "n": legacy_n,
            "trials": 1000,
            # Phase 3 additions.
            "ar1_series": {
                "values": ar_series.tolist(),
                "rho": ar_rho,
                "n": ar_n,
            },
            "stationary_bootstrap": stationary,
            "moving_block_bootstrap": moving_block,
            "wild_mammen": wild_mammen,
            "wild_rademacher": wild_rademacher,
            "fast_double_bootstrap": {
                "series": fdb_series.tolist(),
                "observed": fdb_observed,
                "outer_count": 2000,
                "seed": 20260505,
                "pvalue_right_tail": fdb_pvalue_right,
                "pvalue_left_tail": fdb_pvalue_left,
                "pvalue_two_sided": fdb_pvalue_two,
                "observed_upper": fdb_observed_upper,
                **fdb_upper,
            },
            "subsampler": {
                "series": sub_series.tolist(),
                "subsample_length": sub_b,
                "sorted_distribution": sub_distribution.tolist(),
            },
            "politis_white": {
                "series": pw_series.tolist(),
                **pw_result,
            },
            "politis_white_negative": {
                "series": pw_negative_series.tolist(),
                "paper_stationary_block_length": pw_negative_block,
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote bootstrap.json")
