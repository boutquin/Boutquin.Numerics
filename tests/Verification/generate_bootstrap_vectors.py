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
  reference p-value, hand-ported into numpy.
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
        resample = pos[0][0]
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
        resample = pos[0][0]
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
) -> float:
    """Two-stage FDB p-value for the "absolute mean deviation" statistic.

    Reference: Davidson & MacKinnon (2007), "Improving the Reliability of
    Bootstrap Tests with the Fast Double Bootstrap", CSDA 51(7),
    3259–3281.

    Verbatim port of the C# ``FastDoubleBootstrap.PValue`` algorithm.
    The ``tail`` argument selects the C# ``PValueTail`` convention:
      * ``"right"`` — fraction of outer ≥ target (classical FDB).
      * ``"left"`` — fraction of outer ≤ target.
      * ``"two"`` — ``min(2·min(right, left), 1)`` (standard bilateral
        p-value under symmetry).

    Both the outer <c>p*</c> and the final rank apply the same tail
    convention — mixing tails silently inverts the p-value.

    Statistic: ``mean(sample)`` — the (signed) sample mean. Using the
    signed statistic exposes both tails of the bootstrap distribution,
    giving the test a non-trivial left-tail probability.
    """

    if tail not in ("right", "left", "two"):
        raise ValueError(f"Unknown tail '{tail}'.")

    rng = np.random.default_rng(seed)
    n = observations.size

    def statistic(sample: np.ndarray) -> float:
        return float(sample.mean())

    def tail_pvalue(sorted_dist: np.ndarray, target: float) -> float:
        right = float(np.sum(sorted_dist >= target)) / sorted_dist.size
        left = float(np.sum(sorted_dist <= target)) / sorted_dist.size
        if tail == "right":
            return right
        if tail == "left":
            return left
        # two-sided: min(2·min(right, left), 1), with the cap matching the C#.
        return min(2.0 * min(right, left), 1.0)

    outer_stats = np.empty(outer_count)
    inner_stats = np.empty(outer_count)
    for b in range(outer_count):
        outer_idx = rng.integers(0, n, size=n)
        outer = observations[outer_idx]
        outer_stats[b] = statistic(outer)

        inner_idx = rng.integers(0, n, size=n)
        inner = outer[inner_idx]
        inner_stats[b] = statistic(inner)

    sorted_outer = np.sort(outer_stats)
    sorted_inner = np.sort(inner_stats)

    # p* = rank of observed in outer under the chosen tail.
    p_star = tail_pvalue(sorted_outer, observed)

    # q* = linear-interp percentile at rank p* in sorted_inner — matches
    # the C# ``Percentile.Compute`` convention.
    if outer_count <= 1:
        q_star = float(sorted_inner[0])
    else:
        rank = p_star * (outer_count - 1)
        lo = int(np.floor(rank))
        hi = int(np.ceil(rank))
        if lo == hi:
            q_star = float(sorted_inner[lo])
        else:
            frac = rank - lo
            q_star = float(
                sorted_inner[lo] + frac * (sorted_inner[hi] - sorted_inner[lo])
            )

    return tail_pvalue(sorted_outer, q_star)


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

    # Subsampler — deterministic; 200-obs series, b = ⌈T^(2/3)⌉ ≈ 35.
    sub_series = _build_ar1_series(seed=20260506, n=200, rho=0.2)
    sub_b = int(np.ceil(sub_series.size ** (2.0 / 3.0)))
    sub_distribution = reference_subsampler_mean(sub_series, subsample_length=sub_b)

    # Politis-White — 512-obs AR(1) with ρ=0.4 so the optimal block is
    # comfortably above the trivial value of 1.
    pw_series = _build_ar1_series(seed=20260507, n=512, rho=0.4)
    pw_result = reference_politis_white_block_length(pw_series)

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
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote bootstrap.json")
