"""Golden vectors for WelfordMoments and PrincipalComponentAnalysis.

Welford expectations use exact rational batch central moments of the binary
input values; no streaming or merge recurrence is copied from the C# source.
PCA uses the independent numpy.linalg.eigh solver.

WelfordMoments
--------------
  Welford (1962) single-pass algorithm. The C# accumulates:
    delta  = x - mean_prev
    mean  += delta / count
    m2    += delta * (x - mean_new)
  Variance = m2 / (n - 1)   (sample, N-1 divisor)
  PopulationVariance = m2 / n
  StdDev = sqrt(Variance)

  Pearson inner class: bivariate Welford (Bennett 2009 form).
    coMoment += deltaX * (y - meanY_new)
  Covariance = coMoment / (n - 1)
  Correlation = coMoment / sqrt(m2X * m2Y)

  Merge (Chan-Golub-LeVeque 1979):
    delta  = meanB - meanA
    mean   = meanA + delta * nB / (nA + nB)
    m2     = m2A + m2B + delta^2 * nA * nB / (nA + nB)

PrincipalComponentAnalysis
--------------------------
  Uses numpy.linalg.eigh on a symmetric matrix.
  Post-processing to match C# exactly:
    1. Sort eigenvalues descending, reorder eigenvector columns.
    2. Sign convention: for each column, if the component with the
       largest absolute value is negative, flip the entire column.
    3. Explained variance ratio = λk / sum(max(λk, 0)) (clamp negatives).
    4. Cumulative = cumsum of ratio.

  FromReturns: column means, sample covariance (ddof=1), then PCA on cov.

Library pins:
  numpy>=2  scipy>=1.13
Fixed seed per section — regenerating produces bit-identical JSON.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from conftest import save_vector

SEED = 20260417


# =========================================================================
#  Independent batch central-moment oracle
# =========================================================================


def welford_moments(series: list[float]) -> dict:
    """Exact rational batch central moments; independent of streaming updates."""
    values = [Fraction.from_float(float(x)) for x in series]
    n = len(values)
    mean = sum(values, Fraction()) / n if n else Fraction()
    squares = sum(((x - mean) ** 2 for x in values), Fraction())
    variance = squares / (n - 1) if n >= 2 else Fraction()
    return {
        "count": n,
        "mean": float(mean),
        "variance": float(variance),
        "population_variance": float(squares / n) if n else 0.0,
        "std_dev": math.sqrt(float(variance)),
    }


def welford_pearson(xs: list[float], ys: list[float]) -> dict:
    """Exact rational batch products; normalize before conversion to float."""
    assert len(xs) == len(ys)
    n = len(xs)
    xx = [Fraction.from_float(float(x)) for x in xs]
    yy = [Fraction.from_float(float(y)) for y in ys]
    mean_x = sum(xx, Fraction()) / n if n else Fraction()
    mean_y = sum(yy, Fraction()) / n if n else Fraction()
    dx = [x - mean_x for x in xx]
    dy = [y - mean_y for y in yy]
    squares_x = sum((x * x for x in dx), Fraction())
    squares_y = sum((y * y for y in dy), Fraction())
    products = sum((x * y for x, y in zip(dx, dy)), Fraction())
    squared_correlation = products ** 2 / (squares_x * squares_y) if squares_x and squares_y else Fraction()
    correlation = math.copysign(math.sqrt(float(squared_correlation)), -1 if products < 0 else 1)
    return {
        "count": n,
        "mean_x": float(mean_x),
        "mean_y": float(mean_y),
        "covariance": float(products / (n - 1)) if n >= 2 else 0.0,
        "correlation": correlation,
    }


def welford_merge(a_vals: list[float], b_vals: list[float]) -> dict:
    """Recompute the combined population without any merge recurrence."""
    full = welford_moments(a_vals + b_vals)
    return {
        "partition_a": a_vals,
        "partition_b": b_vals,
        "merged_mean": full["mean"],
        "merged_variance": full["variance"],
        "full_mean": full["mean"],
        "full_variance": full["variance"],
    }


# =========================================================================
#  PCA reference implementation (mirrors C# PrincipalComponentAnalysis<T>)
# =========================================================================


def _apply_sign_convention(vectors: np.ndarray) -> np.ndarray:
    """Mirror of PrincipalComponentAnalysis<T>.ApplySignConvention.

    For each eigenvector column, if the component with the largest absolute
    value is negative, flip the entire column.
    """
    v = vectors.copy()
    for k in range(v.shape[1]):
        col = v[:, k]
        max_abs_idx = int(np.argmax(np.abs(col)))
        if col[max_abs_idx] < 0:
            v[:, k] = -col
    return v


def _explained_variance(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of PrincipalComponentAnalysis<T>.ComputeExplainedVariance."""
    positive = np.where(eigenvalues > 0, eigenvalues, 0.0)
    total = float(positive.sum())
    if total == 0.0:
        return np.zeros_like(eigenvalues), np.zeros_like(eigenvalues)
    ratio = positive / total
    cumulative = np.cumsum(ratio)
    return ratio, cumulative


def pca_decompose(cov: np.ndarray) -> dict:
    """PCA via numpy.linalg.eigh + C# post-processing."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Apply sign convention
    eigenvectors = _apply_sign_convention(eigenvectors)
    ratio, cumulative = _explained_variance(eigenvalues)
    return {
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "explained_variance_ratio": ratio.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
    }


def pca_from_returns(returns: np.ndarray) -> dict:
    """PCA.FromReturns: column means → sample cov (ddof=1) → PCA."""
    t, n = returns.shape
    mean = returns.mean(axis=0)
    cov = np.cov(returns.T, ddof=1)  # n×n sample covariance
    result = pca_decompose(cov)
    result["mean"] = mean.tolist()
    result["t"] = t
    result["n"] = n
    return result


# =========================================================================
#  Orchestrator
# =========================================================================


def generate() -> None:
    rng = np.random.default_rng(SEED)

    # ---- WelfordMoments: small integer series -------------------------
    small_series = [1.0, 2.0, 3.0, 4.0, 5.0]
    small_result = welford_moments(small_series)

    # ---- WelfordMoments: float series (50 values) --------------------
    float_series = rng.normal(loc=0.0005, scale=0.01, size=50).tolist()
    float_result = welford_moments(float_series)

    # ---- WelfordMoments: bivariate Pearson ---------------------------
    n_biv = 80
    biv_x = rng.normal(loc=0.0, scale=1.0, size=n_biv).tolist()
    # y is correlated with x: y = 0.6*x + noise
    biv_y = (0.6 * np.array(biv_x) + rng.normal(scale=0.8, size=n_biv)).tolist()
    pearson_result = welford_pearson(biv_x, biv_y)

    # ---- WelfordMoments: merge consistency ---------------------------
    # Partition the float_series into first-30 / last-20.
    merge_result = welford_merge(float_series[:30], float_series[30:])

    # ---- PCA: small 3×3 known covariance matrix ----------------------
    # Construct a simple symmetric PD matrix with well-separated eigenvalues
    # so the sign convention is unambiguous and eigenvectors are unique.
    raw3 = np.array([[4.0, 2.0, 0.6],
                     [2.0, 5.0, 1.5],
                     [0.6, 1.5, 3.0]])
    pca3 = pca_decompose(raw3)

    # ---- PCA: FromReturns on 50×4 return matrix ----------------------
    rng2 = np.random.default_rng(SEED + 1)  # independent stream for PCA
    # Common factor + idiosyncratic noise, giving moderate inter-asset correlation
    factor = rng2.normal(scale=0.01, size=50)
    returns_4 = np.column_stack([
        factor * 0.8 + rng2.normal(scale=0.005, size=50),
        factor * 0.6 + rng2.normal(scale=0.007, size=50),
        factor * 0.4 + rng2.normal(scale=0.009, size=50),
        factor * 0.2 + rng2.normal(scale=0.011, size=50),
    ])
    pca4 = pca_from_returns(returns_4)

    # ---- PCA: top-2 projection scores --------------------------------
    # Use the 50×4 result to project onto the top 2 PCs.
    # Center by mean, then multiply by the first 2 eigenvector columns.
    eigvec = np.array(pca4["eigenvectors"])
    means = np.array(pca4["mean"])
    centered = returns_4 - means
    scores_top2 = (centered @ eigvec[:, :2]).tolist()

    save_vector(
        "welford_pca",
        {
            "library_pins": {"numpy": "2"},
            "welford_moments": {
                "small_integer_series": {
                    "series": small_series,
                    **small_result,
                },
                "float_series": {
                    "series": float_series,
                    **float_result,
                },
                "bivariate_pearson": {
                    "x": biv_x,
                    "y": biv_y,
                    **pearson_result,
                },
                "merge_consistency": merge_result,
            },
            "pca": {
                "decompose_3x3": {
                    "covariance_matrix": raw3.tolist(),
                    **pca3,
                },
                "from_returns_50x4": {
                    "returns": returns_4.tolist(),
                    **pca4,
                    "scores_top2": scores_top2,
                },
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote welford_pca.json")
