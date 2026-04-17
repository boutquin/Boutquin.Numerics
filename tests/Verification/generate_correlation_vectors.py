"""Pearson / Spearman / Kendall / distance-correlation reference values."""

from __future__ import annotations

import numpy as np
from scipy import stats

from conftest import save_vector


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Székely-Rizzo-Bakirov (2007) distance correlation, independent of C#."""

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)

    if dcov2_xx <= 0.0 or dcov2_yy <= 0.0:
        return 0.0
    return float(np.sqrt(dcov2_xy / np.sqrt(dcov2_xx * dcov2_yy)))


def generate() -> None:
    rng = np.random.default_rng(seed=7)
    n = 200
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(scale=0.5, size=n)

    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(stats.spearmanr(x, y).correlation)
    kendall = float(stats.kendalltau(x, y).correlation)
    dcor = distance_correlation(x, y)

    save_vector(
        "correlation",
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "pearson": pearson,
            "spearman": spearman,
            "kendall_tau_b": kendall,
            "distance_correlation": dcor,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote correlation.json")
