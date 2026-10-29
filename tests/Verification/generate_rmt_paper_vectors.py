"""Replay López de Prado (2020), snippets 2.3–2.4 / Figure 2.2.

The book omits the random realization. Seed 2026 is a declared test fixture,
not a reconstruction of the author's unpublished seed. Store the NumPy
correlation spectrum so routine .NET tests need neither NumPy nor a dense
1000-by-1000 eigendecomposition. SciPy minimizes the primary MP/KDE objective;
the reference is generated without calling or porting the C# fitter.

Run: OPENBLAS_NUM_THREADS=1 python3 generate_rmt_paper_vectors.py
"""

from __future__ import annotations

import numpy as np
import scipy
from scipy.optimize import minimize_scalar

from conftest import save_vector


def main() -> None:
    seed, observations, variables, factors = 2026, 10000, 1000, 100
    alpha, bandwidth = 0.995, 0.01
    rng = np.random.RandomState(seed)
    sample = np.cov(rng.normal(size=(observations, variables)), rowvar=False)
    loadings = rng.normal(size=(variables, factors))
    signal = loadings @ loadings.T + np.diag(rng.uniform(size=variables))
    covariance = alpha * sample + (1 - alpha) * signal
    standard_deviations = np.sqrt(np.diag(covariance))
    correlation = np.clip(
        covariance / np.outer(standard_deviations, standard_deviations), -1, 1
    )
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]

    q = observations / variables
    left, right = (1 - np.sqrt(1 / q)) ** 2, (1 + np.sqrt(1 / q)) ** 2
    abscissae = np.linspace(left, right, 1000)
    density = q * np.sqrt(
        np.maximum(0, (right - abscissae) * (abscissae - left))
    ) / (2 * np.pi * abscissae)

    def objective(variance: float) -> float:
        distances = (variance * abscissae[:, None] - eigenvalues) / bandwidth
        empirical = np.exp(-0.5 * distances**2).sum(axis=1)
        empirical /= variables * bandwidth * np.sqrt(2 * np.pi)
        return float(np.sum((density / variance - empirical) ** 2))

    # A denser independent scan locates every candidate basin; SciPy's bounded
    # minimizer refines them. This verifies the library's documented global fit,
    # rather than depending on the book's local search starting at 0.5.
    grid = np.linspace(1e-5, 1 - 1e-5, 2049)
    scores = np.array([objective(value) for value in grid])
    candidates = [
        minimize_scalar(
            objective,
            bounds=(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        for i in range(len(grid))
        if (i == 0 or scores[i] <= scores[i - 1])
        and (i == len(grid) - 1 or scores[i] <= scores[i + 1])
    ]
    best = min(candidates, key=lambda result: result.fun)
    if not best.success:
        raise RuntimeError(best.message)
    edge = float(best.x * right)
    result = {
        "source": "López de Prado (2020), snippets 2.3–2.4, Figure 2.2",
        "provenance": "Declared seed; the book does not publish its realization",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "seed": seed,
        "observations": observations,
        "variables": variables,
        "factors": factors,
        "alpha": alpha,
        "q": q,
        "bandwidth": bandwidth,
        "eigenvalues": eigenvalues,
        "fitted_noise_variance": float(best.x),
        "mp_upper_edge": edge,
        "signal_count": int(np.count_nonzero(eigenvalues > edge)),
    }
    save_vector("paper_rmt", result)
    print(f"sigma²={best.x:.12g}; edge={edge:.12g}; factors={result['signal_count']}")


if __name__ == "__main__":
    main()
