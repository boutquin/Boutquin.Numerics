"""Cholesky / Gaussian elimination / Jacobi eigendecomposition reference values."""

from __future__ import annotations

import numpy as np

from conftest import save_vector


def generate() -> None:
    rng = np.random.default_rng(seed=42)

    # Small SPD matrix: A · Aᵀ + n·I.
    n = 4
    raw = rng.normal(size=(n, n))
    spd = raw @ raw.T + n * np.eye(n)

    cholesky = np.linalg.cholesky(spd)
    eigenvalues, eigenvectors = np.linalg.eigh(spd)

    # Linear system: find x given A and b = A·ones.
    b = spd @ np.ones(n)
    x = np.linalg.solve(spd, b)

    save_vector(
        "linalg",
        {
            "spd_matrix": spd.tolist(),
            "cholesky_lower": cholesky.tolist(),
            "eigenvalues_ascending": eigenvalues.tolist(),
            "eigenvectors_columns": eigenvectors.tolist(),
            "linear_system": {
                "b": b.tolist(),
                "x": x.tolist(),
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote linalg.json")
