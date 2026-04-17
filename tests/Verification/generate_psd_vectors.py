"""Higham nearest-PSD vectors.

The Higham 1988 alternating-projection nearest correlation matrix is
implemented here to produce a reference against which the C# version can
be validated — scipy does not ship an implementation.
"""

from __future__ import annotations

import numpy as np

from conftest import save_vector


def project_spd(a: np.ndarray) -> np.ndarray:
    """Eigenvalue-clip projection onto the PSD cone."""

    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(vals) @ vecs.T


def higham_nearest_correlation(a: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Alternating-projection iteration toward the nearest correlation matrix."""

    n = a.shape[0]
    y = np.array(a, dtype=float, copy=True)
    delta_s = np.zeros_like(y)

    for _ in range(max_iter):
        r = y - delta_s
        x = project_spd(r)
        delta_s = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y, ord="fro") / max(np.linalg.norm(y, ord="fro"), 1e-12) < tol:
            y = y_new
            break
        y = y_new

    return 0.5 * (y + y.T)


def generate() -> None:
    # Classic Higham 1988 test case — not a valid correlation matrix.
    a = np.array(
        [
            [1.0, -0.55, -0.15, -0.1],
            [-0.55, 1.0, 0.9, 0.9],
            [-0.15, 0.9, 1.0, 0.9],
            [-0.1, 0.9, 0.9, 1.0],
        ],
        dtype=float,
    )
    nearest = higham_nearest_correlation(a)

    save_vector(
        "psd",
        {
            "input": a.tolist(),
            "nearest_correlation": nearest.tolist(),
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote psd.json")
