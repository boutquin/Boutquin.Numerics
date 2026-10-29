"""Active-set QP solver reference vectors.

Two portfolio-optimization problems are verified against
``scipy.optimize.minimize`` with method ``'SLSQP'``:

* **MinVar** — min w′Σw  s.t. 1′w=1, lb ≤ w_i ≤ ub
* **MeanVar** — max w′μ − (λ/2)w′Σw  s.t. 1′w=1, lb ≤ w_i ≤ ub

SLSQP (Kraft 1988) is a completely different algorithm from Boutquin.Numerics'
Cholesky-based active-set method. Because they solve the same convex QP, they
must produce identical optimal weights (up to solver tolerance). The tolerance
floor for the comparison is set by SLSQP's ``ftol=1e-12``; any residual below
1e-6 in per-weight absolute error is solver noise, not a bug.

Library pins:

* numpy==2.x
* scipy==1.13+

Fixed seed; regenerating produces bit-identical JSON.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from conftest import save_vector


SEED = 20260415


def _certify_bounded_simplex(weights: np.ndarray, gradient: np.ndarray, lb: float, ub: float) -> None:
    """Check feasibility and a convex first-order objective-gap certificate.

    For convex f, f(w)-min(f) <= grad(f,w) dot (w-v), where v minimizes
    grad(f,w) dot v on the box/simplex. Fill the smallest gradient first.
    The 1e-6 scaled gap is an objective certificate, not a per-weight bound.
    """
    if (weights.ndim != 1 or weights.shape != gradient.shape
            or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(gradient))
            or abs(float(np.sum(weights)) - 1.0) > 1e-8
            or np.any(weights < lb - 1e-8) or np.any(weights > ub + 1e-8)):
        raise RuntimeError("QP reference certificate failed: non-finite or infeasible weights")
    vertex = np.full(weights.size, lb, dtype=float)
    remaining = 1.0 - weights.size * lb
    for index in np.argsort(gradient, kind="stable"):
        amount = min(ub - lb, max(0.0, remaining))
        vertex[index] += amount
        remaining -= amount
    if abs(remaining) > 1e-8:
        raise RuntimeError("QP reference certificate failed: infeasible bounds")
    gap = float(gradient @ (weights - vertex))
    if gap > 1e-6 * max(1.0, float(np.max(np.abs(gradient)))):
        raise RuntimeError(f"QP reference certificate failed: convex objective gap {gap}")


def make_psd_cov(vols: list[float], corr: np.ndarray) -> np.ndarray:
    """Construct covariance from volatilities and correlation matrix."""
    d = np.diag(vols)
    cov = d @ corr @ d
    # Matrix products can round mirrored entries differently. Supply an exactly
    # symmetric covariance to both solvers, as the public QP contract requires.
    return (cov + cov.T) / 2


def solve_min_variance_slsqp(
    cov: np.ndarray, lb: float, ub: float
) -> list[float]:
    """SLSQP reference for min w′Σw s.t. 1′w=1, lb≤w≤ub."""
    n = cov.shape[0]
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(lb, ub)] * n
    result = minimize(
        lambda w: float(w @ cov @ w),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed with status {result.status}: {result.message}")
    _certify_bounded_simplex(result.x, 2.0 * cov @ result.x, lb, ub)
    return result.x.tolist()


def solve_mean_variance_slsqp(
    cov: np.ndarray,
    means: list[float],
    risk_aversion: float,
    lb: float,
    ub: float,
) -> list[float]:
    """SLSQP reference for max w′μ − (λ/2)w′Σw s.t. 1′w=1, lb≤w≤ub."""
    n = cov.shape[0]
    mu = np.asarray(means)
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(lb, ub)] * n
    result = minimize(
        lambda w: -float(w @ mu) + (risk_aversion / 2.0) * float(w @ cov @ w),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed with status {result.status}: {result.message}")
    _certify_bounded_simplex(result.x, risk_aversion * cov @ result.x - mu, lb, ub)
    return result.x.tolist()


def generate() -> None:
    cases = []

    # ── Case 1: 3-asset MinVar, no binding box constraint ─────────────────
    cov3 = np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.09, 0.03],
        [0.01, 0.03, 0.16],
    ])
    w_mv3 = solve_min_variance_slsqp(cov3, lb=0.0, ub=1.0)
    cases.append({
        "name": "minvar_3asset_unconstrained",
        "kind": "minvar",
        "covariance": cov3.tolist(),
        "min_weight": 0.0,
        "max_weight": 1.0,
        "weights": w_mv3,
    })

    # ── Case 2: 3-asset MinVar, box constraints force active bounds ────────
    w_mv3_box = solve_min_variance_slsqp(cov3, lb=0.1, ub=0.6)
    cases.append({
        "name": "minvar_3asset_boxed",
        "kind": "minvar",
        "covariance": cov3.tolist(),
        "min_weight": 0.1,
        "max_weight": 0.6,
        "weights": w_mv3_box,
    })

    # ── Case 3: 5-asset MinVar, moderate correlations ─────────────────────
    corr5 = np.array([
        [1.00, 0.35, 0.20, 0.10, 0.05],
        [0.35, 1.00, 0.30, 0.15, 0.10],
        [0.20, 0.30, 1.00, 0.25, 0.20],
        [0.10, 0.15, 0.25, 1.00, 0.40],
        [0.05, 0.10, 0.20, 0.40, 1.00],
    ])
    vols5 = [0.15, 0.20, 0.12, 0.25, 0.18]
    cov5 = make_psd_cov(vols5, corr5)
    w_mv5 = solve_min_variance_slsqp(cov5, lb=0.05, ub=0.50)
    cases.append({
        "name": "minvar_5asset_moderate_corr",
        "kind": "minvar",
        "covariance": cov5.tolist(),
        "min_weight": 0.05,
        "max_weight": 0.50,
        "weights": w_mv5,
    })

    # ── Case 4: 3-asset MeanVar ────────────────────────────────────────────
    means3 = [0.08, 0.12, 0.06]
    risk_aversion = 2.0
    w_mean3 = solve_mean_variance_slsqp(cov3, means3, risk_aversion, lb=0.0, ub=1.0)
    cases.append({
        "name": "meanvar_3asset_unconstrained",
        "kind": "meanvar",
        "covariance": cov3.tolist(),
        "means": means3,
        "risk_aversion": risk_aversion,
        "min_weight": 0.0,
        "max_weight": 1.0,
        "weights": w_mean3,
    })

    # ── Case 5: 5-asset MeanVar with box constraints ──────────────────────
    means5 = [0.09, 0.12, 0.07, 0.15, 0.10]
    w_mean5 = solve_mean_variance_slsqp(cov5, means5, risk_aversion=3.0, lb=0.05, ub=0.45)
    cases.append({
        "name": "meanvar_5asset_boxed",
        "kind": "meanvar",
        "covariance": cov5.tolist(),
        "means": means5,
        "risk_aversion": 3.0,
        "min_weight": 0.05,
        "max_weight": 0.45,
        "weights": w_mean5,
    })

    save_vector(
        "qp_solver",
        {
            "library_pins": {"numpy": "2", "scipy": "1.13+"},
            "cases": cases,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote qp_solver.json")
