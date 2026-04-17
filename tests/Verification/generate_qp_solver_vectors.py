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


def make_psd_cov(vols: list[float], corr: np.ndarray) -> np.ndarray:
    """Construct covariance from volatilities and correlation matrix."""
    d = np.diag(vols)
    return d @ corr @ d


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
    assert result.success or result.status == 9, (
        f"SLSQP failed: {result.message}"
    )
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
    assert result.success or result.status == 9, (
        f"SLSQP failed: {result.message}"
    )
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
