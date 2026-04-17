"""Ordinary least-squares reference vectors — statsmodels.OLS cross-check.

Covers three regimes:

* ``well_conditioned`` — random design matrix with ``cond(X) ≈ 10²``; the
  easy bar that every double-precision OLS implementation must clear.
* ``polynomial_degree_5`` — Vandermonde-style design with columns ``x, x²,
  …, x⁵``; ``cond(X) ≈ 10⁶``; a moderately-conditioned regime that
  exercises the QR pivoting path.
* ``no_intercept`` — well-conditioned design fit with no intercept column;
  verifies the uncentred R² convention.

Library pins (documented so regression detection is decoupled from a
library update shifting the reference):

* statsmodels==0.14.x
* numpy==2.x

Fixed numpy seed at the top; same seed → bit-identical JSON.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from conftest import save_vector


SEED = 20260415


def fit_with_statsmodels(x_no_const: np.ndarray, y: np.ndarray, include_intercept: bool) -> dict:
    """Fit ``y ~ x`` with statsmodels.OLS and return the quantities the
    C# :class:`OlsResult` carries.

    statsmodels emits the same β̂, classical standard errors, RSS, σ̂,
    degrees of freedom, and R² we assert against. The returned dict is
    already JSON-serialisable.
    """

    design = sm.add_constant(x_no_const, has_constant="add") if include_intercept else x_no_const
    model = sm.OLS(y, design)
    results = model.fit()

    # statsmodels reports ``centered_tss`` for intercept models and
    # ``uncentered_tss`` for no-intercept models, and its ``rsquared``
    # matches the convention the C# ``OlsResult`` uses. Read both for
    # diagnostic clarity.
    return {
        "x": x_no_const.tolist(),
        "y": y.tolist(),
        "include_intercept": bool(include_intercept),
        "coefficients": results.params.tolist(),
        "standard_errors": results.bse.tolist(),
        "residual_sum_of_squares": float(results.ssr),
        "residual_standard_deviation": float(np.sqrt(results.mse_resid)),
        "degrees_of_freedom": int(results.df_resid),
        "r_squared": float(results.rsquared),
    }


def generate() -> None:
    rng = np.random.default_rng(SEED)

    regimes: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 1. Well-conditioned — random design with ``cond(X) ≈ 10²``.
    # ------------------------------------------------------------------
    m1, p1 = 40, 3
    x1 = rng.normal(size=(m1, p1))
    beta1 = np.array([1.0, -0.5, 2.0, 0.75])  # intercept + 3 slopes
    noise1 = rng.normal(scale=0.1, size=m1)
    y1 = beta1[0] + x1 @ beta1[1:] + noise1
    regimes["well_conditioned"] = fit_with_statsmodels(x1, y1, include_intercept=True)

    # ------------------------------------------------------------------
    # 2. Polynomial degree 5 — Vandermonde in x ∈ [0, 2], ``cond ≈ 10⁶``.
    # ------------------------------------------------------------------
    m2 = 50
    x_vec = np.linspace(0.0, 2.0, m2)
    x2 = np.column_stack([x_vec ** k for k in range(1, 6)])  # x, x², x³, x⁴, x⁵
    # True polynomial: 0.5 + 2x − x² + 0.3x³ − 0.1x⁴ + 0.02x⁵
    beta2 = np.array([0.5, 2.0, -1.0, 0.3, -0.1, 0.02])
    noise2 = rng.normal(scale=0.01, size=m2)
    y2 = beta2[0] + x2 @ beta2[1:] + noise2
    regimes["polynomial_degree_5"] = fit_with_statsmodels(x2, y2, include_intercept=True)

    # ------------------------------------------------------------------
    # 3. No intercept — well-conditioned; uncentred R² convention.
    # ------------------------------------------------------------------
    m3, p3 = 30, 2
    x3 = rng.uniform(0.5, 2.5, size=(m3, p3))
    beta3 = np.array([1.5, -0.7])  # pure slopes, no intercept
    noise3 = rng.normal(scale=0.05, size=m3)
    y3 = x3 @ beta3 + noise3
    regimes["no_intercept"] = fit_with_statsmodels(x3, y3, include_intercept=False)

    save_vector(
        "ols",
        {
            "seed": SEED,
            "library_pins": {"statsmodels": "0.14", "numpy": "2"},
            "regimes": regimes,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote ols.json")
