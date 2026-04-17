"""Levenberg–Marquardt reference vectors — scipy ``least_squares`` cross-check.

Four smooth regimes (none overlap with the NIST StRD Nonlinear suite that
already covers the Numerics LM solver):

* ``exponential_decay_noisy`` — ``y = a·exp(−b·t) + ε``, Gaussian noise.
* ``sum_of_sinusoids`` — ``y = a·sin(ω₁t) + b·cos(ω₂t)``, well-separated
  frequencies.
* ``gompertz_growth`` — ``y = a·exp(−b·exp(−c·t))``, the classical Gompertz
  growth curve.
* ``logistic`` — ``y = a / (1 + exp(−b·(t − c)))``, standard logistic with
  three parameters.

``scipy.optimize.least_squares(method='lm')`` delegates to MINPACK lmder —
the same family of algorithms as the Numerics LM solver, so precision
parity at ``1e-5`` relative is achievable on smooth problems.

Library pins:

* scipy==1.13+ (any ``least_squares(method='lm')`` compatible version)
* numpy==2.x

Fixed numpy seed at the top; same seed → bit-identical JSON.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from conftest import save_vector


SEED = 20260415


def _fit_and_record(
    name: str,
    residual: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
) -> dict:
    """Run scipy LM and return the JSON-serialisable reference block."""

    result = least_squares(
        residual,
        theta0,
        args=(t, y),
        method="lm",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=10_000,
    )

    if not result.success:
        raise RuntimeError(
            f"scipy LM failed on regime '{name}': status={result.status} message={result.message}"
        )

    # scipy ``least_squares`` reports ``cost = 0.5 · Σ rᵢ²`` — the same
    # convention Numerics' ``MultivariateSolverResult.FinalCost`` uses, so
    # no scaling adjustment is needed on the C# side.
    return {
        "t": t.tolist(),
        "y": y.tolist(),
        "initial_guess": theta0.tolist(),
        "parameters": result.x.tolist(),
        "final_cost": float(result.cost),
    }


# ---------------------------------------------------------------------
# Residual functions — ``r(θ) = y_model(θ, t) − y_obs``. Copied verbatim
# from the model equation to make the port traceable; Numerics tests use
# the identical closed form.
# ---------------------------------------------------------------------


def residual_exponential_decay(theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``y = θ₀ · exp(−θ₁ · t)``."""

    a, b = theta
    return a * np.exp(-b * t) - y


def residual_sum_of_sinusoids(theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``y = θ₀ · sin(θ₂ · t) + θ₁ · cos(θ₃ · t)``."""

    a, b, omega1, omega2 = theta
    return a * np.sin(omega1 * t) + b * np.cos(omega2 * t) - y


def residual_gompertz(theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``y = θ₀ · exp(−θ₁ · exp(−θ₂ · t))``."""

    a, b, c = theta
    return a * np.exp(-b * np.exp(-c * t)) - y


def residual_logistic(theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``y = θ₀ / (1 + exp(−θ₁ · (t − θ₂)))``."""

    a, b, c = theta
    return a / (1.0 + np.exp(-b * (t - c))) - y


def generate() -> None:
    rng = np.random.default_rng(SEED)

    regimes: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 1. Exponential decay with Gaussian noise.
    # ------------------------------------------------------------------
    t1 = np.linspace(0.0, 5.0, 30)
    a1_true, b1_true = 2.5, 0.8
    y1 = a1_true * np.exp(-b1_true * t1) + rng.normal(scale=0.01, size=t1.size)
    theta0_1 = np.array([1.0, 1.0])
    regimes["exponential_decay_noisy"] = _fit_and_record(
        "exponential_decay_noisy", residual_exponential_decay, t1, y1, theta0_1
    )

    # ------------------------------------------------------------------
    # 2. Sum of two sinusoids with well-separated frequencies.
    # ------------------------------------------------------------------
    t2 = np.linspace(0.0, 4.0 * np.pi, 80)
    a2_true, b2_true = 1.5, 0.7
    omega1_true, omega2_true = 1.0, 3.0
    y2 = (
        a2_true * np.sin(omega1_true * t2)
        + b2_true * np.cos(omega2_true * t2)
        + rng.normal(scale=0.005, size=t2.size)
    )
    theta0_2 = np.array([1.0, 1.0, 0.9, 2.8])
    regimes["sum_of_sinusoids"] = _fit_and_record(
        "sum_of_sinusoids", residual_sum_of_sinusoids, t2, y2, theta0_2
    )

    # ------------------------------------------------------------------
    # 3. Gompertz growth curve.
    # ------------------------------------------------------------------
    t3 = np.linspace(0.0, 10.0, 50)
    a3_true, b3_true, c3_true = 10.0, 3.0, 0.5
    y3 = a3_true * np.exp(-b3_true * np.exp(-c3_true * t3)) + rng.normal(scale=0.02, size=t3.size)
    theta0_3 = np.array([8.0, 2.0, 0.3])
    regimes["gompertz_growth"] = _fit_and_record(
        "gompertz_growth", residual_gompertz, t3, y3, theta0_3
    )

    # ------------------------------------------------------------------
    # 4. Logistic.
    # ------------------------------------------------------------------
    t4 = np.linspace(-5.0, 5.0, 40)
    a4_true, b4_true, c4_true = 1.0, 1.5, 0.5
    y4 = a4_true / (1.0 + np.exp(-b4_true * (t4 - c4_true))) + rng.normal(scale=0.005, size=t4.size)
    theta0_4 = np.array([0.5, 1.0, 0.0])
    regimes["logistic"] = _fit_and_record(
        "logistic", residual_logistic, t4, y4, theta0_4
    )

    save_vector(
        "lm",
        {
            "seed": SEED,
            "library_pins": {"scipy": "1.13+", "numpy": "2"},
            "regimes": regimes,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote lm.json")
