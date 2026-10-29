"""Independent arch 8.0.0 recursion + natural-coordinate SLSQP qualification.

Calls arch's installed implementation; does not copy its source or the C#
likelihood. Aligns the library's candidate-dependent presample second moment,
all-observation mean convention, omitted Gaussian constant, and fixed variance
target. No claim of whole arch.fit parity or a global QMLE optimum is made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import arch
import numpy as np
import scipy
from arch.univariate import GARCH, Normal
from scipy.optimize import minimize

from conftest import save_vector

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "vectors/garch.json"


def generate():
    if arch.__version__ != "8.0.0":
        raise RuntimeError("This oracle is pinned to arch 8.0.0")
    # Reuse the independently simulated observations already committed in garch.json.
    # Change units only: scaling by 100 makes the natural intercept well-conditioned.
    observations = 100 * np.asarray(json.loads(SOURCE.read_text())["returns"])
    volatility, distribution = GARCH(p=1, o=0, q=1, power=2.0), Normal()
    bounds = np.tile([0.0, np.inf], (len(observations), 1))
    lagged = np.r_[observations[0], observations[:-1]]
    cases = []
    for name, fixed_zero, ar1, targeting in [
        ("constant_mean", False, False, False),
        ("zero_mean", True, False, False),
        ("ar1_mean", False, True, False),
        ("variance_target", False, False, True),
    ]:
        initial_mean = 0.0 if fixed_zero else float(np.mean(observations))
        target = float(np.mean((observations - initial_mean) ** 2))

        def decode(parameters):
            offset = 0
            mu = 0.0 if fixed_zero else parameters[offset]
            offset += not fixed_zero
            phi = parameters[offset] if ar1 else 0.0
            offset += ar1
            if targeting:
                alpha, beta = parameters[offset:offset + 2]
                omega = (1 - alpha - beta) * target
            else:
                omega, alpha, beta = parameters[offset:offset + 3]
            return float(mu), float(phi), float(omega), float(alpha), float(beta)

        def evaluate(parameters):
            mu, phi, omega, alpha, beta = decode(parameters)
            residuals = observations - mu - phi * lagged
            backcast = float(np.mean(residuals ** 2))
            variances = volatility.compute_variance(
                np.array([omega, alpha, beta]), residuals,
                np.empty(len(observations)), backcast, bounds)
            if not np.all(np.isfinite(variances)) or not np.all(variances > 0):
                return 1e50, residuals, variances, backcast
            # arch reports a sum including -log(2*pi)/2. Numerics reports the
            # average and omits that constant; neither changes the optimum.
            nll = -float(distribution.loglikelihood([], residuals, variances)) / len(observations)
            nll -= np.log(2 * np.pi) / 2
            return nll, residuals, variances, backcast

        prefix = ([] if fixed_zero else [initial_mean]) + ([0.0] if ar1 else [])
        search_bounds = ([] if fixed_zero else [(None, None)]) + ([(-.999999, .999999)] if ar1 else [])
        search_bounds += ([] if targeting else [(1e-10, None)]) + [(1e-10, .999999999), (1e-10, .999999999)]
        results = []
        for alpha, beta in [(.10, .80), (.20, .60), (.05, .90)]:
            start = prefix + ([] if targeting else [(1 - alpha - beta) * target]) + [alpha, beta]
            result = minimize(
                lambda p: evaluate(p)[0], start, method="SLSQP", bounds=search_bounds,
                constraints={"type": "ineq", "fun": lambda p: .999999999 - p[-2] - p[-1]},
                options={"maxiter": 1500, "ftol": 1e-12})
            if not result.success or not np.isfinite(result.fun):
                raise AssertionError(f"{name}: independent constrained fit failed: {result.message}")
            results.append(result)
        best = min(results, key=lambda result: result.fun)
        spread = max(result.fun for result in results) - best.fun
        if spread > 1e-8:
            raise AssertionError(f"{name}: independent starts disagree in objective: {spread}")
        mu, phi, omega, alpha, beta = decode(best.x)
        nll, residuals, variances, backcast = evaluate(best.x)
        forecasts = volatility.forecast(
            np.array([omega, alpha, beta]), residuals, backcast, bounds,
            horizon=10, method="analytic").forecasts[-1]
        cases.append({
            "name": name, "fixed_zero_mean": fixed_zero, "ar1_mean": ar1,
            "variance_targeting": targeting, "mu": mu, "phi": phi,
            "omega": omega, "alpha": alpha, "beta": beta,
            "log_likelihood": -nll, "start_objective_spread": float(spread),
            "optimizer_successes": len(results),
            "conditional_variances": variances.tolist(),
            "standardized_residuals": (residuals / np.sqrt(variances)).tolist(),
            "forecasts": forecasts.tolist(),
        })
        print(name, mu, phi, omega, alpha, beta, -nll, "start spread", spread)
    output = {
        "arch_version": arch.__version__, "scipy_version": scipy.__version__,
        "numpy_version": np.__version__, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_observations": len(observations), "input_scale": 100,
        "scope": "arch recursion/normal likelihood and independent SLSQP natural-coordinate local fits; no global-optimum claim",
        "cases": cases,
    }
    save_vector("garch_independent_qualification", output)


if __name__ == "__main__":
    generate()
