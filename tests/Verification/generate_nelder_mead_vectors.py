#!/usr/bin/env python3
"""
generate_nelder_mead_vectors.py
Cross-language reference vectors for NelderMeadMinimizer<T>.

Run with:
    python3 tests/Verification/generate_nelder_mead_vectors.py

Requires scipy >= 1.7.
"""

import math

from scipy.optimize import minimize

from conftest import save_vector


def rosenbrock(x):
    """f(x,y) = (1-x)^2 + 100*(y-x^2)^2, min at [1,1]"""
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def sphere_3d(x):
    """f(x,y,z) = x^2 + y^2 + z^2, min at [0,0,0]"""
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


def beale(x):
    """Beale function, min at [3, 0.5]"""
    t1 = 1.5 - x[0] + x[0] * x[1]
    t2 = 2.25 - x[0] + x[0] * x[1] ** 2
    t3 = 2.625 - x[0] + x[0] * x[1] ** 3
    return t1 ** 2 + t2 ** 2 + t3 ** 2


CASES = [
    {
        "name": "rosenbrock",
        "function": rosenbrock,
        "x0": [0.0, 0.0],
        "expected_argmin": [1.0, 1.0],
        "expected_fmin": 0.0,
        "options": {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
        "precision": 1e-4,
    },
    {
        "name": "sphere_3d",
        "function": sphere_3d,
        "x0": [1.0, 2.0, -1.0],
        "expected_argmin": [0.0, 0.0, 0.0],
        "expected_fmin": 0.0,
        "options": {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
        "precision": 1e-6,
    },
    {
        "name": "beale",
        "function": beale,
        "x0": [0.0, 0.0],
        "expected_argmin": [3.0, 0.5],
        "expected_fmin": 0.0,
        "options": {"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        "precision": 1e-3,
    },
]


def run_case(case):
    result = minimize(
        case["function"],
        case["x0"],
        method="Nelder-Mead",
        options=case["options"],
    )
    return {
        "name": case["name"],
        "x0": case["x0"],
        "argmin": result.x.tolist(),
        "fmin": float(result.fun),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "precision": case["precision"],
        "expected_argmin": case["expected_argmin"],
        "expected_fmin": case["expected_fmin"],
    }


def generate():
    cases = [run_case(c) for c in CASES]

    # Verify each case meets its precision requirement
    for c in cases:
        argmin = c["argmin"]
        expected = c["expected_argmin"]
        prec = c["precision"]
        for i, (a, e) in enumerate(zip(argmin, expected)):
            err = abs(a - e)
            assert err <= prec, (
                f"Case {c['name']} coord {i}: |{a} - {e}| = {err} > {prec}"
            )
        assert c["fmin"] <= prec ** 2 + 1e-10, (
            f"Case {c['name']} fmin {c['fmin']} > {prec**2}"
        )
        print(f"  {c['name']}: argmin={[round(v, 8) for v in argmin]}, "
              f"fmin={c['fmin']:.3e}, iters={c['iterations']}, "
              f"converged={c['converged']}")

    output = {
        "description": "Nelder-Mead minimiser reference vectors (scipy.optimize.minimize, method='Nelder-Mead')",
        "cases": cases,
    }

    save_vector("nelder_mead", output)

    print("\nWrote vectors/nelder_mead.json")


if __name__ == "__main__":
    generate()
