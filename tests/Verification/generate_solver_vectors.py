"""Root-finding test functions with known roots.

The top-level ``cases`` pool is shared across every root-solver
verification test (Brent, Bisection, Newton, Secant, Muller) because the
mathematical roots are independent of the solver used to find them. Each
case stores the analytic or 16-digit-certified root of the test function;
the verification harness asserts every solver converges to that value.

For Muller — which has no scipy equivalent — we use
``mpmath.findroot(..., solver='muller')`` as the independent reference.
mpmath is an arbitrary-precision library (part of the SymPy project,
separate authors and codebase) with its own Muller implementation. This
matters because a numpy hand-port of Faires & Burden §2.6 would share
the same algorithmic choices as the Numerics C# class — a sign-flip bug
in the textbook's "choose denominator with larger magnitude" rule would
propagate to both implementations and the cross-check would pass while
shipping wrong numbers. mpmath runs at a different precision against a
different algorithm layout and so catches such bugs at generation time.

Library pins:
* mpmath==1.3+ (findroot solver='muller')
"""

from __future__ import annotations

import math
from typing import Callable

import mpmath

from conftest import save_vector


# =====================================================================
#  Test-function pool — shared with the C# SolverVerificationTests.
# =====================================================================


def _resolve_function(name: str) -> Callable[[float], float]:
    """Double-precision versions of the test functions (for generator
    self-checks)."""

    if name == "cubic":
        return lambda x: x * x * x - x - 2.0
    if name == "quadratic":
        return lambda x: x * x - 2.0
    if name == "log":
        return lambda x: math.log(x) - 1.0
    if name == "cosine":
        return lambda x: math.cos(x) - x
    if name == "exp_minus_linear":
        return lambda x: math.exp(-x) - x
    raise ValueError(f"Unknown function name: {name}")


def _resolve_mpmath_function(name: str) -> Callable[[mpmath.mpf], mpmath.mpf]:
    """mpmath.mpf (arbitrary-precision) versions of the same test
    functions, for use with ``mpmath.findroot``. The ``log``, ``cos``,
    and ``exp`` functions from the ``math`` module do not accept mpf
    arguments, so the mpmath equivalents are substituted here."""

    if name == "cubic":
        return lambda x: x * x * x - x - 2
    if name == "quadratic":
        return lambda x: x * x - 2
    if name == "log":
        return lambda x: mpmath.log(x) - 1
    if name == "cosine":
        return lambda x: mpmath.cos(x) - x
    if name == "exp_minus_linear":
        return lambda x: mpmath.exp(-x) - x
    raise ValueError(f"Unknown function name: {name}")


def generate() -> None:
    cases = [
        {
            "name": "cubic",
            "description": "f(x) = x^3 - x - 2",
            "bracket": [1.0, 2.0],
            "initial_guess": 1.5,
            "root": 1.5213797068045676,
        },
        {
            "name": "quadratic",
            "description": "f(x) = x^2 - 2  (positive root)",
            "bracket": [1.0, 2.0],
            "initial_guess": 1.5,
            "root": math.sqrt(2.0),
        },
        {
            "name": "log",
            "description": "f(x) = log(x) - 1",
            "bracket": [2.0, 3.0],
            "initial_guess": 2.5,
            "root": math.e,
        },
        {
            "name": "cosine",
            "description": "f(x) = cos(x) - x (Dottie number)",
            "bracket": [0.0, 1.0],
            "initial_guess": 0.75,
            "root": 0.7390851332151607,
        },
        {
            "name": "exp_minus_linear",
            "description": "f(x) = e^{-x} - x",
            "bracket": [0.0, 1.0],
            "initial_guess": 0.5,
            "root": 0.5671432904097838,
        },
    ]

    # Validate every stored root against mpmath's independent Muller
    # implementation. Fails the generator loudly if the case's certified
    # value drifts from the mpmath-produced root — a smarter check than a
    # numpy-hand-port assertion because mpmath's internals share no code
    # or algorithmic choices with the Numerics C# ``MullerSolver``, so a
    # shared implementation bug cannot produce a false-positive match.
    for c in cases:
        f_mp = _resolve_mpmath_function(c["name"])
        # Muller seed: the stored ``initial_guess`` at double precision.
        # mpmath.findroot auto-perturbs two flanking points internally;
        # same convention as the Numerics single-point overload
        # (Δ = max(1e-4, |x|·1e-4)).
        mpmath_root = float(mpmath.findroot(f_mp, mpmath.mpf(c["initial_guess"]), solver="muller"))
        certified = c["root"]
        drift = abs(mpmath_root - certified)
        if drift > 1e-13:
            raise AssertionError(
                f"mpmath reference drift on '{c['name']}': "
                f"mpmath returned {mpmath_root}, certified {certified}, "
                f"|Δ| = {drift:.3e}"
            )

    save_vector(
        "solvers",
        {
            "library_pins": {"mpmath": "1.3+"},
            "muller_reference": "mpmath.findroot(..., solver='muller')",
            "muller_reference_tolerance": 1e-13,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote solvers.json")
