"""Independent Jarre (arXiv:2507.05083v1) Tables 1–5 controls.

Uses SciPy's spline implementation, power-basis quartics for Q endpoints, and
an unreduced dense curvature system for RNAK (§4 plus both §4.1 safeguards).
Extrema are roots of s' - f', bracketed within every knot interval and refined
with Brent's method. Production tests instead maximize |s-f| directly.

The paper does not publish the Table 3/4 knots or RNG seed. Those cases are
explicit reconstructions: sorted xorshift32 uniforms, affinely scaled so their
minimum/maximum equal the interval endpoints. Seeds are 250705083 + table.
They are not attempts to reproduce the printed random realizations.

Some uniform Q/RNAK entries disagree with the literal preprint definitions.
Both printed values and independent values are retained; disagreement is
explicitly asserted, never concealed by widening the table-rounding tolerance.
The later journal version changes the RNAK safeguard and several printed cells;
this fixture deliberately targets the cited July 2025 preprint only.

Run: python3 tests/Verification/generate_jarre_spline_vectors.py
Requires NumPy and SciPy; neither the library nor .NET is used.
"""
from __future__ import annotations

from conftest import VECTORS_DIR, save_vector

import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

COUNTS = [6, 12, 24, 48, 96]
KINDS = ["NAT", "NAK", "Q", "RNAK"]
PRINTED = {
    1: [[4.5e-4, 1.8e-5, 9.1e-7, 5.2e-8, 3.1e-9],
        [2.7e-3, 5.5e-5, 1.4e-6, 5.2e-8, 3.1e-9],
        [2.2e-3, 4.0e-5, 9.6e-7, 5.6e-8, 3.1e-9],
        [1.6e-3, 1.8e-5, 9.1e-7, 5.2e-8, 3.1e-9]],
    2: [[1.4e-2, 2.9e-3, 6.5e-4, 1.6e-4, 3.8e-5],
        [4.3e-3, 1.7e-4, 7.9e-6, 4.3e-7, 2.5e-8],
        [1.6e-3, 5.5e-5, 2.2e-6, 1.1e-7, 6.0e-9],
        [1.6e-3, 4.6e-5, 9.1e-7, 5.2e-8, 3.1e-9]],
    3: [[4.2e-2, 1.2e-3, 5.2e-3, 1.7e-4, 3.4e-4],
        [4.7e-2, 1.2e-3, 5.2e-4, 1.4e-5, 8.5e-7],
        [4.6e-2, 1.2e-3, 5.2e-4, 1.4e-5, 8.5e-7],
        [4.7e-2, 1.2e-3, 5.2e-4, 1.4e-5, 8.5e-7]],
    4: [[1.1e-1, 1.5e-2, 1.7e-3, 1.8e-4, 5.7e-5],
        [1.4e-1, 6.2e-3, 1.3e-3, 1.8e-4, 3.6e-5],
        [2.7e-1, 3.0e-2, 1.3e-3, 1.8e-4, 3.6e-5],
        [1.4e-1, 2.1e-2, 1.3e-3, 1.8e-4, 3.6e-5]],
    5: [[5.5e-3, 9.6e-4, 2.1e-4, 5.1e-5, 1.2e-5],
        [5.8e-4, 1.3e-4, 8.0e-6, 4.6e-7, 2.7e-8],
        [2.3e-3, 1.1e-4, 8.2e-7, 1.0e-7, 6.6e-9],
        [9.5e-4, 1.3e-4, 1.0e-6, 4.4e-8, 2.7e-9]],
}


def divided_differences(x, y):
    values = np.array(y).copy()
    for order in range(1, len(x)):
        values[order:] = ((values[order:] - values[order - 1:-1])
                          / (x[order:] - x[:-order]))
    return values


def jump(x, y):
    d = divided_differences(x[:6], y[:6])
    rho, fifth = d[4], d[5]
    if rho * fifth > 0:
        rho = np.sign(rho) * max(0, abs(rho) - 5 * abs(fifth) * (x[2] - x[1]))
    if fifth != 0:
        rho *= min(1, abs(d[4]) / (5 * abs(fifth) * (x[4] - x[0])))
    return 12 * rho * (x[2] - x[0])


def spline(x, y, kind):
    if kind == "NAT":
        return CubicSpline(x, y, bc_type="natural")
    if kind == "NAK":
        return CubicSpline(x, y)
    if kind == "Q":
        # Independent power-basis fit, not the production divided-difference formula.
        left = np.polynomial.Polynomial.fit(x[:5] - x[0], y[:5], 4).convert()
        right = np.polynomial.Polynomial.fit(x[-5:] - x[-1], y[-5:], 4).convert()
        return CubicSpline(x, y, bc_type=((2, left.deriv(2)(0)), (2, right.deriv(2)(0))))
    n, h = len(x), np.diff(x)
    matrix, rhs = np.zeros((n, n)), np.zeros(n)
    for i in range(1, n - 1):
        matrix[i, i-1:i+2] = [h[i-1], 2 * (h[i-1] + h[i]), h[i]]
        rhs[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    matrix[0, :3] = [1/h[0], -1/h[0]-1/h[1], 1/h[1]]
    matrix[-1, -3:] = [1/h[-2], -1/h[-2]-1/h[-1], 1/h[-1]]
    rhs[0] = jump(x, y)
    rhs[-1] = jump(-x[::-1], y[::-1])
    curvature = np.linalg.solve(matrix, rhs)
    return CubicSpline(x, y, bc_type=((2, curvature[0]), (2, curvature[-1])))


def maximum(x, function, derivative, interpolator, panels):
    candidates = list(x)
    error_derivative = lambda t: interpolator(t, 1) - derivative(t)
    for lo, hi in zip(x[:-1], x[1:]):
        grid = np.linspace(lo, hi, panels + 1)
        for left, right in zip(grid[:-1], grid[1:]):
            if error_derivative(left) * error_derivative(right) < 0:
                candidates.append(brentq(error_derivative, left, right, xtol=5e-15))
    location = max(candidates, key=lambda t: abs(interpolator(t) - function(t)))
    return float(abs(interpolator(location) - function(location))), float(location)


def random_mesh(n, left, right, seed):
    state, raw = seed, []
    for _ in range(n):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        raw.append(state / 4294967296.0)
    values = np.sort(raw)
    return left + (right - left) * (values - values[0]) / (values[-1] - values[0])


def generate():
    tables = []
    logistic = lambda x: 1 / (1 + np.exp(-x))
    for table in range(1, 6):
        if table == 1:
            function, derivative, left, right = np.sin, np.cos, 0, np.pi
        elif table in (2, 3):
            function, derivative, left, right = np.sin, np.cos, np.pi/4, 5*np.pi/4
        elif table == 4:
            function = lambda x: 1 / (1 + x*x)
            derivative = lambda x: -2*x / (1 + x*x)**2
            left, right = -1, 3
        else:
            function, derivative = logistic, lambda x: logistic(x) * (1 - logistic(x))
            left, right = -1, 4
        random = table in (3, 4)
        cases = []
        for index, count in enumerate(COUNTS):
            x = (random_mesh(count, left, right, 250705083 + table) if random
                 else np.linspace(left, right, count))
            rows = []
            for method, kind in enumerate(KINDS):
                interpolator = spline(x, function(x), kind)
                error, location = maximum(x, function, derivative, interpolator, 32)
                refined, _ = maximum(x, function, derivative, interpolator, 64)
                assert abs(error - refined) <= 5e-13, (table, count, kind)
                printed = PRINTED[table][method][index]
                rounding = 0.5 * 10 ** (np.floor(np.log10(printed)) - 1)
                rows.append({"boundary": kind, "max_error": error, "arg_max": location,
                             "printed_error": printed, "printed_half_unit": float(rounding),
                             "matches_printed": None if random else bool(abs(error - printed) <= rounding)})
            cases.append({"knots": x.tolist(), "values": function(x).tolist(), "results": rows})
        tables.append({"table": table, "mesh": "independent_reconstruction" if random else "published_uniform",
                       "seed": 250705083 + table if random else None, "cases": cases})
    output = {"source": "https://arxiv.org/abs/2507.05083v1", "numpy": np.__version__,
              "scipy": scipy.__version__, "tables": tables}
    destination = VECTORS_DIR / "jarre_splines.json"
    save_vector("jarre_splines", output)
    for table in tables:
        for case in table["cases"]:
            for row in case["results"]:
                if row["matches_printed"] is False:
                    print(f'Table {table["table"]} {row["boundary"]} n={len(case["knots"])}: '
                          f'printed={row["printed_error"]:.9g}, literal source={row["max_error"]:.12g}')
    print(f"Wrote {destination}")


if __name__ == "__main__":
    generate()
