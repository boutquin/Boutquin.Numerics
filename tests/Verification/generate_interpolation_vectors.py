"""Interpolation reference vectors verified against scipy.interpolate.

Covers three legacy interpolators (Linear, PCHIP/MonotoneCubic, Natural
cubic spline) unchanged from the prior ship, four interpolator additions
cross-checked against numpy / scipy hand-ports, and one finance-specific
interpolator cross-checked against a paper-faithful Python port:

* ``log_linear`` — ``numpy.interp(log x, log y)`` reference (see the
  reference function below for the exact form the C# ``LogLinearInterpolator``
  implements).
* ``flat_forward`` — step-forward rate reference equivalent to LogLinear
  under continuous compounding; separately verified so consumers who pick
  the rate-form class directly still have a cross-check.
* ``two_point_linear`` — manual two-point linear reference; a sanity
  check that ``TwoPointLinearInterpolator.Interpolate`` matches the
  textbook formula at a handful of query points.
* ``monotone_cubic_n2`` — the N=2 degenerate case of the monotone cubic
  spline. ``MonotoneCubicSpline`` is required to accept N ≥ 2 and
  degenerate to linear interpolation on the two-point case; the reference
  is ``scipy.interpolate.PchipInterpolator`` on the same two-point input,
  which degenerates to linear in the interior too.
* ``monotone_convex`` — Hagan-West (2006) monotone-convex interpolation
  for discount-factor curves. No library equivalent exists for this
  finance-specific algorithm, so the reference is a paper-faithful
  hand-port (see ``reference_monotone_convex`` for the equation citations).

Library pins:
* numpy==2.x
* scipy==1.13+
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate

from conftest import save_vector


# =====================================================================
#  Phase 4 references — paper-faithful numpy ports of the C# algorithms.
# =====================================================================


def reference_log_linear(x: float, xs: list[float], ys: list[float]) -> float:
    """Log-linear interpolation: linear in ``log(y)`` against ``x``.

    Matches the C# ``LogLinearInterpolator.Interpolate`` formula:
      * Boundary: flat-extrapolate to ``ys[0]`` / ``ys[-1]``.
      * Interior: ``y(x) = exp(ln(ys[i]) + t·(ln(ys[i+1]) − ln(ys[i])))``
        with ``t = (x − xs[i]) / (xs[i+1] − xs[i])``.

    Precondition: all ``ys > 0``.
    """

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if x <= xs_arr[0]:
        return float(ys_arr[0])
    if x >= xs_arr[-1]:
        return float(ys_arr[-1])
    # Search for the interval.
    i = int(np.searchsorted(xs_arr, x, side="right") - 1)
    i = max(0, min(i, xs_arr.size - 2))
    ln_y_i = np.log(ys_arr[i])
    ln_y_i1 = np.log(ys_arr[i + 1])
    t = (x - xs_arr[i]) / (xs_arr[i + 1] - xs_arr[i])
    return float(np.exp(ln_y_i + t * (ln_y_i1 - ln_y_i)))


def reference_flat_forward(x: float, xs: list[float], ys: list[float]) -> float:
    """Flat-forward interpolation on discount factors.

    Matches the C# ``FlatForwardInterpolator.Interpolate`` formula:
      * Boundary: flat-extrapolate.
      * Interior: ``DF(x) = DF_i · exp(−f · (x − t_i))`` with
        ``f = (ln(DF_i) − ln(DF_{i+1})) / (t_{i+1} − t_i)``.

    Mathematically equivalent to ``reference_log_linear`` under continuous
    compounding (``DF_i`` is the log-linear interpolant of ``DF``), but
    expressed in rate form so downstream consumers who read the code
    instead of testing can confirm the convention matches Numerics'.
    """

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if x <= xs_arr[0]:
        return float(ys_arr[0])
    if x >= xs_arr[-1]:
        return float(ys_arr[-1])
    i = int(np.searchsorted(xs_arr, x, side="right") - 1)
    i = max(0, min(i, xs_arr.size - 2))
    dx = xs_arr[i + 1] - xs_arr[i]
    f = (np.log(ys_arr[i]) - np.log(ys_arr[i + 1])) / dx
    return float(ys_arr[i] * np.exp(-f * (x - xs_arr[i])))


def reference_two_point_linear(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    """Verbatim port of ``TwoPointLinearInterpolator.Interpolate``.

    ``y(x) = y0 + (x − x0) / (x1 − x0) · (y1 − y0)`` — the textbook linear
    interpolation formula, no safety rails against extrapolation.
    """

    return float(y0 + (x - x0) / (x1 - x0) * (y1 - y0))


def _hagan_west_deviation(g_left: float, g_right: float, x: float) -> float:
    """Pointwise forward deviation g(x) = f(t) - F_i on one interval, x in [0, 1].

    Hagan & West (2006), "Enforcing Monotonicity and Convexity": the basic
    quadratic of eq. 47 in sector (i), the flat-then-quadratic of eqs. 49-50
    in sector (ii), the quadratic-then-flat of eqs. 51-52 in sector (iii), and
    the two quadratics of eqs. 53-56 in sector (iv). The sector boundaries are
    read from the paper's list of regions (i)-(iv).
    """

    a, b = g_left, g_right
    if a == 0.0 and b == 0.0:
        return 0.0  # footnote 7: flat curve
    if (a > 0 and -2 * a <= b <= -a / 2) or (a < 0 and -a / 2 <= b <= -2 * a):
        # sector (i), eq. 47
        return a * (1 - 4 * x + 3 * x * x) + b * (-2 * x + 3 * x * x)
    if (a < 0 and b > -2 * a) or (a > 0 and b < -2 * a):
        # sector (ii), eqs. 49-50
        eta = (b + 2 * a) / (b - a)
        return a if x <= eta else a + (b - a) * ((x - eta) / (1 - eta)) ** 2
    if (a > 0 and -a / 2 < b < 0) or (a < 0 and 0 < b < -a / 2):
        # sector (iii), eqs. 51-52
        eta = 3 * b / (b - a)
        return b + (a - b) * ((eta - x) / eta) ** 2 if x < eta else b
    # sector (iv), eqs. 53-56
    eta = b / (a + b)
    big_a = -a * b / (a + b)
    if x < eta:
        return big_a + (a - big_a) * ((eta - x) / eta) ** 2
    if x == eta:
        return big_a
    return big_a + (b - big_a) * ((x - eta) / (1 - eta)) ** 2


def reference_monotone_convex(x: float, xs: list[float], ys: list[float]) -> float:
    """Hagan-West (2006) monotone-convex interpolation, written from the paper.

    Paper: Hagan, P. S. and West, G. (2006), "Interpolation Methods for
    Curve Construction", Applied Mathematical Finance 13(2), pp. 89-129. The
    paper has no numbered sections, so the citations are equation numbers:
    node forwards eq. 30 (interior) and eqs. 31-32 (ends); positivity bounds
    eqs. 60-62; the monotonicity/convexity sectors eqs. 45-56; the NCR as the
    integral of the forward, eq. 37 and eq. 81. The amelioration step
    (eqs. 63-80) is not applied, matching the C# class.

    Input convention (shared with C# ``MonotoneConvexInterpolator``):
    ``xs[0] = 0, ys[0] = 0`` is the virtual origin; ``xs[1..]`` are actual
    node times (year fractions) and ``ys[i] = -ln P(xs[i])`` are NCR
    values. Flat extrapolation outside the node range.

    Independence from the C# code: the forward deviation is evaluated
    pointwise from the paper's piecewise definitions and integrated
    numerically with ``scipy.integrate.quad`` (split at the sector
    breakpoint), whereas the C# class uses closed-form integrals. A shared
    error would need the same misreading of the paper's sector table.
    """

    from scipy import integrate

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    if x <= xs_arr[0]:
        return float(ys_arr[0])
    if x >= xs_arr[-1]:
        return float(ys_arr[-1])

    n = xs_arr.size - 1  # t_0 = 0, ..., t_n

    # Discrete forwards f_i^d on [t_{i-1}, t_i], i = 1..n.
    t = xs_arr
    fd = {i: (ys_arr[i] - ys_arr[i - 1]) / (t[i] - t[i - 1]) for i in range(1, n + 1)}

    # Node forwards: eq. 30 (interior), eqs. 31-32 (ends).
    f = {}
    for i in range(1, n):
        f[i] = ((t[i] - t[i - 1]) / (t[i + 1] - t[i - 1])) * fd[i + 1] + (
            (t[i + 1] - t[i]) / (t[i + 1] - t[i - 1])
        ) * fd[i]
    if n == 1:
        f[0] = f[1] = fd[1]
    else:
        f[0] = fd[1] - 0.5 * (f[1] - fd[1])
        f[n] = fd[n] - 0.5 * (f[n - 1] - fd[n])

    # Positivity bounds, eqs. 60-62, bound(a, x, b) = min(max(a, x), b).
    def bound(lo: float, v: float, hi: float) -> float:
        return min(max(lo, v), hi)

    f[0] = bound(0.0, f[0], 2 * fd[1])
    for i in range(1, n):
        f[i] = bound(0.0, f[i], 2 * min(fd[i], fd[i + 1]))
    f[n] = bound(0.0, f[n], 2 * fd[n])

    # Interval i with t_{i-1} < x <= t_i.
    i = int(np.searchsorted(t, x, side="left"))
    width = t[i] - t[i - 1]
    u = (x - t[i - 1]) / width
    g_left = f[i - 1] - fd[i]
    g_right = f[i] - fd[i]

    # Breakpoint of the piecewise definition, so quad integrates smooth pieces.
    points = []
    a, b = g_left, g_right
    if (a < 0 and b > -2 * a) or (a > 0 and b < -2 * a):
        points = [(b + 2 * a) / (b - a)]
    elif (a > 0 and -a / 2 < b < 0) or (a < 0 and 0 < b < -a / 2):
        points = [3 * b / (b - a)]
    elif not (a == 0.0 and b == 0.0) and not (
        (a > 0 and -2 * a <= b <= -a / 2) or (a < 0 and -a / 2 <= b <= -2 * a)
    ):
        points = [b / (a + b)]
    points = [p for p in points if 0.0 < p < u]

    integral, _ = integrate.quad(
        lambda s: _hagan_west_deviation(g_left, g_right, s),
        0.0,
        u,
        points=points or None,
        epsabs=1e-14,
        epsrel=1e-12,
        limit=200,
    )

    # eq. 81: r(t)t = r_{i-1}t_{i-1} + f_i^d (t - t_{i-1}) + integral of g, and eq. 37
    # rescales the integral from x units to time units.
    return float(ys_arr[i - 1] + fd[i] * (x - t[i - 1]) + width * integral)


# =====================================================================
#  Generation.
# =====================================================================


def generate() -> None:
    # --- Legacy block (unchanged semantics) --------------------------
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [0.0, 0.841, 0.909, 0.141, -0.757, -0.959]
    queries = [0.5, 1.5, 2.5, 3.5, 4.5]

    linear = interpolate.interp1d(xs, ys, kind="linear")
    linear_queries = [float(linear(q)) for q in queries]

    pchip = interpolate.PchipInterpolator(xs, ys)
    pchip_queries = [float(pchip(q)) for q in queries]

    natural = interpolate.CubicSpline(xs, ys, bc_type="natural")
    natural_queries = [float(natural(q)) for q in queries]

    # --- Phase 4 additions ------------------------------------------
    #
    # Positive-ys dataset for LogLinear / FlatForward (both require ys > 0).
    # Pattern: discount-factor-like — monotonically decreasing starting at
    # 1.0 with realistic short-rate curvature.
    pos_xs = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    pos_ys = [1.0, 0.9875, 0.975, 0.95, 0.90, 0.78, 0.60]
    pos_queries = [0.1, 0.375, 0.75, 1.5, 3.0, 7.0]

    log_linear_values = [
        reference_log_linear(q, pos_xs, pos_ys) for q in pos_queries
    ]
    flat_forward_values = [
        reference_flat_forward(q, pos_xs, pos_ys) for q in pos_queries
    ]

    # TwoPointLinearInterpolator — inputs/outputs for five query points.
    two_point_x0, two_point_y0 = 1.0, 10.0
    two_point_x1, two_point_y1 = 4.0, 25.0
    two_point_queries = [0.0, 1.0, 2.5, 4.0, 5.0]  # includes both ends + ext
    two_point_values = [
        reference_two_point_linear(
            two_point_x0, two_point_y0, two_point_x1, two_point_y1, q
        )
        for q in two_point_queries
    ]

    # MonotoneCubicSpline with N=2 — scipy PCHIP degenerates to linear in
    # the interior for a two-point input. The C# class is required to
    # accept N = 2 (per its docstring it needs xs.Count >= 2) and produce
    # the same linear-in-the-interior behaviour.
    n2_xs = [0.0, 1.0]
    n2_ys = [2.0, 5.0]
    n2_queries = [0.0, 0.25, 0.5, 0.75, 1.0]
    n2_pchip = interpolate.PchipInterpolator(n2_xs, n2_ys)
    monotone_cubic_n2_values = [float(n2_pchip(q)) for q in n2_queries]

    # MonotoneConvexInterpolator — Hagan-West (2006). Input is a yield-curve
    # shaped NCR node array: xs[0]=0, ys[0]=0 is the virtual origin; the
    # remaining nodes encode NCR = −ln(DF) for an upward-sloping curve.
    # Query points exercise three distinct segments plus two near-boundary
    # locations where the interior node forwards (eq. 30) meet the boundary
    # forwards (eqs. 31-32).
    import math
    mc_discounts = [1.0, 0.995, 0.97, 0.83, 0.68, 0.25]
    mc_xs = [0.0, 0.25, 1.0, 5.0, 10.0, 30.0]
    mc_ys = [0.0 if df == 1.0 else -math.log(df) for df in mc_discounts]
    mc_queries = [0.1, 0.5, 3.0, 7.5, 20.0]
    monotone_convex_values = [
        reference_monotone_convex(q, mc_xs, mc_ys) for q in mc_queries
    ]

    save_vector(
        "interpolation",
        {
            "library_pins": {"numpy": "2", "scipy": "1.13+"},
            # Legacy keys — unchanged.
            "xs": xs,
            "ys": ys,
            "queries": queries,
            "linear": linear_queries,
            "pchip": pchip_queries,
            "natural_cubic_spline": natural_queries,
            # Phase 4 additions.
            "log_linear": {
                "xs": pos_xs,
                "ys": pos_ys,
                "queries": pos_queries,
                "values": log_linear_values,
            },
            "flat_forward": {
                "xs": pos_xs,
                "ys": pos_ys,
                "queries": pos_queries,
                "values": flat_forward_values,
            },
            "two_point_linear": {
                "x0": two_point_x0,
                "y0": two_point_y0,
                "x1": two_point_x1,
                "y1": two_point_y1,
                "queries": two_point_queries,
                "values": two_point_values,
            },
            "monotone_cubic_n2": {
                "xs": n2_xs,
                "ys": n2_ys,
                "queries": n2_queries,
                "values": monotone_cubic_n2_values,
            },
            "monotone_convex": {
                "xs": mc_xs,
                "ys": mc_ys,
                "queries": mc_queries,
                "values": monotone_convex_values,
            },
        },
    )


if __name__ == "__main__":
    generate()
    print("Wrote interpolation.json")
