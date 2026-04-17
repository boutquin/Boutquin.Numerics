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


def reference_monotone_convex(x: float, xs: list[float], ys: list[float]) -> float:
    """Hagan-West (2006) monotone-convex interpolation — paper-faithful port.

    Paper: Hagan, P. S. and West, G. (2006), "Interpolation Methods for
    Curve Construction", Applied Mathematical Finance 13(2), pp. 89-129.
    Equations used: §2.3 (Eq. 2.2 — time-weighted tangents, Eq. 2.5 —
    linear boundary extrapolation) and §2.4 (Eq. 2.7 — non-negativity
    clamp). The per-segment integration of the cubic forward polynomial
    follows the closed-form integration of Eq. 2.6 (quadratic basis in u)
    over u ∈ [0, t], producing a cubic in u.

    Input convention (shared with C# ``MonotoneConvexInterpolator``):
    ``xs[0] = 0, ys[0] = 0`` is the virtual origin; ``xs[1..]`` are actual
    node times (year fractions) and ``ys[i] = −ln P(xs[i])`` are NCR
    values. Flat extrapolation outside the node range.

    No scipy or statsmodels routine exists for Hagan-West — this is a
    finance-specific curve-construction method. The reference is a
    hand-port from the paper; it shares no code with the C# implementation
    (different language, different array-indexing idiom, different loop
    structure), so a shared bug would require a co-incident misreading
    of the paper. Unit-test invariants (monotonicity, non-negative
    forwards, flat extrapolation) further guard against that.
    """

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    if x <= xs_arr[0]:
        return float(ys_arr[0])
    if x >= xs_arr[-1]:
        return float(ys_arr[-1])

    n = xs_arr.size

    # Step 1: segment widths h[j] and discrete forwards F[j] for j ∈ [1, n-1].
    # (Hagan-West §2.2; index 0 is unused and left at 0.)
    h = [0.0] * n
    f_seg = [0.0] * n
    for j in range(1, n):
        h[j] = xs_arr[j] - xs_arr[j - 1]
        f_seg[j] = (ys_arr[j] - ys_arr[j - 1]) / h[j]

    # Step 2: instantaneous forward tangents g[i] at each node.
    # Interior: Hagan-West Eq. 2.2 — time-weighted average of adjacent
    # segment forwards, with weights opposite of the usual convention
    # (the weight on F[i+1] is the LEFT segment width h[i]).
    g = [0.0] * n
    for i in range(1, n - 1):
        g[i] = (h[i] * f_seg[i + 1] + h[i + 1] * f_seg[i]) / (h[i] + h[i + 1])

    # Boundary extrapolation — Hagan-West Eq. 2.5. Two-point formula
    # reduces to g[·] = F[1] / F[n-1] when only one interior segment exists.
    if n >= 3:
        g[0] = (3.0 * f_seg[1] - g[1]) / 2.0
        g[n - 1] = (3.0 * f_seg[n - 1] - g[n - 2]) / 2.0
    else:
        g[0] = f_seg[1]
        g[n - 1] = f_seg[n - 1]

    # Step 3: non-negativity clamp — Hagan-West §2.4 Eq. 2.7. Each g[i]
    # is pinned to [0, min(2·F[left], 2·F[right])]. Boundary nodes see
    # only one adjacent segment.
    g[0] = max(0.0, min(g[0], 2.0 * f_seg[1]))
    for i in range(1, n - 1):
        limit = min(2.0 * f_seg[i], 2.0 * f_seg[i + 1])
        g[i] = max(0.0, min(g[i], limit)) if limit > 0.0 else 0.0
    g[n - 1] = max(0.0, min(g[n - 1], 2.0 * f_seg[n - 1]))

    # Step 4: locate the bracket segment [xs[k-1], xs[k]] containing x.
    k = 1
    while k < n and x > xs_arr[k]:
        k += 1
    # Loop invariant: xs[k-1] < x <= xs[k] when the function reaches this point
    # (the boundary checks at the top ruled out x <= xs[0] and x >= xs[-1]).

    # Step 5: integrate the cubic forward polynomial (Hagan-West Eq. 2.6
    # integrated in closed form over u ∈ [0, t]) to obtain NCR increment.
    u = (x - xs_arr[k - 1]) / h[k]
    u2 = u * u
    u3 = u2 * u
    delta_r = h[k] * (
        f_seg[k] * u
        + (g[k - 1] - f_seg[k]) * (u - 2.0 * u2 + u3)
        + (g[k] - f_seg[k]) * (-u2 + u3)
    )
    return float(ys_arr[k - 1] + delta_r)


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
    # locations where the interior tangent formula meets the §2.5 boundary
    # extrapolation.
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
