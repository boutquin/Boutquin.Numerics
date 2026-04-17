// Copyright (c) 2026 Pierre G. Boutquin. All rights reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License").
//   You may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//
//   See the License for the specific language governing permissions and
//   limitations under the License.
//

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Classifies interpolation schemes by mathematical behavior and continuity class,
/// serving as the discriminator for <see cref="InterpolatorFactory"/>. Stateless
/// singletons are reachable via the factory; stateful interpolators
/// (<see cref="CubicSplineInterpolator"/>, <see cref="MonotoneCubicSpline"/>) must
/// be constructed directly with data.
/// </summary>
public enum InterpolatorKind
{
    /// <summary>
    /// Piecewise linear, C⁰ continuity — maps to <see cref="LinearInterpolator"/>.
    /// O(h²) pointwise error on smooth data; no derivative continuity at nodes.
    /// Flat extrapolation beyond the node range.
    /// </summary>
    Linear = 0,

    /// <summary>
    /// Linear interpolation in log space — maps to <see cref="LogLinearInterpolator"/>.
    /// Preserves strict positivity and produces constant instantaneous forward rates
    /// between adjacent nodes when applied to discount factors. All y-values must be
    /// strictly positive.
    /// </summary>
    LogLinear = 1,

    /// <summary>
    /// Constant forward rate between nodes — maps to <see cref="FlatForwardInterpolator"/>.
    /// Mathematically equivalent to <see cref="LogLinear"/> under continuous compounding,
    /// but expressed directly in rate form so the formula remains meaningful when
    /// extrapolating or composing with other rate curves.
    /// </summary>
    FlatForward = 2,

    /// <summary>
    /// Fritsch-Carlson monotone-preserving cubic Hermite interpolation — maps to
    /// <see cref="MonotoneCubicInterpolator"/>. Guarantees monotonicity preservation
    /// on each segment (no spurious oscillation between data points). Reference:
    /// Fritsch &amp; Carlson 1980 (SIAM J. Numer. Anal.); boundary tangent refinement
    /// per arXiv:2402.01324. Stateless; tangents are recomputed per call.
    /// </summary>
    MonotoneCubic = 3,

    /// <summary>
    /// Natural cubic spline with C² continuity and second-derivative zero at the
    /// boundary (<c>S''(x₀) = S''(xₙ) = 0</c>) — maps to the natural-boundary case
    /// of <see cref="CubicSplineInterpolator"/> when the factory is used. For other
    /// boundary conditions (clamped, not-a-knot, modified not-a-knot, Q-spline) or
    /// when repeated evaluations on the same node set are needed, construct
    /// <see cref="CubicSplineInterpolator"/> directly (it is stateful and precomputes
    /// coefficients).
    /// </summary>
    CubicSpline = 4,

    /// <summary>
    /// Hagan-West (2006) monotone-convex interpolation — maps to <see cref="MonotoneConvexInterpolator"/>.
    /// Guarantees non-negative instantaneous forward rates everywhere when applied to normalized
    /// cumulative return (NCR) node arrays derived from discount factors. Preferred for
    /// arbitrage-free curve construction where forward-rate shape discipline matters.
    /// Reference: Hagan &amp; West (2006), Applied Mathematical Finance 13(2), pp. 89–129.
    /// </summary>
    MonotoneConvex = 5,
}
