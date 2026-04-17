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

using System.Numerics;

/// <summary>
/// Specifies boundary conditions for cubic spline construction.
/// </summary>
/// <remarks>
/// Five boundary types are supported:
/// <list type="bullet">
/// <item><description><see cref="Natural"/>: S''(x0) = S''(xn) = 0 — the "free" boundary; suitable when endpoint derivatives are unknown (Faires &amp; Burden, §3.5).</description></item>
/// <item><description><see cref="CreateClamped"/>: S'(x0) = f'(x0), S'(xn) = f'(xn) — produces more accurate approximations when endpoint derivatives are available (Faires &amp; Burden, §3.5).</description></item>
/// <item><description><see cref="NotAKnot"/>: Forces the third derivative to be continuous at the second and penultimate nodes (Faires &amp; Burden, §3.5).</description></item>
/// <item><description><see cref="ModifiedNotAKnot"/>: Improved not-a-knot with better conditioning on non-uniform grids (Jarre 2025, arXiv:2507.05083).</description></item>
/// <item><description><see cref="QSpline"/>: Fourth-order optimal error bounds using only function values — no derivatives required (Jarre 2025, arXiv:2507.05083).</description></item>
/// </list>
/// Legacy facade: for backward-compatible <c>double</c> consumers. New code should
/// prefer <see cref="SplineBoundary{T}"/>.
/// </remarks>
public abstract class SplineBoundary
{
    private SplineBoundary() { }

    /// <summary>
    /// Natural boundary condition: zero second derivative at both endpoints
    /// <c>S''(x₀) = S''(xₙ) = 0</c>. Appropriate when endpoint derivatives are
    /// unknown; simple and stable but introduces O(h²) error near the boundary
    /// (Faires &amp; Burden, §3.5).
    /// </summary>
    public static SplineBoundary Natural { get; } = new NaturalBoundary();

    /// <summary>
    /// Classical not-a-knot boundary condition (de Boor 1978, "A Practical Guide
    /// to Splines"): enforces continuity of the third derivative across the first
    /// and last interior knots, so the first two and last two cubic pieces share
    /// the same polynomial. Preferred on uniform or near-uniform grids.
    /// Requires at least 4 data points.
    /// </summary>
    public static SplineBoundary NotAKnot { get; } = new NotAKnotBoundary();

    /// <summary>
    /// Modified not-a-knot boundary with improved conditioning on non-uniform grids.
    /// Uses spacing-ratio-weighted conditions instead of pure third-derivative
    /// continuity, achieving 4th-order accuracy on uniform grids while remaining
    /// well-conditioned on highly non-uniform grids.
    /// Requires at least 4 data points.
    /// </summary>
    /// <remarks>Based on Jarre (2025), arXiv:2507.05083, §3.</remarks>
    public static SplineBoundary ModifiedNotAKnot { get; } = new ModifiedNotAKnotBoundary();

    /// <summary>
    /// Q-spline boundary: achieves 4th-order asymptotically optimal error bounds
    /// using only function values (no derivative information required).
    /// Requires at least 4 data points.
    /// </summary>
    /// <remarks>Based on Jarre (2025), arXiv:2507.05083, §4.</remarks>
    public static SplineBoundary QSpline { get; } = new QSplineBoundary();

    /// <summary>
    /// Clamped boundary: S'(x0) = <paramref name="leftSlope"/>, S'(xn) = <paramref name="rightSlope"/>.
    /// </summary>
    /// <param name="leftSlope">Prescribed first derivative at the left endpoint.</param>
    /// <param name="rightSlope">Prescribed first derivative at the right endpoint.</param>
    /// <returns>A clamped boundary condition instance.</returns>
    public static SplineBoundary CreateClamped(double leftSlope, double rightSlope)
        => new ClampedBoundary(leftSlope, rightSlope);

    internal sealed class NaturalBoundary : SplineBoundary;

    internal sealed class NotAKnotBoundary : SplineBoundary;

    internal sealed class ModifiedNotAKnotBoundary : SplineBoundary;

    internal sealed class QSplineBoundary : SplineBoundary;

    internal sealed class ClampedBoundary(double leftSlope, double rightSlope) : SplineBoundary
    {
        internal double LeftSlope { get; } = leftSlope;
        internal double RightSlope { get; } = rightSlope;
    }
}

/// <summary>
/// Specifies boundary conditions for cubic spline construction with generic type support.
/// </summary>
/// <typeparam name="T">The floating-point type for slope values in clamped boundaries.</typeparam>
/// <remarks>
/// Five boundary types are supported:
/// <list type="bullet">
/// <item><description><see cref="Natural"/>: S''(x0) = S''(xn) = 0.</description></item>
/// <item><description><see cref="CreateClamped"/>: S'(x0) = leftSlope, S'(xn) = rightSlope.</description></item>
/// <item><description><see cref="NotAKnot"/>: Third derivative continuity at first and last interior knots.</description></item>
/// <item><description><see cref="ModifiedNotAKnot"/>: Improved not-a-knot with better conditioning on non-uniform grids.</description></item>
/// <item><description><see cref="QSpline"/>: Fourth-order optimal error bounds using only function values.</description></item>
/// </list>
/// </remarks>
public abstract class SplineBoundary<T> where T : IFloatingPoint<T>
{
    private SplineBoundary() { }

    /// <inheritdoc cref="SplineBoundary.Natural"/>
    public static SplineBoundary<T> Natural { get; } = new NaturalBoundary();

    /// <inheritdoc cref="SplineBoundary.NotAKnot"/>
    public static SplineBoundary<T> NotAKnot { get; } = new NotAKnotBoundary();

    /// <inheritdoc cref="SplineBoundary.ModifiedNotAKnot"/>
    public static SplineBoundary<T> ModifiedNotAKnot { get; } = new ModifiedNotAKnotBoundary();

    /// <inheritdoc cref="SplineBoundary.QSpline"/>
    public static SplineBoundary<T> QSpline { get; } = new QSplineBoundary();

    /// <summary>
    /// Clamped boundary: S'(x0) = <paramref name="leftSlope"/>, S'(xn) = <paramref name="rightSlope"/>.
    /// </summary>
    /// <param name="leftSlope">Prescribed first derivative at the left endpoint.</param>
    /// <param name="rightSlope">Prescribed first derivative at the right endpoint.</param>
    /// <returns>A clamped boundary condition instance.</returns>
    public static SplineBoundary<T> CreateClamped(T leftSlope, T rightSlope)
        => new ClampedBoundary(leftSlope, rightSlope);

    internal sealed class NaturalBoundary : SplineBoundary<T>;

    internal sealed class NotAKnotBoundary : SplineBoundary<T>;

    internal sealed class ModifiedNotAKnotBoundary : SplineBoundary<T>;

    internal sealed class QSplineBoundary : SplineBoundary<T>;

    internal sealed class ClampedBoundary(T leftSlope, T rightSlope) : SplineBoundary<T>
    {
        internal T LeftSlope { get; } = leftSlope;
        internal T RightSlope { get; } = rightSlope;
    }
}
