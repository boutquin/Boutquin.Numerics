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
/// Six boundary types are supported:
/// <list type="bullet">
/// <item><description><see cref="Natural"/>: S''(x0) = S''(xn) = 0 — the "free" boundary; suitable when endpoint derivatives are unknown (Faires &amp; Burden, §3.5).</description></item>
/// <item><description><see cref="CreateClamped"/>: S'(x0) = f'(x0), S'(xn) = f'(xn) — produces more accurate approximations when endpoint derivatives are available (Faires &amp; Burden, §3.5).</description></item>
/// <item><description><see cref="NotAKnot"/>: Forces the third derivative to be continuous at the second and penultimate nodes (Faires &amp; Burden, §3.5).</description></item>
/// <item><description><see cref="ModifiedNotAKnot"/>: Not-a-knot endpoint curvature scaled by the end-interval spacing ratio; a library rule, not the revised not-a-knot spline of Jarre (2025), arXiv:2507.05083, §4.</description></item>
/// <item><description><see cref="RevisedNotAKnot"/>: Safeguarded third-derivative jumps from Jarre, JCAM 478 (2026), Appendix; author code dated October 29, 2025.</description></item>
/// <item><description><see cref="QSpline"/>: Six-point safeguarded endpoint-curvature estimates from Jarre, JCAM 478 (2026), Appendix.</description></item>
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
    /// Modified not-a-knot boundary. Scales the not-a-knot endpoint curvature by the
    /// spacing ratio <c>min(h₀, h₁) / max(h₀, h₁)</c> of the two end intervals
    /// (symmetrically at the right end), so it coincides with <see cref="NotAKnot"/>
    /// on uniform grids and moves toward a natural end condition as the end spacing
    /// becomes uneven. On non-uniform grids its accuracy can fall to O(h²), below
    /// <see cref="NotAKnot"/>.
    /// Requires at least 4 data points.
    /// </summary>
    /// <remarks>
    /// This is a library rule. It is not the revised not-a-knot (RNAK) spline of
    /// Jarre (2025), arXiv:2507.05083, §4, which prescribes a third-derivative jump
    /// at the first interior knot to reduce the interpolation error.
    /// </remarks>
    public static SplineBoundary ModifiedNotAKnot { get; } = new ModifiedNotAKnotBoundary();

    /// <summary>
    /// Revised not-a-knot (RNAK) boundary from Jarre, JCAM 478 (2026), Appendix.
    /// Prescribes third-derivative jumps at the first and last interior knots using
    /// fourth divided differences and the fifth-difference safeguard with factor 5/2. Requires at least 6 data points.
    /// </summary>
    /// <remarks>
    /// The safeguard is heuristic; this boundary need not improve on classical
    /// not-a-knot for every data set. This follows the journal Appendix and authenticated
    /// author code dated October 29, 2025, superseding the July preprint safeguard.
    /// </remarks>
    public static SplineBoundary RevisedNotAKnot { get; } = new RevisedNotAKnotBoundary();

    /// <summary>
    /// Q-spline boundary: estimates endpoint second derivatives using a cubic
    /// interpolant with a fourth-difference correction, damped by the fifth
    /// difference when it indicates an overestimate. Requires at least 6 data points.
    /// </summary>
    /// <remarks>Based on Jarre, JCAM 478 (2026), Appendix Q steps 1–7 and author code dated October 29, 2025. Theorem 2 analyzes the undamped quartic endpoint estimate; the practical safeguard is heuristic.</remarks>
    public static SplineBoundary QSpline { get; } = new QSplineBoundary();

    /// <summary>
    /// Clamped boundary: S'(x0) = <paramref name="leftSlope"/>, S'(xn) = <paramref name="rightSlope"/>.
    /// </summary>
    /// <param name="leftSlope">Prescribed first derivative at the left endpoint.</param>
    /// <param name="rightSlope">Prescribed first derivative at the right endpoint.</param>
    /// <returns>A clamped boundary condition instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A prescribed slope is not finite.</exception>
    public static SplineBoundary CreateClamped(double leftSlope, double rightSlope)
    {
        if (!double.IsFinite(leftSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(leftSlope));
        }

        if (!double.IsFinite(rightSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(rightSlope));
        }

        return new ClampedBoundary(leftSlope, rightSlope);
    }

    internal sealed class NaturalBoundary : SplineBoundary;

    internal sealed class NotAKnotBoundary : SplineBoundary;

    internal sealed class ModifiedNotAKnotBoundary : SplineBoundary;

    internal sealed class RevisedNotAKnotBoundary : SplineBoundary;

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
/// Six boundary types are supported:
/// <list type="bullet">
/// <item><description><see cref="Natural"/>: S''(x0) = S''(xn) = 0.</description></item>
/// <item><description><see cref="CreateClamped"/>: S'(x0) = leftSlope, S'(xn) = rightSlope.</description></item>
/// <item><description><see cref="NotAKnot"/>: Third derivative continuity at first and last interior knots.</description></item>
/// <item><description><see cref="ModifiedNotAKnot"/>: Not-a-knot endpoint curvature scaled by the end-interval spacing ratio.</description></item>
/// <item><description><see cref="RevisedNotAKnot"/>: Safeguarded third-derivative jumps from Jarre, JCAM 478 (2026), Appendix; author code dated October 29, 2025.</description></item>
/// <item><description><see cref="QSpline"/>: Six-point safeguarded endpoint-curvature estimates using only function values.</description></item>
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

    /// <inheritdoc cref="SplineBoundary.RevisedNotAKnot"/>
    public static SplineBoundary<T> RevisedNotAKnot { get; } = new RevisedNotAKnotBoundary();

    /// <inheritdoc cref="SplineBoundary.QSpline"/>
    public static SplineBoundary<T> QSpline { get; } = new QSplineBoundary();

    /// <summary>
    /// Clamped boundary: S'(x0) = <paramref name="leftSlope"/>, S'(xn) = <paramref name="rightSlope"/>.
    /// </summary>
    /// <param name="leftSlope">Prescribed first derivative at the left endpoint.</param>
    /// <param name="rightSlope">Prescribed first derivative at the right endpoint.</param>
    /// <returns>A clamped boundary condition instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A prescribed slope is not finite.</exception>
    public static SplineBoundary<T> CreateClamped(T leftSlope, T rightSlope) =>
        CreateValidatedClamped(leftSlope, rightSlope, null);

    // A clamped boundary built from the non-generic double API: the working-type slopes are the caller's slopes
    // narrowed to T, and the double slopes ride along so the spline constructor can scale them to its normalized
    // units in double and narrow the scaled value once. A slope narrowed to T before that scaling can be erased
    // (or lose most of its digits) although the scaled slope, and every spline value it drives, is representable.
    internal static SplineBoundary<T> CreateClampedWithDoubleSlopes(T leftSlope, T rightSlope, double leftDouble, double rightDouble) =>
        CreateValidatedClamped(leftSlope, rightSlope, (leftDouble, rightDouble));

    // The non-generic API's entry: the slopes are validated as finite doubles only. The working-type slopes are a
    // saturating narrowing that may be infinite for a slope above T's range; they are placeholders until the
    // spline constructor scales the double slopes into its normalized units, and the constructor refuses the
    // boundary (ArgumentOutOfRangeException) if a slope is still unrepresentable once no further scaling applies.
    internal static SplineBoundary<T> CreateClampedFromDoubleSlopes(double leftSlope, double rightSlope)
    {
        if (!double.IsFinite(leftSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(leftSlope));
        }

        if (!double.IsFinite(rightSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(rightSlope));
        }

        return new ClampedBoundary(
            T.CreateSaturating(leftSlope), T.CreateSaturating(rightSlope), (leftSlope, rightSlope));
    }

    internal static void RequireRepresentableSlopes(T leftSlope, T rightSlope, (double Left, double Right)? doubleSlopes)
    {
        if (!T.IsFinite(leftSlope) || (doubleSlopes is { } left && !IsRepresentable(left.Left)))
        {
            throw new ArgumentOutOfRangeException(nameof(leftSlope));
        }

        if (!T.IsFinite(rightSlope) || (doubleSlopes is { } right && !IsRepresentable(right.Right)))
        {
            throw new ArgumentOutOfRangeException(nameof(rightSlope));
        }
    }

    // CreateChecked throws OverflowException for a slope beyond a type that has no infinity (decimal); the
    // IEEE types narrow it to an infinity instead, which the finiteness test refuses. Both are the same
    // condition (the slope is outside T's finite range), so the overflow is reported as "not representable"
    // and the caller raises the documented ArgumentOutOfRangeException at every T.
    private static bool IsRepresentable(double slope)
    {
        try
        {
            return T.IsFinite(T.CreateChecked(slope));
        }
        catch (OverflowException)
        {
            return false;
        }
    }

    private static SplineBoundary<T> CreateValidatedClamped(T leftSlope, T rightSlope, (double Left, double Right)? doubleSlopes)
    {
        if (!T.IsFinite(leftSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(leftSlope));
        }

        if (!T.IsFinite(rightSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(rightSlope));
        }

        return new ClampedBoundary(leftSlope, rightSlope, doubleSlopes);
    }

    internal sealed class NaturalBoundary : SplineBoundary<T>;

    internal sealed class NotAKnotBoundary : SplineBoundary<T>;

    internal sealed class ModifiedNotAKnotBoundary : SplineBoundary<T>;

    internal sealed class RevisedNotAKnotBoundary : SplineBoundary<T>;

    internal sealed class QSplineBoundary : SplineBoundary<T>;

    internal sealed class ClampedBoundary(T leftSlope, T rightSlope, (double Left, double Right)? doubleSlopes = null) : SplineBoundary<T>
    {
        internal T LeftSlope { get; } = leftSlope;
        internal T RightSlope { get; } = rightSlope;

        // The slopes at double precision, in the same units as LeftSlope and RightSlope; null when the boundary
        // was built from working-type slopes, which are already the caller's exact values.
        internal (double Left, double Right)? DoubleSlopes { get; } = doubleSlopes;
    }
}
