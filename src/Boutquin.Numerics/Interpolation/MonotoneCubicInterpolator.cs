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

using System.Numerics;
using Boutquin.Numerics.Internal;

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Fritsch-Carlson monotone-preserving cubic Hermite interpolation
/// (Fritsch &amp; Carlson 1980, SIAM J. Numer. Anal. 17(2), 238–246). Guarantees
/// that the interpolant is monotone on every segment — no spurious oscillation or
/// overshoot between data points. Interior tangents use the three-point
/// (parabolic) difference formula and endpoint tangents the one-sided three-point
/// estimate, set to zero when that estimate has the wrong sign. Flat extrapolation
/// beyond the node range.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A+sqrt: Requires basic floating-point arithmetic plus square root via NumericPrecision.
/// </remarks>
public sealed class MonotoneCubicInterpolator<T> : IInterpolator<T> where T : unmanaged, IFloatingPoint<T>
{
    /// <inheritdoc cref="MonotoneCubicInterpolator.Instance"/>
    public static readonly MonotoneCubicInterpolator<T> Instance = new();

    private static readonly T s_one = T.One;
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);

    private const int StackAllocThreshold = 256;

    private MonotoneCubicInterpolator() { }

    /// <inheritdoc/>
    public string Name => "MonotoneCubic";

    /// <inheritdoc/>
    /// <exception cref="ArgumentException"><paramref name="xs"/> is not strictly ascending, or <paramref name="ys"/> holds a non-finite value.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="x"/> is not finite.</exception>
    /// <exception cref="InvalidOperationException">Coordinate normalization cannot preserve distinct knots in the working type.</exception>
    public T Interpolate(T x, ReadOnlySpan<T> xs, ReadOnlySpan<T> ys)
    {
        if (xs.Length != ys.Length)
        {
            throw new ArgumentException("xs and ys must have the same length.");
        }

        if (xs.Length == 0)
        {
            throw new ArgumentException("At least one node is required.");
        }

        InterpolationHelper.ValidateSortedAscending(xs, nameof(xs));
        InterpolationHelper.ValidateFiniteOrdinates(ys, nameof(ys));

        if (!T.IsFinite(x))
        {
            throw new ArgumentOutOfRangeException(nameof(x));
        }

        if (xs.Length == 1)
        {
            return ys[0];
        }

        if (x <= xs[0])
        {
            return ys[0];
        }

        if (x >= xs[^1])
        {
            return ys[^1];
        }

        var n = xs.Length;
        if (n == 2)
        {
            return InterpolationHelper.LinearInterpolate(x, xs[0], ys[0], xs[1], ys[1]);
        }

        var originalXs = xs;
        var originalX = x;
        var coordinateScale = InterpolationHelper.CoordinateScale(xs);
        if (coordinateScale != T.One)
        {
            var normalized = new T[n];
            for (var i = 0; i < n; i++)
            {
                normalized[i] = xs[i] / coordinateScale;
            }

            xs = normalized;
            x /= coordinateScale;
        }

        var ordinateScale = InterpolationHelper.OrdinateScale(ys);
        var scaleOrdinates = ordinateScale != T.One;
        if (scaleOrdinates)
        {
            var normalized = new T[n];
            for (var i = 0; i < n; i++)
            {
                normalized[i] = ys[i] / ordinateScale;
            }

            ys = normalized;
        }

        var derivativeScale = InterpolationHelper.DerivativeScale(xs, ys, out var robust);
        using var scratch = new PooledScratch<T>((n) > StackAllocThreshold ? checked(2 * (n)) : 0);
        var rented = scratch.Buffer;
        {
            Span<T> delta = (n - 1) <= StackAllocThreshold ? stackalloc T[n - 1] : rented.AsSpan(0, n - 1);
            Span<T> m = n <= StackAllocThreshold ? stackalloc T[n] : rented.AsSpan(n, n);
            MonotoneTangentLimiter<T>.BuildTangents(xs, ys, delta, m, derivativeScale, robust);

            // Bracket and measure the query in the caller's own units: dividing a small query
            // by the grid magnitude before subtracting would erase it.
            var normalizedCoordinates = coordinateScale != T.One;
            var k = normalizedCoordinates
                ? InterpolationHelper.FindInterval(originalX, originalXs)
                : InterpolationHelper.FindInterval(x, xs);
            var h = xs[k + 1] - xs[k];
            T offset;
            T s;
            var offsetExact = true;
            if (!normalizedCoordinates)
            {
                offset = x - xs[k];
                s = offset / h;
            }
            else if (InterpolationHelper.TryLocalOffset(originalX, originalXs[k], originalXs[k + 1], out offset, out var originalWidth))
            {
                s = offset / originalWidth;
            }
            else
            {
                offsetExact = false;
                offset = T.Zero;
                s = (x - xs[k]) / h;
            }

            var s2 = s * s;
            var s3 = s2 * s;

            var h00 = s_two * s3 - s_three * s2 + s_one;
            var h10 = s3 - s_two * s2 + s;
            var h01 = -s_two * s3 + s_three * s2;
            var h11 = s3 - s2;

            // A fraction that underflowed (or sits on decimal's grid) has lost the digits of the
            // first-order term h10 * h * m[k]. Take that term from the unrounded offset times the
            // stored slope instead and keep the higher-order remainder of h10, which needs no such
            // digits.
            var linear = T.Zero;
            if (offsetExact && offset != T.Zero && InterpolationHelper.IsCoarseFraction(s)
                && InterpolationHelper.TryScaledProduct(m[k], offset, coordinateScale * derivativeScale, out var contribution))
            {
                linear = contribution;
                h10 = s3 - s_two * s2;
            }

            var result = derivativeScale == T.One
                ? h00 * ys[k] + h10 * h * m[k] + linear + h01 * ys[k + 1] + h11 * h * m[k + 1]
                : h00 * ys[k] + InterpolationHelper.IntegratedDerivative(h10 * m[k], h, derivativeScale) + linear
                    + h01 * ys[k + 1] + InterpolationHelper.IntegratedDerivative(h11 * m[k + 1], h, derivativeScale);
            return scaleOrdinates ? result * ordinateScale : result;
        }
    }
}

internal static class MonotoneTangentLimiter<T> where T : IFloatingPoint<T>
{
    private static readonly T s_three = T.CreateChecked(3);

    // Both public forms handle one/two nodes before constructing cubic tangents.
    // Scratch and destination spans let the stateless form use stack/pool storage
    // while the cached spline owns its tangent array.
    internal static void BuildTangents(
        ReadOnlySpan<T> xs, ReadOnlySpan<T> ys, Span<T> delta, Span<T> tangents,
        T derivativeScale, bool robust)
    {
        var n = xs.Length;
        for (var i = 0; i < n - 1; i++)
        {
            delta[i] = InterpolationHelper.ScaledSecant(
                ys[i + 1] - ys[i], xs[i + 1] - xs[i], derivativeScale, robust);
        }

        var first = InterpolationHelper.EndpointTangent(xs[1] - xs[0], xs[2] - xs[1], delta[0], delta[1], robust);
        tangents[0] = SameStrictSign(first, delta[0]) ? first : T.Zero;
        var last = InterpolationHelper.EndpointTangent(
            xs[n - 1] - xs[n - 2], xs[n - 2] - xs[n - 3], delta[n - 2], delta[n - 3], robust);
        tangents[n - 1] = SameStrictSign(last, delta[n - 2]) ? last : T.Zero;

        for (var i = 1; i < n - 1; i++)
        {
            tangents[i] = SameStrictSign(delta[i - 1], delta[i])
                ? InterpolationHelper.InteriorTangent(xs[i] - xs[i - 1], xs[i + 1] - xs[i], delta[i - 1], delta[i], robust)
                : T.Zero;
        }

        for (var i = 0; i < n - 1; i++)
        {
            Limit(delta[i], ref tangents[i], ref tangents[i + 1]);
        }
    }

    // Equivalent to a * b > 0 without forming the product, which can overflow for
    // finite slopes (decimal throws; IEEE types saturate to infinity).
    internal static bool SameStrictSign(T a, T b) =>
        (a > T.Zero && b > T.Zero) || (a < T.Zero && b < T.Zero);

    internal static void Limit(T delta, ref T left, ref T right)
    {
        if (delta == T.Zero)
        {
            left = T.Zero;
            right = T.Zero;
            return;
        }

        var absDelta = T.Abs(delta);
        var scale = T.Max(T.Abs(left), T.Abs(right));
        if (scale <= absDelta)
        {
            return;
        }

        // The Fritsch-Carlson ratios can overflow even when every slope is finite.
        // Normalize first, then compare the unit-vector norm against 3 * delta / scale.
        var leftUnit = left / scale;
        var rightUnit = right / scale;
        var norm = NumericPrecision<T>.Sqrt(leftUnit * leftUnit + rightUnit * rightUnit);
        if (norm > s_three * (absDelta / scale))
        {
            // τ·m = m·3|δ|/‖(m_left, m_right)‖: the rescale keeps each tangent's sign.
            var boundedSlope = absDelta * (s_three / norm);
            left = boundedSlope * leftUnit;
            right = boundedSlope * rightUnit;
        }
    }
}

/// <summary>
/// Fritsch-Carlson monotone-preserving cubic Hermite interpolation
/// (Fritsch &amp; Carlson 1980, SIAM J. Numer. Anal. 17(2), 238–246). Guarantees
/// that the interpolant is monotone on every segment — no spurious oscillation or
/// overshoot between data points. Interior tangents use the three-point
/// (parabolic) difference formula and endpoint tangents the one-sided three-point
/// estimate, set to zero when that estimate has the wrong sign. Flat
/// extrapolation beyond the node range.
/// </summary>
/// <remarks>
/// <para>
/// Algorithm (Fritsch &amp; Carlson 1980, SIAM J. Numer. Anal. 17, §3):
/// </para>
/// <list type="number">
///   <item>Compute finite-difference slopes <c>δᵢ = (yᵢ₊₁ − yᵢ) / (xᵢ₊₁ − xᵢ)</c>.</item>
///   <item>Initial interior tangents (Fritsch &amp; Carlson §4, Step 1): the three-point difference formula <c>mᵢ = (hᵢ·δᵢ₋₁ + hᵢ₋₁·δᵢ) / (hᵢ₋₁ + hᵢ)</c>, the derivative of the parabola through the three nodes, zeroed at sign changes to preserve monotonicity.</item>
///   <item>Endpoint tangents: the one-sided three-point (parabolic) estimate <c>m₀ = ((2h₀ + h₁)δ₀ − h₀δ₁)/(h₀ + h₁)</c> (and its mirror at the last node), set to zero when its sign differs from <c>δ₀</c> / <c>δₙ₋₂</c>, as Fritsch &amp; Carlson recommend over the secant slope. Magnitude is left to the next step.</item>
///   <item>Monotonicity enforcement: when <c>φ = (mᵢ/δᵢ)² + (mᵢ₊₁/δᵢ)² &gt; 9</c>, rescale both tangents by <c>τ = 3/√φ</c>, bringing the segment into the Fritsch-Carlson S2 circle. Their de Boor–Swartz attribution applies to the different S1 box <c>0 ≤ α, β ≤ 3</c>.</item>
///   <item>Evaluate the cubic Hermite polynomial <c>y(x) = h₀₀(s)yₖ + h₁₀(s)·h·mₖ + h₀₁(s)yₖ₊₁ + h₁₁(s)·h·mₖ₊₁</c>.</item>
/// </list>
/// <para>
/// Stateless; exposed as the singleton <see cref="Instance"/>. Every call
/// recomputes tangents from the <c>xs</c>/<c>ys</c> node arrays.
/// For repeated evaluations on the same node set, use
/// <see cref="MonotoneCubicSpline"/> which precomputes tangents at construction
/// (O(n) setup, O(log n) per evaluation vs. O(n) per call here).
/// </para>
/// <para>
/// Prefer this over <see cref="CubicSplineInterpolator"/> when the data is
/// known to be monotone (yield curves, cumulative distributions) and
/// overshoot between nodes would be physically or semantically invalid;
/// prefer cubic splines when C² smoothness matters more than monotonicity.
/// </para>
/// </remarks>
[Obsolete(
    "Use MonotoneCubicInterpolator<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class MonotoneCubicInterpolator : IInterpolator
{
    /// <summary>
    /// Shared singleton instance.
    /// </summary>
    public static readonly MonotoneCubicInterpolator Instance = new();

    private MonotoneCubicInterpolator() { }

    /// <inheritdoc />
    public string Name => "MonotoneCubic";

    /// <inheritdoc />
    public double Interpolate(double x, ReadOnlySpan<double> xs, ReadOnlySpan<double> ys)
    {
        return MonotoneCubicInterpolator<double>.Instance.Interpolate(x, xs, ys);
    }
}
