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

namespace Boutquin.Numerics.Interpolation;

/// <summary>
/// Pre-computed Fritsch-Carlson monotone-preserving cubic Hermite spline
/// (Fritsch &amp; Carlson 1980, SIAM J. Numer. Anal. 17(2), 238–246). Interior
/// tangents use the three-point (parabolic) difference formula and endpoint tangents
/// the one-sided three-point estimate, set to zero when that estimate has the wrong
/// sign. Monotonicity is preserved on every segment — the spline
/// does not overshoot between data points. O(n) setup cost at construction;
/// each <see cref="Interpolate"/> call is O(log n) bracket search + O(1) cubic
/// Hermite evaluation.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// <para>
/// Tier A+sqrt: Requires basic floating-point arithmetic plus square root via NumericPrecision.
/// </para>
/// <para>
/// Constraint policy: unlike <see cref="CubicSplineInterpolator{T}"/> and the stateless
/// span-based interpolators, this type constrains <typeparamref name="T"/> only to
/// <see cref="IFloatingPoint{TSelf}"/> — it does NOT add the <c>unmanaged</c> constraint. The
/// <c>unmanaged</c> constraint exists solely to enable per-call <c>stackalloc</c> of transient
/// scratch buffers; this type precomputes its tangents into heap arrays (<c>_xs</c>, <c>_ys</c>,
/// <c>_m</c>) once at construction and allocates nothing per <see cref="Interpolate"/> call, so the
/// constraint would buy nothing and would needlessly exclude managed-only future
/// <typeparamref name="T"/> implementations.
/// </para>
/// </remarks>
public sealed class MonotoneCubicSpline<T> where T : IFloatingPoint<T>
{
    private readonly T[] _xs;
    private readonly T[] _originalXs;
    private readonly T[] _ys;
    private readonly T[] _workingYs;
    private readonly T _ordinateScale;
    private readonly T _coordinateScale = T.One;
    private readonly T _derivativeScale = T.One;
    private readonly T[] _m;

    private static readonly T s_one = T.One;
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);

    /// <inheritdoc cref="MonotoneCubicSpline(IReadOnlyList{double}, IReadOnlyList{double})"/>
    /// <exception cref="ArgumentNullException"><paramref name="xs"/> or <paramref name="ys"/> is <see langword="null"/>.</exception>
    /// <exception cref="InvalidOperationException">Coordinate normalization cannot preserve distinct knots in the working type.</exception>
    public MonotoneCubicSpline(IReadOnlyList<T> xs, IReadOnlyList<T> ys)
    {
        ArgumentNullException.ThrowIfNull(xs);
        ArgumentNullException.ThrowIfNull(ys);

        if (xs.Count < 2)
        {
            throw new ArgumentException("At least 2 data points are required.", nameof(xs));
        }

        if (xs.Count != ys.Count)
        {
            throw new ArgumentException("xs and ys must have the same length.", nameof(ys));
        }

        var n = xs.Count;
        _xs = new T[n];
        _ys = new T[n];
        for (var i = 0; i < n; i++)
        {
            _xs[i] = xs[i];
            _ys[i] = ys[i];
        }

        InterpolationHelper.ValidateSortedAscending<T>(_xs, nameof(xs));
        InterpolationHelper.ValidateFiniteOrdinates<T>(_ys, nameof(ys));

        _originalXs = _xs;
        _workingYs = _ys;
        _ordinateScale = T.One;
        _m = new T[n];
        if (n == 2)
        {
            return;
        }

        _coordinateScale = InterpolationHelper.CoordinateScale<T>(_xs);
        if (_coordinateScale != T.One)
        {
            // Queries are bracketed and measured in the caller's own units, so the knots keep
            // an unnormalized copy: dividing a small query by the grid magnitude erases it.
            _originalXs = (T[])_xs.Clone();
            for (var i = 0; i < n; i++)
            {
                _xs[i] /= _coordinateScale;
            }
        }

        var ordinateScale = InterpolationHelper.OrdinateScale<T>(_ys);
        if (ordinateScale != T.One)
        {
            _ordinateScale = ordinateScale;
            _workingYs = new T[n];
            for (var i = 0; i < n; i++)
            {
                _workingYs[i] = _ys[i] / ordinateScale;
            }
        }

        _derivativeScale = InterpolationHelper.DerivativeScale<T>(_xs, _workingYs, out var robust);
        var delta = new T[n - 1];
        MonotoneTangentLimiter<T>.BuildTangents(_xs, _workingYs, delta, _m, _derivativeScale, robust);
    }

    /// <inheritdoc cref="MonotoneCubicSpline.Interpolate"/>
    public T Interpolate(T x)
    {
        if (!T.IsFinite(x))
        {
            throw new ArgumentOutOfRangeException(nameof(x));
        }

        // Bracket and measure the query in the caller's own units: dividing a small query by the
        // grid magnitude before subtracting would erase it. Without normalization the original
        // knots are the working knots and nothing changes.
        var originalX = x;
        x /= _coordinateScale;

        if (originalX <= _originalXs[0])
        {
            return _ys[0];
        }

        if (originalX >= _originalXs[^1])
        {
            return _ys[^1];
        }

        if (_xs.Length == 2)
        {
            return InterpolationHelper.LinearInterpolate(x, _xs[0], _ys[0], _xs[1], _ys[1]);
        }

        int lo = 0;
        int hi = _xs.Length - 2;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            if (originalX < _originalXs[mid])
            {
                hi = mid - 1;
            }
            else if (originalX >= _originalXs[mid + 1])
            {
                lo = mid + 1;
            }
            else
            {
                lo = mid;
                break;
            }
        }

        var k = lo;
        var h = _xs[k + 1] - _xs[k];
        T offset;
        T s;
        var offsetExact = true;
        if (_coordinateScale == T.One)
        {
            offset = x - _xs[k];
            s = offset / h;
        }
        else if (InterpolationHelper.TryLocalOffset(originalX, _originalXs[k], _originalXs[k + 1], out offset, out var originalWidth))
        {
            s = offset / originalWidth;
        }
        else
        {
            offsetExact = false;
            offset = T.Zero;
            s = (x - _xs[k]) / h;
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
            && InterpolationHelper.TryScaledProduct(_m[k], offset, _coordinateScale * _derivativeScale, out var contribution))
        {
            linear = contribution;
            h10 = s3 - s_two * s2;
        }

        var result = _derivativeScale == T.One
            ? h00 * _workingYs[k] + h10 * h * _m[k] + linear + h01 * _workingYs[k + 1] + h11 * h * _m[k + 1]
            : h00 * _workingYs[k] + InterpolationHelper.IntegratedDerivative(h10 * _m[k], h, _derivativeScale) + linear
                + h01 * _workingYs[k + 1] + InterpolationHelper.IntegratedDerivative(h11 * _m[k + 1], h, _derivativeScale);
        return result * _ordinateScale;
    }
}

/// <summary>
/// Pre-computed Fritsch-Carlson monotone-preserving cubic Hermite spline
/// (Fritsch &amp; Carlson 1980, SIAM J. Numer. Anal. 17(2), 238–246). Interior
/// tangents use the three-point (parabolic) difference formula and endpoint tangents
/// the one-sided three-point estimate, set to zero when that estimate has the wrong
/// sign. Monotonicity is preserved on every segment — the spline
/// does not overshoot between data points. O(n) setup cost at construction;
/// each <see cref="Interpolate"/> call is O(log n) bracket search + O(1) cubic
/// Hermite evaluation.
/// </summary>
/// <remarks>
/// <para>
/// On smooth data on regular refining grids, the second-order three-point tangents
/// give O(h³) error between knots when the limiter is inactive, including boundary
/// segments. Values at the knots reproduce the data. The Fritsch-Carlson monotonicity
/// constraint (<c>φ = (mᵢ/δᵢ)² + (mᵢ₊₁/δᵢ)² ≤ 9</c>) can downgrade local
/// accuracy when it is actively clipping, but guarantees no overshoot.
/// </para>
/// <para>
/// Choose this over the stateless <see cref="MonotoneCubicInterpolator"/> when the
/// same node set is evaluated many times (post-bootstrap curve queries, Monte
/// Carlo path evaluation on a fixed maturity grid). For one-shot evaluation
/// where nodes change every call (in-loop bootstrap), prefer the stateless form.
/// Choose this over <see cref="CubicSplineInterpolator"/> when the data is
/// known to be monotone and overshoot would be invalid (cumulative distributions,
/// yield curves), accepting C¹ instead of C² in exchange for the monotonicity
/// guarantee.
/// </para>
/// <para>
/// Endpoint tangents use the one-sided three-point (parabolic) estimate, set to zero when its
/// sign differs from the end secant, as Fritsch &amp; Carlson recommend. Flat extrapolation
/// beyond the node range.
/// </para>
/// </remarks>
[Obsolete(
    "Use MonotoneCubicSpline<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class MonotoneCubicSpline
{
    private readonly MonotoneCubicSpline<double> _impl;

    /// <summary>
    /// Constructs a monotone cubic spline from the given data points, pre-computing
    /// Fritsch-Carlson tangents.
    /// </summary>
    /// <param name="xs">Strictly increasing x-coordinates. Must contain at least 2 elements.</param>
    /// <param name="ys">Y-values corresponding to each x-coordinate.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when fewer than 2 points are provided, lengths differ, x-values are
    /// not strictly increasing, or a y-value is not finite.
    /// </exception>
    public MonotoneCubicSpline(IReadOnlyList<double> xs, IReadOnlyList<double> ys)
    {
        _impl = new MonotoneCubicSpline<double>(xs, ys);
    }

    /// <summary>
    /// Evaluates the spline at the given x-coordinate.
    /// Flat extrapolation at boundaries.
    /// </summary>
    /// <param name="x">The x-coordinate at which to evaluate.</param>
    /// <returns>The interpolated y-value.</returns>
    public double Interpolate(double x)
    {
        return _impl.Interpolate(x);
    }
}
