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
/// Pre-computed Fritsch-Carlson monotone-preserving cubic Hermite spline
/// (Fritsch &amp; Carlson 1980) with improved endpoint derivatives
/// (arXiv:2402.01324). Monotonicity is preserved on every segment — the spline
/// does not overshoot between data points. O(n) setup cost at construction;
/// each <see cref="Interpolate"/> call is O(log n) bracket search + O(1) cubic
/// Hermite evaluation.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A+sqrt: Requires basic floating-point arithmetic plus square root via NumericPrecision.
/// </remarks>
public sealed class MonotoneCubicSpline<T> where T : IFloatingPoint<T>
{
    private readonly T[] _xs;
    private readonly T[] _ys;
    private readonly T[] _m;

    private static readonly T s_zero = T.Zero;
    private static readonly T s_one = T.One;
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);
    private static readonly T s_nine = T.CreateChecked(9);

    /// <inheritdoc cref="MonotoneCubicSpline(IReadOnlyList{double}, IReadOnlyList{double})"/>
    public MonotoneCubicSpline(IReadOnlyList<T> xs, IReadOnlyList<T> ys)
    {
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

        for (var i = 1; i < n; i++)
        {
            if (_xs[i] <= _xs[i - 1])
            {
                throw new ArgumentException("x-values must be strictly increasing.", nameof(xs));
            }
        }

        _m = new T[n];
        var delta = new T[n - 1];
        var h = new T[n - 1];

        for (var i = 0; i < n - 1; i++)
        {
            h[i] = _xs[i + 1] - _xs[i];
            delta[i] = (_ys[i + 1] - _ys[i]) / h[i];
        }

        if (n >= 3)
        {
            var m0 = ((s_two * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
            _m[0] = m0 * delta[0] > s_zero && T.Abs(m0) <= s_three * T.Abs(delta[0]) ? m0 : delta[0];
        }
        else
        {
            _m[0] = delta[0];
        }

        if (n >= 3)
        {
            var hLast = h[n - 2];
            var hPrev = h[n - 3];
            var mn = ((s_two * hLast + hPrev) * delta[n - 2] - hLast * delta[n - 3]) / (hLast + hPrev);
            _m[n - 1] = mn * delta[n - 2] > s_zero && T.Abs(mn) <= s_three * T.Abs(delta[n - 2]) ? mn : delta[n - 2];
        }
        else
        {
            _m[n - 1] = delta[n - 2];
        }

        for (var i = 1; i < n - 1; i++)
        {
            if (delta[i - 1] * delta[i] <= s_zero)
            {
                _m[i] = s_zero;
            }
            else
            {
                _m[i] = (delta[i - 1] + delta[i]) / s_two;
            }
        }

        for (var i = 0; i < n - 1; i++)
        {
            if (delta[i] == s_zero)
            {
                _m[i] = s_zero;
                _m[i + 1] = s_zero;
            }
            else
            {
                var alpha = _m[i] / delta[i];
                var beta = _m[i + 1] / delta[i];
                var phi = alpha * alpha + beta * beta;

                if (phi > s_nine)
                {
                    var tau = s_three / NumericPrecision<T>.Sqrt(phi);
                    _m[i] = tau * alpha * delta[i];
                    _m[i + 1] = tau * beta * delta[i];
                }
            }
        }
    }

    /// <inheritdoc cref="MonotoneCubicSpline.Interpolate"/>
    public T Interpolate(T x)
    {
        if (x <= _xs[0])
        {
            return _ys[0];
        }

        if (x >= _xs[^1])
        {
            return _ys[^1];
        }

        int lo = 0;
        int hi = _xs.Length - 2;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            if (x < _xs[mid])
            {
                hi = mid - 1;
            }
            else if (x >= _xs[mid + 1])
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
        var s = (x - _xs[k]) / h;
        var s2 = s * s;
        var s3 = s2 * s;

        var h00 = s_two * s3 - s_three * s2 + s_one;
        var h10 = s3 - s_two * s2 + s;
        var h01 = -s_two * s3 + s_three * s2;
        var h11 = s3 - s2;

        return h00 * _ys[k] + h10 * h * _m[k] + h01 * _ys[k + 1] + h11 * h * _m[k + 1];
    }
}

/// <summary>
/// Pre-computed Fritsch-Carlson monotone-preserving cubic Hermite spline
/// (Fritsch &amp; Carlson 1980) with improved endpoint derivatives
/// (arXiv:2402.01324). Monotonicity is preserved on every segment — the spline
/// does not overshoot between data points. O(n) setup cost at construction;
/// each <see cref="Interpolate"/> call is O(log n) bracket search + O(1) cubic
/// Hermite evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Convergence: O(h⁴) at interior knots for smooth data, O(h²) near the boundary
/// due to the three-point tangent formula. The Fritsch-Carlson monotonicity
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
/// Endpoint tangents use a one-sided three-point formula when the data supports it,
/// reducing boundary error from O(h) to O(h²). Falls back to the standard one-point
/// formula when the three-point estimate would break monotonicity. Flat
/// extrapolation beyond the node range.
/// </para>
/// </remarks>
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
    /// Thrown when fewer than 2 points are provided, lengths differ, or x-values are
    /// not strictly increasing.
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
