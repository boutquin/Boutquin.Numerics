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
/// Fritsch-Carlson monotone-preserving cubic Hermite interpolation with
/// improved endpoint derivatives (arXiv:2402.01324). Guarantees that the
/// interpolant is monotone on every segment — no spurious oscillation or
/// overshoot between data points. Flat extrapolation beyond the node range.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A+sqrt: Requires basic floating-point arithmetic plus square root via NumericPrecision.
/// </remarks>
public sealed class MonotoneCubicInterpolator<T> : IInterpolator<T> where T : IFloatingPoint<T>
{
    /// <inheritdoc cref="MonotoneCubicInterpolator.Instance"/>
    public static readonly MonotoneCubicInterpolator<T> Instance = new();

    private static readonly T s_zero = T.Zero;
    private static readonly T s_one = T.One;
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_three = T.CreateChecked(3);
    private static readonly T s_nine = T.CreateChecked(9);

    private MonotoneCubicInterpolator() { }

    /// <inheritdoc/>
    public string Name => "MonotoneCubic";

    /// <inheritdoc/>
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

        T[] delta = new T[n - 1];
        for (var i = 0; i < n - 1; i++)
        {
            delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
        }

        T[] m = new T[n];

        if (n >= 3)
        {
            var h0 = xs[1] - xs[0];
            var h1 = xs[2] - xs[1];
            var m0 = ((s_two * h0 + h1) * delta[0] - h0 * delta[1]) / (h0 + h1);
            m[0] = m0 * delta[0] > s_zero && T.Abs(m0) <= s_three * T.Abs(delta[0]) ? m0 : delta[0];
        }
        else
        {
            m[0] = delta[0];
        }

        if (n >= 3)
        {
            var hLast = xs[n - 1] - xs[n - 2];
            var hPrev = xs[n - 2] - xs[n - 3];
            var mn = ((s_two * hLast + hPrev) * delta[n - 2] - hLast * delta[n - 3]) / (hLast + hPrev);
            m[n - 1] = mn * delta[n - 2] > s_zero && T.Abs(mn) <= s_three * T.Abs(delta[n - 2]) ? mn : delta[n - 2];
        }
        else
        {
            m[n - 1] = delta[n - 2];
        }

        for (var i = 1; i < n - 1; i++)
        {
            if (delta[i - 1] * delta[i] <= s_zero)
            {
                m[i] = s_zero;
            }
            else
            {
                m[i] = (delta[i - 1] + delta[i]) / s_two;
            }
        }

        for (var i = 0; i < n - 1; i++)
        {
            if (delta[i] == s_zero)
            {
                m[i] = s_zero;
                m[i + 1] = s_zero;
            }
            else
            {
                var alpha = m[i] / delta[i];
                var beta = m[i + 1] / delta[i];
                var phi = alpha * alpha + beta * beta;

                if (phi > s_nine)
                {
                    var tau = s_three / NumericPrecision<T>.Sqrt(phi);
                    m[i] = tau * alpha * delta[i];
                    m[i + 1] = tau * beta * delta[i];
                }
            }
        }

        var k = InterpolationHelper.FindInterval(x, xs);
        var h = xs[k + 1] - xs[k];
        var s = (x - xs[k]) / h;
        var s2 = s * s;
        var s3 = s2 * s;

        var h00 = s_two * s3 - s_three * s2 + s_one;
        var h10 = s3 - s_two * s2 + s;
        var h01 = -s_two * s3 + s_three * s2;
        var h11 = s3 - s2;

        return h00 * ys[k] + h10 * h * m[k] + h01 * ys[k + 1] + h11 * h * m[k + 1];
    }
}

/// <summary>
/// Fritsch-Carlson monotone-preserving cubic Hermite interpolation with
/// improved endpoint derivatives (arXiv:2402.01324). Guarantees that the
/// interpolant is monotone on every segment — no spurious oscillation or
/// overshoot between data points. Flat extrapolation beyond the node range.
/// </summary>
/// <remarks>
/// <para>
/// Algorithm (Fritsch &amp; Carlson 1980, SIAM J. Numer. Anal. 17, §3):
/// </para>
/// <list type="number">
///   <item>Compute finite-difference slopes <c>δᵢ = (yᵢ₊₁ − yᵢ) / (xᵢ₊₁ − xᵢ)</c>.</item>
///   <item>Initial interior tangents: <c>mᵢ = (δᵢ₋₁ + δᵢ) / 2</c>, zeroed at sign changes to preserve monotonicity.</item>
///   <item>Endpoint tangents: one-sided three-point formula (arXiv:2402.01324) when it preserves monotonicity, falling back to <c>δ₀</c> / <c>δₙ₋₂</c> otherwise. The refinement reduces boundary error from O(h) to O(h²).</item>
///   <item>Monotonicity enforcement: when <c>φ = (mᵢ/δᵢ)² + (mᵢ₊₁/δᵢ)² &gt; 9</c>, rescale both tangents by <c>τ = 3/√φ</c>, bringing the segment into the Fritsch-Carlson monotonicity region (de Boor &amp; Swartz 1977 form).</item>
///   <item>Evaluate the cubic Hermite polynomial <c>y(x) = h₀₀(s)yₖ + h₁₀(s)·h·mₖ + h₀₁(s)yₖ₊₁ + h₁₁(s)·h·mₖ₊₁</c>.</item>
/// </list>
/// <para>
/// Stateless; exposed as the singleton <see cref="Instance"/>. Every call
/// recomputes tangents from <paramref>xs</paramref>/<paramref>ys</paramref>.
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
