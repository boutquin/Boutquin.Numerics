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
/// Performs linear interpolation between two data points, used for rate curve
/// lookups and volatility surface queries between known tenors or strikes.
/// </summary>
/// <typeparam name="T">The floating-point type for interpolation.</typeparam>
/// <remarks>
/// Tier A: Requires only basic floating-point arithmetic.
/// </remarks>
public static class TwoPointLinearInterpolator<T> where T : IFloatingPoint<T>
{
    /// <inheritdoc cref="TwoPointLinearInterpolator.Interpolate"/>
    public static T Interpolate(T x0, T y0, T x1, T y1, T x)
    {
        if (x1 == x0)
        {
            throw new ArgumentException("x0 and x1 must be distinct.", nameof(x1));
        }

        try
        {
            T weight = InterpolationHelper.LinearWeight(x, x0, x1);
            if (T.IsFinite(weight) || !T.IsFinite(x) || !T.IsFinite(x0) || !T.IsFinite(x1))
            {
                return InterpolationHelper.LinearInterpolate(x, x0, y0, x1, y1);
            }
        }
        catch (OverflowException)
        {
            // Decimal reports an unrepresentable extrapolation weight by throwing.
        }
        catch (DivideByZeroException) when (T.IsFinite(x) && T.IsFinite(x0) && T.IsFinite(x1))
        {
            // The helper's overflow scaling can round a tiny decimal interval to zero.
        }

        // A coordinate ratio can overflow even when the slope and final value fit.
        // Divide the ordinate difference first so small intervals cancel before extrapolation.
        T slope = (y1 - y0) / (x1 - x0);
        return y0 + (x - x0) * slope;
    }
}

/// <summary>
/// Performs linear interpolation between two data points, used for rate curve
/// lookups and volatility surface queries between known tenors or strikes.
/// </summary>
/// <remarks>
/// Renamed from <c>Analytics.Numerics.Interpolation.LinearInterpolator</c> to avoid
/// collision with the multi-node <see cref="LinearInterpolator"/> (from MarketData).
/// </remarks>
[Obsolete(
    "Use TwoPointLinearInterpolator<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class TwoPointLinearInterpolator
{
    /// <summary>
    /// Linearly interpolates (or extrapolates) the y-value at <paramref name="x"/>
    /// given two reference points (x0, y0) and (x1, y1).
    /// </summary>
    /// <param name="x0">The x-coordinate of the first reference point.</param>
    /// <param name="y0">The y-value at <paramref name="x0"/>.</param>
    /// <param name="x1">The x-coordinate of the second reference point. Must differ from <paramref name="x0"/>.</param>
    /// <param name="y1">The y-value at <paramref name="x1"/>.</param>
    /// <param name="x">The x-coordinate at which to interpolate.</param>
    /// <returns>The interpolated y-value: y0 + (x - x0) / (x1 - x0) * (y1 - y0).</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="x0"/> and <paramref name="x1"/> are equal.
    /// </exception>
    /// <remarks>
    /// The query coordinate <paramref name="x"/> is the <b>last</b> parameter here,
    /// following this method's original two-point signature. This differs from the
    /// multi-node <see cref="IInterpolator{T}.Interpolate"/> convention (implemented
    /// by <see cref="LinearInterpolator{T}"/> and its siblings), where the query
    /// value is the <b>first</b> parameter: <c>Interpolate(T x, ReadOnlySpan&lt;T&gt; xs, ReadOnlySpan&lt;T&gt; ys)</c>.
    /// Callers switching between the two-point and multi-node interpolators should
    /// not assume a shared argument order.
    /// </remarks>
    public static double Interpolate(double x0, double y0, double x1, double y1, double x)
    {
        return TwoPointLinearInterpolator<double>.Interpolate(x0, y0, x1, y1, x);
    }
}
