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
    private static readonly T s_epsilon = T.CreateChecked(1e-12);

    /// <inheritdoc cref="TwoPointLinearInterpolator.Interpolate"/>
    public static T Interpolate(T x0, T y0, T x1, T y1, T x)
    {
        if (T.Abs(x1 - x0) < s_epsilon)
        {
            throw new ArgumentException("x0 and x1 must be distinct.");
        }

        T weight = (x - x0) / (x1 - x0);
        return y0 + (weight * (y1 - y0));
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
    /// Thrown when <paramref name="x0"/> and <paramref name="x1"/> are not distinct
    /// (differ by less than 1e-12).
    /// </exception>
    public static double Interpolate(double x0, double y0, double x1, double y1, double x)
    {
        return TwoPointLinearInterpolator<double>.Interpolate(x0, y0, x1, y1, x);
    }
}
