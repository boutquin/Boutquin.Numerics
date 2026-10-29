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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Generic Pearson product-moment correlation between paired series. Uses a two-pass
/// algorithm (compute means first, then deviations) to avoid the catastrophic
/// cancellation that affects the one-pass form when the
/// series length is large or the series is noisy.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// The result is clamped to [-1, 1] to absorb floating-point noise that can
/// produce values marginally outside the valid range. Returns 0 for constant
/// series, for fewer than three observations, or when either series has zero
/// variance — the correlation is undefined in those cases.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class PearsonCorrelation<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Pearson correlation between two equal-length series.
    /// </summary>
    /// <param name="x">First series.</param>
    /// <param name="y">Second series. Must have the same length as <paramref name="x"/>.</param>
    /// <returns>Correlation in [-1, 1], or 0 for degenerate inputs.</returns>
    public static T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Series must have the same length.", nameof(y));
        }

        ValidateFinite(x, nameof(x));
        ValidateFinite(y, nameof(y));

        var n = x.Length;
        if (n < 3)
        {
            return T.Zero;
        }

        if (typeof(T) == typeof(Half))
        {
            return ComputeHalf(x, y);
        }

        var centerX = NormalizationCenter(x);
        var centerY = NormalizationCenter(y);
        var scaleX = T.Zero;
        var scaleY = T.Zero;
        for (var i = 0; i < n; i++)
        {
            scaleX = T.Max(scaleX, T.Abs(x[i] - centerX));
            scaleY = T.Max(scaleY, T.Abs(y[i] - centerY));
        }

        if (scaleX == T.Zero || scaleY == T.Zero)
        {
            return T.Zero;
        }

        // Correlation is unchanged by positive scaling. Normalize before sums and
        // products so finite observations cannot overflow intermediate arithmetic.
        var nT = T.CreateChecked(n);
        T sumX = T.Zero, sumY = T.Zero;
        for (var i = 0; i < n; i++)
        {
            sumX += (x[i] - centerX) / scaleX;
            sumY += (y[i] - centerY) / scaleY;
        }

        var meanX = sumX / nT;
        var meanY = sumY / nT;

        T covXY = T.Zero, varX = T.Zero, varY = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var dx = (x[i] - centerX) / scaleX - meanX;
            var dy = (y[i] - centerY) / scaleY - meanY;
            covXY += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }

        if (varX == T.Zero || varY == T.Zero)
        {
            return T.Zero;
        }

        var correlation = covXY / NumericPrecision<T>.Sqrt(varX) / NumericPrecision<T>.Sqrt(varY);

        return T.Clamp(correlation, -T.One, T.One);
    }

    /// <summary>
    /// Rolling Pearson correlation between two series. Each window is
    /// computed independently with the two-pass algorithm. Returns an array
    /// of length <c>returnsA.Length - windowSize + 1</c>.
    /// </summary>
    /// <param name="returnsA">First return series.</param>
    /// <param name="returnsB">Second return series (must be the same length as <paramref name="returnsA"/>).</param>
    /// <param name="windowSize">Rolling window size. Must lie in [2, series length].</param>
    /// <exception cref="ArgumentException">Series lengths differ, or window size is out of range.</exception>
    public static T[] Rolling(ReadOnlySpan<T> returnsA, ReadOnlySpan<T> returnsB, int windowSize)
    {
        if (returnsA.Length != returnsB.Length)
        {
            throw new ArgumentException("Return series must have the same length.", nameof(returnsB));
        }

        if (windowSize < 2 || windowSize > returnsA.Length)
        {
            throw new ArgumentException(
                $"Window size must be between 2 and {returnsA.Length}, got {windowSize}.",
                nameof(windowSize));
        }

        ValidateFinite(returnsA, nameof(returnsA));
        ValidateFinite(returnsB, nameof(returnsB));

        var resultCount = returnsA.Length - windowSize + 1;
        var result = new T[resultCount];
        if (windowSize < 3)
        {
            return result;
        }

        for (var start = 0; start < resultCount; start++)
        {
            result[start] = Compute(
                returnsA.Slice(start, windowSize), returnsB.Slice(start, windowSize));
        }

        return result;
    }

    // Translate before scaling: dividing a large offset first can erase the
    // representable spread. A range midpoint keeps every deviation representable.
    internal static T NormalizationCenter(ReadOnlySpan<T> values)
    {
        var min = values[0];
        var max = values[0];
        foreach (var value in values)
        {
            min = T.Min(min, value);
            max = T.Max(max, value);
        }

        var two = T.CreateChecked(2);
        return T.IsNegative(min) == T.IsNegative(max)
            ? min + (max - min) / two
            : min / two + max / two;
    }

    private static T ComputeHalf(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        var scaleX = 0.0;
        var scaleY = 0.0;
        for (var i = 0; i < x.Length; i++)
        {
            scaleX = Math.Max(scaleX, Math.Abs(double.CreateChecked(x[i])));
            scaleY = Math.Max(scaleY, Math.Abs(double.CreateChecked(y[i])));
        }

        if (scaleX == 0 || scaleY == 0)
        {
            return T.Zero;
        }

        var sumX = 0.0;
        var sumY = 0.0;
        for (var i = 0; i < x.Length; i++)
        {
            sumX += double.CreateChecked(x[i]) / scaleX;
            sumY += double.CreateChecked(y[i]) / scaleY;
        }

        var meanX = sumX / x.Length;
        var meanY = sumY / y.Length;
        var covariance = 0.0;
        var varianceX = 0.0;
        var varianceY = 0.0;
        for (var i = 0; i < x.Length; i++)
        {
            var dx = double.CreateChecked(x[i]) / scaleX - meanX;
            var dy = double.CreateChecked(y[i]) / scaleY - meanY;
            covariance += dx * dy;
            varianceX += dx * dx;
            varianceY += dy * dy;
        }

        if (varianceX == 0 || varianceY == 0)
        {
            return T.Zero;
        }

        return T.CreateChecked(Math.Clamp(covariance / Math.Sqrt(varianceX) / Math.Sqrt(varianceY), -1.0, 1.0));
    }

    private static void ValidateFinite(ReadOnlySpan<T> values, string parameterName)
    {
        for (var i = 0; i < values.Length; i++)
        {
            if (!T.IsFinite(values[i]))
            {
                throw new ArgumentOutOfRangeException(
                    parameterName, values[i], "Pearson-correlation observations must be finite.");
            }
        }
    }
}

/// <summary>
/// Pearson product-moment correlation between paired series. Uses a two-pass
/// algorithm (compute means first, then deviations) to avoid the catastrophic
/// cancellation that affects the one-pass <c>n.SumXY - SumX.SumY</c> form when the
/// series length is large or the series is noisy.
/// </summary>
/// <remarks>
/// The result is clamped to [-1, 1] to absorb floating-point noise that can
/// produce values marginally outside the valid range. Returns 0 for constant
/// series, for fewer than three observations, or when either series has zero
/// variance — the correlation is undefined in those cases.
/// </remarks>
[Obsolete(
    "Use PearsonCorrelation<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class PearsonCorrelation
{
    /// <summary>
    /// Pearson correlation between two equal-length series.
    /// </summary>
    /// <param name="x">First series. Must have the same length as <paramref name="y"/>.</param>
    /// <exception cref="ArgumentException">The series have different lengths or contain non-finite values.</exception>
    /// <param name="y">Second series.</param>
    /// <returns>Correlation in [-1, 1], or 0 for degenerate inputs.</returns>
    public static decimal Compute(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => PearsonCorrelation<decimal>.Compute(x, y);

    /// <summary>
    /// Rolling Pearson correlation between two series. Each window is
    /// computed independently with the two-pass algorithm. Returns an array
    /// of length <c>returnsA.Length - windowSize + 1</c>.
    /// </summary>
    /// <param name="returnsA">First return series.</param>
    /// <param name="returnsB">Second return series (must be the same length as <paramref name="returnsA"/>).</param>
    /// <param name="windowSize">Rolling window size. Must lie in [2, series length].</param>
    /// <exception cref="ArgumentException">Series lengths differ, or window size is out of range.</exception>
    public static decimal[] Rolling(ReadOnlySpan<decimal> returnsA, ReadOnlySpan<decimal> returnsB, int windowSize)
        => PearsonCorrelation<decimal>.Rolling(returnsA, returnsB, windowSize);
}
