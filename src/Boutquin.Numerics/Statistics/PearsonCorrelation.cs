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
    /// <param name="x">First series. Length is the minimum of the two spans.</param>
    /// <param name="y">Second series.</param>
    /// <returns>Correlation in [-1, 1], or 0 for degenerate inputs.</returns>
    public static T Compute(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        var n = Math.Min(x.Length, y.Length);
        if (n < 3)
        {
            return T.Zero;
        }

        var nT = T.CreateChecked(n);
        T sumX = T.Zero, sumY = T.Zero;
        for (var i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
        }

        var meanX = sumX / nT;
        var meanY = sumY / nT;

        T covXY = T.Zero, varX = T.Zero, varY = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var dx = x[i] - meanX;
            var dy = y[i] - meanY;
            covXY += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }

        if (varX == T.Zero || varY == T.Zero)
        {
            return T.Zero;
        }

        var denominator = NumericPrecision<T>.Sqrt(varX * varY);
        var correlation = covXY / denominator;

        return T.Clamp(correlation, -T.One, T.One);
    }

    /// <summary>
    /// Pearson correlation over the slice <c>[start, end)</c> of two arrays.
    /// </summary>
    /// <param name="x">First series.</param>
    /// <param name="y">Second series.</param>
    /// <param name="start">Inclusive start index.</param>
    /// <param name="end">Exclusive end index.</param>
    /// <returns>Correlation in [-1, 1], or 0 for degenerate inputs.</returns>
    public static T Compute(T[] x, T[] y, int start, int end)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        var n = end - start;
        if (n < 3)
        {
            return T.Zero;
        }

        return Compute(x.AsSpan(start, n), y.AsSpan(start, n));
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
    public static T[] Rolling(T[] returnsA, T[] returnsB, int windowSize)
    {
        ArgumentNullException.ThrowIfNull(returnsA);
        ArgumentNullException.ThrowIfNull(returnsB);

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

        var windowSizeT = T.CreateChecked(windowSize);
        var resultCount = returnsA.Length - windowSize + 1;
        var result = new T[resultCount];

        for (var start = 0; start < resultCount; start++)
        {
            var meanA = T.Zero;
            var meanB = T.Zero;
            for (var i = start; i < start + windowSize; i++)
            {
                meanA += returnsA[i];
                meanB += returnsB[i];
            }

            meanA /= windowSizeT;
            meanB /= windowSizeT;

            var covAB = T.Zero;
            var varA = T.Zero;
            var varB = T.Zero;
            for (var i = start; i < start + windowSize; i++)
            {
                var dA = returnsA[i] - meanA;
                var dB = returnsB[i] - meanB;
                covAB += dA * dB;
                varA += dA * dA;
                varB += dB * dB;
            }

            if (varA == T.Zero || varB == T.Zero)
            {
                result[start] = T.Zero;
                continue;
            }

            var denominator = NumericPrecision<T>.Sqrt(varA * varB);
            result[start] = T.Clamp(covAB / denominator, -T.One, T.One);
        }

        return result;
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
public static class PearsonCorrelation
{
    /// <summary>
    /// Pearson correlation between two equal-length series.
    /// </summary>
    /// <param name="x">First series. Length is the minimum of the two spans.</param>
    /// <param name="y">Second series.</param>
    /// <returns>Correlation in [-1, 1], or 0 for degenerate inputs.</returns>
    public static decimal Compute(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => PearsonCorrelation<decimal>.Compute(x, y);

    /// <summary>
    /// Pearson correlation over the slice <c>[start, end)</c> of two arrays.
    /// </summary>
    /// <param name="x">First series.</param>
    /// <param name="y">Second series.</param>
    /// <param name="start">Inclusive start index.</param>
    /// <param name="end">Exclusive end index.</param>
    /// <returns>Correlation in [-1, 1], or 0 for degenerate inputs.</returns>
    public static decimal Compute(decimal[] x, decimal[] y, int start, int end)
        => PearsonCorrelation<decimal>.Compute(x, y, start, end);

    /// <summary>
    /// Rolling Pearson correlation between two series. Each window is
    /// computed independently with the two-pass algorithm. Returns an array
    /// of length <c>returnsA.Length - windowSize + 1</c>.
    /// </summary>
    /// <param name="returnsA">First return series.</param>
    /// <param name="returnsB">Second return series (must be the same length as <paramref name="returnsA"/>).</param>
    /// <param name="windowSize">Rolling window size. Must lie in [2, series length].</param>
    /// <exception cref="ArgumentException">Series lengths differ, or window size is out of range.</exception>
    public static decimal[] Rolling(decimal[] returnsA, decimal[] returnsB, int windowSize)
        => PearsonCorrelation<decimal>.Rolling(returnsA, returnsB, windowSize);
}
