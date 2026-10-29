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
/// Computes Fisher's adjusted sample excess kurtosis (G₂) over a span of
/// observations using the finite-sample correction factor.
/// </summary>
/// <remarks>
/// <para>
/// The formula applied is:
/// <code>
/// G₂ = [n(n+1) / ((n-1)(n-2)(n-3))] * Σ((xᵢ - μ̄)/s)⁴
///      − [3(n-1)² / ((n-2)(n-3))]
/// </code>
/// where μ̄ is the sample mean and s is the sample standard deviation (N-1 divisor).
/// This matches the convention used by Excel KURT, SciPy <c>stats.kurtosis</c>
/// with <c>bias=False, fisher=True</c>, and most statistical packages. A normal
/// distribution has an excess kurtosis of zero under this definition.
/// </para>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/> implementing
/// <see cref="IFloatingPoint{TSelf}"/>. Square-root uses <see cref="NumericPrecision{T}.Sqrt"/>
/// to support <c>decimal</c>.
/// </para>
/// <para>
/// Algorithm: two-pass. Pass 1 computes the mean and sample variance via
/// <see cref="WelfordMoments{T}.Compute"/>. Pass 2 accumulates the sum of
/// fourth-power standardized deviations, computed as <c>(z²)²</c> to avoid
/// an explicit fourth-power operation. Correction-factor products are formed
/// in <typeparamref name="T"/> to avoid integer overflow for large n. Half
/// observations are widened losslessly to double for the calculation and the
/// final result is rounded to Half.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class SampleExcessKurtosis<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Computes Fisher's adjusted sample excess kurtosis of <paramref name="values"/>.
    /// </summary>
    /// <param name="values">A read-only span of at least four observations.</param>
    /// <returns>The bias-corrected sample excess kurtosis.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="values"/> contains fewer than 4 elements or
    /// a non-finite observation.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the sample standard deviation is zero (all values are identical).
    /// Also thrown when the observation count cannot convert to a finite
    /// <typeparamref name="T"/>; <see cref="Half"/> accepts at most 65,519 observations.
    /// </exception>
    public static T Compute(ReadOnlySpan<T> values)
    {
        var n = values.Length;
        if (n < 4)
        {
            throw new ArgumentException(
                "Sample excess kurtosis requires at least 4 observations.",
                nameof(values));
        }

        for (var i = 0; i < n; i++)
        {
            if (!T.IsFinite(values[i]))
            {
                throw new ArgumentException(
                    $"Sample excess kurtosis requires finite observations; found a non-finite value at index {i}.",
                    nameof(values));
            }
        }

        if (typeof(T) == typeof(Half))
        {
            if (!T.IsFinite(T.CreateChecked(n)))
            {
                throw new InvalidOperationException($"The observation count {n} cannot be represented as a finite {typeof(T).Name} value.");
            }

            // Half standardized powers can overflow although the adjusted moment
            // is finite. Widen losslessly and round only the final statistic.
            var widened = new double[n];
            for (var i = 0; i < n; i++)
            {
                widened[i] = double.CreateChecked(values[i]);
            }

            return T.CreateChecked(SampleExcessKurtosis<double>.Compute(widened));
        }

        // Translate before scaling so a large common offset cannot erase the
        // representable spread when normalized deviations are squared.
        var center = PearsonCorrelation<T>.NormalizationCenter(values);
        var scale = T.Zero;
        foreach (var x in values)
        {
            scale = T.Max(scale, T.Abs(x - center));
        }

        if (scale != T.Zero)
        {
            var scaledBuffer = new T[n];
            for (var i = 0; i < n; i++)
            {
                scaledBuffer[i] = (values[i] - center) / scale;
            }

            values = scaledBuffer;
        }

        var (mean, variance) = WelfordMoments<T>.Compute(values);
        var stdDev = NumericPrecision<T>.Sqrt(variance);

        if (T.IsZero(stdDev))
        {
            throw new InvalidOperationException(
                "Sample excess kurtosis is undefined when the standard deviation is zero.");
        }

        var nt = T.CreateChecked(n);
        var nm1 = T.CreateChecked(n - 1);
        var nm2 = T.CreateChecked(n - 2);
        var nm3 = T.CreateChecked(n - 3);

        var sumFourth = T.Zero;
        foreach (var x in values)
        {
            var z = (x - mean) / stdDev;
            var z2 = z * z;
            sumFourth += z2 * z2;
        }

        // term1 = n(n+1) / ((n-1)(n-2)(n-3)) * sumFourth.
        var term1 = nt * T.CreateChecked((long)n + 1L) / (nm1 * nm2 * nm3) * sumFourth;

        // term2 = 3(n-1)² / ((n-2)(n-3))
        var term2 = T.CreateChecked(3) * nm1 * nm1 / (nm2 * nm3);

        return term1 - term2;
    }
}

/// <summary>
/// Decimal-typed facade for <see cref="SampleExcessKurtosis{T}"/> at <c>T = decimal</c>.
/// </summary>
[Obsolete(
    "Use SampleExcessKurtosis<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class SampleExcessKurtosis
{
    /// <summary>
    /// Computes Fisher's adjusted sample excess kurtosis using 28-digit decimal arithmetic.
    /// </summary>
    /// <param name="values">A read-only span of at least four <see cref="decimal"/> observations.</param>
    /// <returns>The bias-corrected sample excess kurtosis as a <see cref="decimal"/>.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="values"/> contains fewer than 4 elements.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the sample standard deviation is zero.
    /// </exception>
    public static decimal Compute(ReadOnlySpan<decimal> values)
        => SampleExcessKurtosis<decimal>.Compute(values);
}
