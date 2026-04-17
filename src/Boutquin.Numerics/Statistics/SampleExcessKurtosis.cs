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
/// observations using the unbiased bias-correction factor.
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
/// an explicit fourth-power operation. Integer correction-factor products use
/// <c>long</c> arithmetic to prevent overflow for large n.
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
    /// Thrown when <paramref name="values"/> contains fewer than 4 elements.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the sample standard deviation is zero (all values are identical).
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

        var (mean, variance) = WelfordMoments<T>.Compute(values);
        var stdDev = NumericPrecision<T>.Sqrt(variance);

        if (T.IsZero(stdDev))
        {
            throw new InvalidOperationException(
                "Sample excess kurtosis is undefined when the standard deviation is zero.");
        }

        var sumFourth = T.Zero;
        foreach (var x in values)
        {
            var z = (x - mean) / stdDev;
            var z2 = z * z;
            sumFourth += z2 * z2;
        }

        // Use long arithmetic for the integer products to avoid int32 overflow for n > ~1290.
        var nl = (long)n;

        // term1 = n(n+1) / ((n-1)(n-2)(n-3)) * sumFourth
        var term1 = T.CreateChecked(nl * (nl + 1L))
                    / T.CreateChecked((nl - 1L) * (nl - 2L) * (nl - 3L))
                    * sumFourth;

        // term2 = 3(n-1)² / ((n-2)(n-3))
        var term2 = T.CreateChecked(3L * (nl - 1L) * (nl - 1L))
                    / T.CreateChecked((nl - 2L) * (nl - 3L));

        return term1 - term2;
    }
}

/// <summary>
/// Decimal-typed facade for <see cref="SampleExcessKurtosis{T}"/> at <c>T = decimal</c>.
/// </summary>
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
