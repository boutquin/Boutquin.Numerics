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
/// Computes the adjusted Fisher-Pearson standardized third moment (sample skewness)
/// over a span of observations using the unbiased bias-correction factor.
/// </summary>
/// <remarks>
/// <para>
/// The formula applied is:
/// <code>
/// skewness = [n / ((n-1)(n-2))] * Σ((xᵢ - μ̄) / s)³
/// </code>
/// where μ̄ is the sample mean and s is the sample standard deviation (N-1 divisor).
/// This is the same correction used by most statistical packages (e.g., Excel SKEW,
/// SciPy <c>stats.skew</c> with <c>bias=False</c>).
/// </para>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/> implementing
/// <see cref="IFloatingPoint{TSelf}"/>. Square-root uses <see cref="NumericPrecision{T}.Sqrt"/>
/// to support <c>decimal</c>.
/// </para>
/// <para>
/// Algorithm: two-pass. Pass 1 computes the mean and sample variance via
/// <see cref="WelfordMoments{T}.Compute"/>. Pass 2 accumulates the sum of
/// cubed standardized deviations. The bias-correction factor is applied once
/// at the end.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class SampleSkewness<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Computes the adjusted Fisher-Pearson sample skewness of <paramref name="values"/>.
    /// </summary>
    /// <param name="values">A read-only span of at least three observations.</param>
    /// <returns>The bias-corrected sample skewness.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="values"/> contains fewer than 3 elements.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the sample standard deviation is zero (all values are identical).
    /// </exception>
    public static T Compute(ReadOnlySpan<T> values)
    {
        var n = values.Length;
        if (n < 3)
        {
            throw new ArgumentException(
                "Sample skewness requires at least 3 observations.",
                nameof(values));
        }

        var (mean, variance) = WelfordMoments<T>.Compute(values);
        var stdDev = NumericPrecision<T>.Sqrt(variance);

        if (T.IsZero(stdDev))
        {
            throw new InvalidOperationException(
                "Sample skewness is undefined when the standard deviation is zero.");
        }

        var sumCubed = T.Zero;
        foreach (var x in values)
        {
            var z = (x - mean) / stdDev;
            sumCubed += z * z * z;
        }

        // Bias-correction factor: n / ((n-1) * (n-2))
        var nT = T.CreateChecked(n);
        var nm1 = T.CreateChecked(n - 1);
        var nm2 = T.CreateChecked(n - 2);
        return nT / (nm1 * nm2) * sumCubed;
    }
}

/// <summary>
/// Decimal-typed facade for <see cref="SampleSkewness{T}"/> at <c>T = decimal</c>.
/// </summary>
public static class SampleSkewness
{
    /// <summary>
    /// Computes the adjusted Fisher-Pearson sample skewness using 28-digit decimal arithmetic.
    /// </summary>
    /// <param name="values">A read-only span of at least three <see cref="decimal"/> observations.</param>
    /// <returns>The bias-corrected sample skewness as a <see cref="decimal"/>.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="values"/> contains fewer than 3 elements.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the sample standard deviation is zero.
    /// </exception>
    public static decimal Compute(ReadOnlySpan<decimal> values)
        => SampleSkewness<decimal>.Compute(values);
}
