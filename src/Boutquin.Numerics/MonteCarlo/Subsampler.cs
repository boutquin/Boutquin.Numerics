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

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Subsampling (Politis, Romano &amp; Wolf 1999). Constructs the empirical
/// distribution of a statistic by evaluating it on every overlapping
/// subsample of length <c>b &lt; T</c>. Unlike the bootstrap, subsampling
/// is consistent under far weaker assumptions — including unit roots,
/// extreme-value statistics, and slowly mixing series — at the cost of
/// requiring a smaller subsample size <c>b ≪ T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Politis, D. N., Romano, J. P. &amp; Wolf, M. (1999).
/// <em>Subsampling</em>. Springer Series in Statistics.
/// </para>
/// <para>
/// Use subsampling when the bootstrap is known to fail (e.g., for the
/// sample maximum, or for non-stationary integrated series) or when
/// distributional assumptions are too strong. The subsample length must
/// satisfy <c>b → ∞</c> and <c>b/T → 0</c>; <c>b ≈ T^(2/3)</c> is a common
/// rule of thumb.
/// </para>
/// <para>
/// Tier A: Arithmetic order-statistic operations on floating-point types.
/// </para>
/// </remarks>
public static class Subsampler
{
    /// <summary>
    /// Computes the subsample distribution of a scalar statistic.
    /// </summary>
    /// <param name="series">Source series (length T).</param>
    /// <param name="subsampleLength">Subsample length b. Must satisfy 1 ≤ b &lt; T.</param>
    /// <param name="statistic">Statistic to evaluate on each subsample.</param>
    /// <returns>Sorted (ascending) array of statistic values, length T − b + 1.</returns>
    public static T[] Run<T>(T[] series, int subsampleLength, Func<ReadOnlySpan<T>, T> statistic)
        where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(series);
        ArgumentNullException.ThrowIfNull(statistic);
        if (subsampleLength < 1 || subsampleLength >= series.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(subsampleLength), subsampleLength,
                $"Subsample length must lie in [1, {series.Length - 1}].");
        }

        var subsampleCount = series.Length - subsampleLength + 1;
        var values = new T[subsampleCount];
        for (var i = 0; i < subsampleCount; i++)
        {
            values[i] = statistic(series.AsSpan(i, subsampleLength));
        }

        Array.Sort(values);
        return values;
    }

    /// <summary>
    /// Suggested subsample length using the b ≈ T^(2/3) rule of thumb.
    /// </summary>
    public static int SuggestSubsampleLength(int seriesLength)
        => Math.Max(2, (int)Math.Round(Math.Pow(seriesLength, 2.0 / 3.0)));
}
