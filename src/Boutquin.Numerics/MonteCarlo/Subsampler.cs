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
/// subsample of length <c>b &lt; T</c>. Run returns sorted raw statistics.
/// A confidence distribution as in Politis–Romano (1994), equation 3.1,
/// requires caller-chosen centering and scaling: <c>τ_b(θ̂_b − θ̂_T)</c>.
/// Consistency requires the applicable mixing and continuity assumptions,
/// <c>b → ∞</c>, <c>b/T → 0</c>, and <c>τ_b/τ_T → 0</c>.
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
    /// <param name="series">
    /// Source series (length T, with T ≥ 2 — see <see cref="SuggestSubsampleLength"/> for
    /// the matching domain on the length side). Accepts any contiguous memory —
    /// <c>T[]</c>, <c>Span&lt;T&gt;</c>, <c>Memory&lt;T&gt;.Span</c>, etc.
    /// </param>
    /// <param name="subsampleLength">Subsample length b. Must satisfy 1 ≤ b &lt; T.</param>
    /// <param name="statistic">Statistic to evaluate on each subsample; each result must be finite.</param>
    /// <returns>Sorted (ascending) array of statistic values, length T − b + 1.</returns>
    public static T[] Run<T>(ReadOnlySpan<T> series, int subsampleLength, Func<ReadOnlySpan<T>, T> statistic)
        where T : IFloatingPoint<T>
    {
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
            var value = statistic(series.Slice(i, subsampleLength));
            if (!T.IsFinite(value))
            {
                throw new ArgumentException($"Statistic returned a non-finite value for subsample {i}.", nameof(statistic));
            }

            values[i] = value;
        }

        Array.Sort(values);
        return values;
    }

    /// <summary>
    /// Suggested subsample length using the b ≈ T^(2/3) rule of thumb.
    /// </summary>
    /// <param name="seriesLength">
    /// Length T of the series the subsample will be drawn from. Must be ≥ 2 — <see cref="Run{T}"/>
    /// requires <c>1 ≤ b &lt; T</c>, and no such <c>b</c> exists for <c>T &lt; 2</c>.
    /// </param>
    /// <returns>
    /// A subsample length clamped to <c>[1, seriesLength − 1]</c> so the result is
    /// always a valid argument to <see cref="Run{T}"/> (which requires
    /// <c>1 ≤ b &lt; seriesLength</c>). For short series the T^(2/3) rule can exceed
    /// <c>seriesLength − 1</c>; the upper clamp keeps the suggestion usable.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="seriesLength"/> is less than 2 — no subsample length in
    /// <c>[1, seriesLength − 1]</c> exists for <see cref="Run{T}"/> to accept.
    /// </exception>
    public static int SuggestSubsampleLength(int seriesLength)
    {
        if (seriesLength < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(seriesLength), seriesLength,
                "Series length must be at least 2 — Run requires 1 <= subsampleLength < seriesLength, " +
                "which admits no valid value for seriesLength < 2.");
        }

        var suggested = Math.Max(2, (int)Math.Round(Math.Pow(seriesLength, 2.0 / 3.0)));
        // Clamp into [1, seriesLength − 1] so Run never rejects the suggestion.
        return Math.Clamp(suggested, 1, seriesLength - 1);
    }
}
