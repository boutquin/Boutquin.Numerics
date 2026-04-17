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

using Boutquin.Numerics.Random;

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Which tail of the empirical bootstrap distribution the p-value measures.
/// </summary>
/// <remarks>
/// <para>
/// Davidson &amp; MacKinnon 2007 §3 describe the fast double bootstrap for
/// both one-sided and two-sided tests. The tail convention must be chosen
/// consistently at both steps of the FDB pipeline (the outer <c>p*</c>
/// rank and the final <c>q*</c> rank) — mixing them silently inverts the
/// p-value. This enum exposes the choice so callers do not have to
/// re-implement the algorithm for the less common tails.
/// </para>
/// <para>
/// Callers who already supply a symmetric "distance from null" statistic
/// (e.g. <c>|mean|</c>, <c>|t-stat|</c>) should stay with
/// <see cref="RightTail"/>: large magnitudes in either direction map to
/// the right tail of a symmetric statistic. <see cref="TwoSided"/> is
/// for asymmetric raw statistics where the caller wants the standard
/// <c>min(2·p_right, 2·p_left, 1)</c> bilateral p-value.
/// </para>
/// </remarks>
public enum PValueTail
{
    /// <summary>
    /// One-sided p-value = fraction of the distribution at or above the
    /// observed statistic. The classical FDB convention — appropriate
    /// whenever large values of the statistic indicate rejection of the
    /// null (e.g. absolute mean deviation, chi-squared, |t-stat|).
    /// </summary>
    RightTail = 0,

    /// <summary>
    /// One-sided p-value = fraction of the distribution at or below the
    /// observed statistic. Use when small values indicate rejection of
    /// the null (e.g. testing for insufficient variance, one-sided
    /// alternatives in the negative direction).
    /// </summary>
    LeftTail = 1,

    /// <summary>
    /// Two-sided p-value <c>min(2·p_right, 2·p_left, 1)</c>. Use when
    /// either tail constitutes a rejection of the null and the caller
    /// is holding a raw (non-absolute) statistic. Exact only under
    /// distributional symmetry; on heavily skewed bootstrap
    /// distributions consider using <see cref="RightTail"/> with a
    /// caller-side <c>|·|</c> transformation instead.
    /// </summary>
    TwoSided = 2,
}

/// <summary>
/// Fast Double Bootstrap (FDB) of Davidson &amp; MacKinnon (2007). Corrects
/// the size distortion of bootstrap p-values without paying the cost of a
/// full nested double bootstrap. Uses one inner resample per outer
/// resample (rather than B nested resamples), giving ~B² cost instead of
/// the naive B³ — typically a 100×–1000× speedup over the textbook
/// double bootstrap with comparable bias correction.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Davidson, R. &amp; MacKinnon, J. G. (2007). "Improving the
/// Reliability of Bootstrap Tests with the Fast Double Bootstrap."
/// Computational Statistics &amp; Data Analysis, 51(7), 3259–3281.
/// </para>
/// <para>
/// Given an observed test statistic τ̂ and a way to resample the null
/// distribution, FDB:
/// <list type="number">
/// <item><description>Generates B outer bootstrap samples, computes τ̂_b on each.</description></item>
/// <item><description>For each outer sample, generates ONE inner resample and computes τ̂_b*.</description></item>
/// <item><description>Computes Q* = quantile of inner τ̂* at the rank of τ̂ in the outer distribution.</description></item>
/// <item><description>Returns the FDB p-value: rank of Q* in the outer distribution.</description></item>
/// </list>
/// </para>
/// <para>
/// The double-bootstrap correction is most useful for small samples where
/// the asymptotic null distribution of the test statistic is a poor
/// approximation. For large samples FDB and single bootstrap converge.
/// </para>
/// </remarks>
public sealed class FastDoubleBootstrap
{
    private readonly int _outerCount;
    private readonly IRandomSource _rng;

    /// <summary>Initializes the FDB engine with the given outer-loop count and random source.</summary>
    public FastDoubleBootstrap(int outerCount, IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (outerCount < 50)
        {
            throw new ArgumentOutOfRangeException(
                nameof(outerCount), outerCount, "Outer bootstrap count must be at least 50.");
        }

        _outerCount = outerCount;
        _rng = random;
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static FastDoubleBootstrap FromSeed(int outerCount, int? seed = null)
        => new(
            outerCount,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<double>.GetEntropySeed()));

    /// <summary>
    /// Computes the FDB p-value for an observed test statistic under the
    /// chosen tail convention.
    /// </summary>
    /// <param name="observations">Original sample.</param>
    /// <param name="observed">Observed test statistic τ̂.</param>
    /// <param name="statistic">Test statistic function (computed on each resampled path).</param>
    /// <param name="tail">
    /// Tail of the bootstrap distribution the p-value measures. Defaults
    /// to <see cref="PValueTail.RightTail"/> — the classical FDB convention
    /// (Davidson &amp; MacKinnon 2007 §3.1). See <see cref="PValueTail"/>
    /// for when to override.
    /// </param>
    /// <returns>FDB-corrected p-value in <c>[0, 1]</c>.</returns>
    /// <remarks>
    /// <para>
    /// Both the outer <c>p*</c> (rank of <paramref name="observed"/>) and
    /// the final <c>p</c> (rank of <c>q*</c> in the outer distribution) use
    /// the same <paramref name="tail"/>. Mixing conventions silently
    /// inverts the result, so the tail choice is exposed here rather than
    /// at the <c>ComputePValue</c> seam.
    /// </para>
    /// </remarks>
    public decimal PValue(
        decimal[] observations,
        decimal observed,
        Func<decimal[], decimal> statistic,
        PValueTail tail = PValueTail.RightTail)
    {
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentNullException.ThrowIfNull(statistic);
        if (observations.Length == 0)
        {
            throw new ArgumentException("Observations must contain at least one element.", nameof(observations));
        }

        var n = observations.Length;
        var outerStats = new decimal[_outerCount];
        var innerStats = new decimal[_outerCount];

        var outer = new decimal[n];
        var inner = new decimal[n];

        for (var b = 0; b < _outerCount; b++)
        {
            // Outer resample.
            for (var i = 0; i < n; i++)
            {
                outer[i] = observations[_rng.NextInt(n)];
            }

            outerStats[b] = statistic(outer);

            // Inner resample (one per outer).
            for (var i = 0; i < n; i++)
            {
                inner[i] = outer[_rng.NextInt(n)];
            }

            innerStats[b] = statistic(inner);
        }

        Array.Sort(outerStats);
        Array.Sort(innerStats);

        // Standard bootstrap p-value: rank of observed in outer.
        var pStar = ComputePValue(outerStats, observed, tail);

        // Q* = quantile of inner distribution at rank pStar.
        var qStar = Percentile.Compute(innerStats, pStar);

        // FDB p-value: rank of Q* in outer distribution under the same tail.
        return ComputePValue(outerStats, qStar, tail);
    }

    private static decimal ComputePValue(decimal[] sortedDistribution, decimal target, PValueTail tail)
    {
        // Count right-tail (>=) and left-tail (<=) in one pass. The two counts
        // overlap at ties (entries exactly equal to ``target`` are counted in
        // both), which is the correct convention for symmetric two-sided
        // p-values and matches the empirical-distribution convention used by
        // SciPy's ``bootstrap``.
        var rightCount = 0;
        var leftCount = 0;
        for (var i = 0; i < sortedDistribution.Length; i++)
        {
            if (sortedDistribution[i] >= target)
            {
                rightCount++;
            }

            if (sortedDistribution[i] <= target)
            {
                leftCount++;
            }
        }

        var length = (decimal)sortedDistribution.Length;
        var right = rightCount / length;
        var left = leftCount / length;

        return tail switch
        {
            PValueTail.RightTail => right,
            PValueTail.LeftTail => left,
            // min(2·min(right, left), 1) — the standard bilateral p-value
            // under symmetry. Capped at 1 to guard the symmetric-median case
            // where both tails include the median mass and round over.
            PValueTail.TwoSided => Math.Min(2m * Math.Min(right, left), 1m),
            _ => throw new ArgumentOutOfRangeException(
                nameof(tail), tail, "Unknown PValueTail value."),
        };
    }
}
