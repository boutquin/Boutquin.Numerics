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
    /// One-sided p-value = fraction of the distribution strictly above the
    /// observed statistic. The classical FDB convention — appropriate
    /// whenever large values of the statistic indicate rejection of the
    /// null (e.g. absolute mean deviation, chi-squared, |t-stat|).
    /// </summary>
    RightTail = 0,

    /// <summary>
    /// One-sided p-value = fraction of the distribution strictly below the
    /// observed statistic. Use when small values indicate rejection of
    /// the null (e.g. testing for insufficient variance, one-sided
    /// alternatives in the negative direction).
    /// </summary>
    LeftTail = 1,

    /// <summary>
    /// Two-sided p-value <c>min(2·p_right, 2·p_left, 1)</c>, folding the two
    /// one-sided FDB p-values. Use when
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
/// resample (rather than B nested resamples). With B outer draws, sample
/// length n, and statistic cost Cstat(n), it performs 2B statistic calls;
/// total work is O(Bn + B·Cstat(n) + B log B), including two sorts.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Davidson, R. &amp; MacKinnon, J. G. (2007). "Improving the
/// Reliability of Bootstrap Tests with the Fast Double Bootstrap."
/// Computational Statistics &amp; Data Analysis, 51(7), 3259–3281.
/// </para>
/// <para>
/// Given an observed test statistic τ̂ and a bootstrap DGP that respects the null
/// hypothesis, FDB:
/// <list type="number">
/// <item><description>Generates B outer bootstrap samples from the null-imposed DGP, computes τ̂_b on each.</description></item>
/// <item><description>For each outer sample, generates ONE inner resample from the null-imposed DGP estimated on that outer sample and computes τ̂_b*.</description></item>
/// <item><description>Computes the single-bootstrap p-value p̂₁ = rank of τ̂ in the outer distribution.</description></item>
/// <item><description>Computes Q* = a quantile of the inner τ̂*: the <c>(1 − p̂₁)</c>-quantile for a right-rejecting test, the <c>p̂₁</c>-quantile for a left-rejecting test.</description></item>
/// <item><description>Returns the FDB p-value: rank of Q* in the outer distribution under the same tail.</description></item>
/// </list>
/// </para>
/// <para>
/// <see cref="PValueImposingNull"/> imposes the null at both levels through a
/// caller-supplied transform. <see cref="PValue"/> resamples the observations and each
/// outer sample as they are, which is the FDB only when the statistic is invariant to
/// the parameters the null restricts (Davidson 2023 uses plain resampling for exactly
/// that reason). A location statistic such as a mean is not invariant: centring the data
/// once (or centring the statistic at the full-sample mean) and then drawing each inner
/// sample from the un-recentred outer sample rejected about 17% of the time at nominal 5%
/// for |mean| on N(0, 1) samples of size 20.
/// </para>
/// <para>
/// The double-bootstrap correction is most useful for small samples where
/// the asymptotic null distribution of the test statistic is a poor
/// approximation. For large samples FDB and single bootstrap converge.
/// </para>
/// <para>
/// Empirical tail counts exclude equality at both ranks: right tails count
/// statistics strictly greater than the observed statistic or inner quantile;
/// left tails count statistics strictly less than those thresholds. This follows
/// Davidson (2023), §4, algorithm steps 4 and 6 and its right-rejecting remark.
/// Tied-data p-values therefore differ from earlier releases, which included ties.
/// Inner quantiles retain linear interpolation (Type 7).
/// </para>
/// <para>
/// The supplied <see cref="IRandomSource"/> is consumed sequentially; do not
/// share a single source instance across engine instances or threads. See
/// <see cref="IRandomSource"/> for thread-safety semantics.
/// </para>
/// <para>
/// <b>Working type: <see cref="decimal"/>.</b> Observations, the observed
/// statistic, and the returned p-value are all <see cref="decimal"/> — this is a
/// non-generic decimal-only engine (there is no <c>double</c> overload). Quasi-MC
/// helpers elsewhere in this namespace work in <see cref="double"/>; do not assume
/// a shared precision regime across the MonteCarlo facades.
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
    /// <param name="observations">
    /// Original sample, resampled as-is at the outer level; each outer sample is resampled
    /// as-is at the inner level. Use this overload only for a statistic whose null
    /// distribution does not depend on the parameters the null restricts; otherwise call
    /// <see cref="PValueImposingNull"/>.
    /// </param>
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
    /// <para>
    /// <c>q*</c> is the <c>(1 − p*)</c>-quantile of the inner distribution for
    /// <see cref="PValueTail.RightTail"/> and the <c>p*</c>-quantile for
    /// <see cref="PValueTail.LeftTail"/>. <see cref="PValueTail.TwoSided"/>
    /// returns <c>min(2·min(p_right, p_left), 1)</c> of the two one-sided FDB
    /// p-values.
    /// </para>
    /// </remarks>
    public decimal PValue(
        ReadOnlySpan<decimal> observations,
        decimal observed,
        Func<ReadOnlySpan<decimal>, decimal> statistic,
        PValueTail tail = PValueTail.RightTail)
        => Compute(observations, observed, statistic, imposeNull: null, tail);

    /// <summary>
    /// Computes the FDB p-value with the null hypothesis imposed on the bootstrap DGP at
    /// both levels, as the fast double bootstrap requires for a statistic that is not
    /// invariant to the parameters the null restricts.
    /// </summary>
    /// <param name="observations">Original sample.</param>
    /// <param name="observed">Observed test statistic τ̂, computed on <paramref name="observations"/>.</param>
    /// <param name="statistic">Test statistic function (computed on each drawn sample as drawn).</param>
    /// <param name="imposeNull">
    /// Transforms a sample, in place, into the support of the null-respecting DGP estimated
    /// from it — for example, recentring to the hypothesised mean for a location test. It is
    /// applied to a copy of <paramref name="observations"/> (the outer DGP) and to a copy of
    /// every outer sample before its inner sample is drawn (the inner DGP). It must not change
    /// the span's length and is never applied to the caller's array or to the samples the
    /// statistic sees.
    /// </param>
    /// <param name="tail">Tail of the bootstrap distribution the p-value measures; see <see cref="PValue"/>.</param>
    /// <returns>FDB-corrected p-value in <c>[0, 1]</c>.</returns>
    /// <remarks>
    /// Davidson (2023, "An Improved Fast Double Bootstrap", §4) draws the outer samples from a
    /// null-imposed bootstrap DGP and each inner sample from the null-imposed DGP estimated on
    /// that outer sample. For H₀: μ = μ₀ with the statistic <c>|mean − μ₀|</c>, pass a transform
    /// that shifts a sample to mean μ₀. On N(0, 1) samples of size 20 under H₀: μ = 0 this
    /// rejects about 5–6% of the time at nominal 5%, where drawing the inner sample from the
    /// un-recentred outer sample rejects about 17%.
    /// </remarks>
    public decimal PValueImposingNull(
        ReadOnlySpan<decimal> observations,
        decimal observed,
        Func<ReadOnlySpan<decimal>, decimal> statistic,
        Action<Span<decimal>> imposeNull,
        PValueTail tail = PValueTail.RightTail)
    {
        ArgumentNullException.ThrowIfNull(imposeNull);
        return Compute(observations, observed, statistic, imposeNull, tail);
    }

    private decimal Compute(
        ReadOnlySpan<decimal> observations,
        decimal observed,
        Func<ReadOnlySpan<decimal>, decimal> statistic,
        Action<Span<decimal>>? imposeNull,
        PValueTail tail)
    {
        ArgumentNullException.ThrowIfNull(statistic);
        if (observations.Length == 0)
        {
            throw new ArgumentException("Observations must contain at least one element.", nameof(observations));
        }

        if (!Enum.IsDefined(tail))
        {
            throw new ArgumentOutOfRangeException(nameof(tail), tail, "Unknown PValueTail value.");
        }

        var n = observations.Length;
        var outerStats = new decimal[_outerCount];
        var innerStats = new decimal[_outerCount];

        var outer = new decimal[n];
        var inner = new decimal[n];

        // Outer DGP: the observations, or their null-imposed transform.
        ReadOnlySpan<decimal> outerPopulation = observations;
        decimal[]? innerPopulation = null;
        if (imposeNull is not null)
        {
            var nullData = observations.ToArray();
            imposeNull(nullData);
            outerPopulation = nullData;
            innerPopulation = new decimal[n];
        }

        for (var b = 0; b < _outerCount; b++)
        {
            // Outer resample.
            for (var i = 0; i < n; i++)
            {
                outer[i] = outerPopulation[_rng.NextInt(n)];
            }

            outerStats[b] = statistic(outer.AsSpan());

            // Inner DGP: the outer sample, or the null-imposed DGP estimated from it.
            var innerSource = outer;
            if (innerPopulation is not null)
            {
                outer.CopyTo(innerPopulation, 0);
                imposeNull!(innerPopulation);
                innerSource = innerPopulation;
            }

            // Inner resample (one per outer).
            for (var i = 0; i < n; i++)
            {
                inner[i] = innerSource[_rng.NextInt(n)];
            }

            innerStats[b] = statistic(inner.AsSpan());
        }

        Array.Sort(outerStats);
        Array.Sort(innerStats);

        if (tail == PValueTail.TwoSided)
        {
            // Equal-tail fold of the two one-sided FDB p-values; each side takes
            // its own quantile of the inner distribution.
            var right = OneSidedPValue(outerStats, innerStats, observed, PValueTail.RightTail);
            var left = OneSidedPValue(outerStats, innerStats, observed, PValueTail.LeftTail);
            return Math.Min(2m * Math.Min(right, left), 1m);
        }

        return OneSidedPValue(outerStats, innerStats, observed, tail);
    }

    private static decimal OneSidedPValue(
        decimal[] sortedOuter, decimal[] sortedInner, decimal observed, PValueTail tail)
    {
        // Keep the integer empirical count through the Type-7 index. Even a
        // decimal k/B can round before amplification when B has other prime factors.
        var n = sortedInner.Length;
        var numerator = tail == PValueTail.RightTail
            ? UpperBound(sortedOuter, observed) : LowerBound(sortedOuter, observed);
        var rank = (long)numerator * (n - 1);
        var lower = (int)(rank / n);
        var remainder = rank % n;
        var upper = Math.Min(lower + 1, n - 1);
        var low = sortedInner[lower];
        var high = sortedInner[upper];
        decimal qStar;
        try
        {
            qStar = low + (high - low) * remainder / n;
        }
        catch (OverflowException)
        {
            // Outside the central range the gap or weighted numerator may exceed
            // decimal even though the convex result fits.
            var fraction = (decimal)remainder / n;
            qStar = (1m - fraction) * low + fraction * high;
        }

        // FDB p-value: rank of Q* in outer distribution under the same tail.
        return ComputePValue(sortedOuter, qStar, tail);
    }

    internal static decimal ComputePValue(decimal[] sortedDistribution, decimal target, PValueTail tail)
    {
        // The distribution is already sorted ascending, so the two tail counts are
        // located with binary search instead of a linear pass:
        //   rightCount = #{x > target} = length - UpperBound(target)
        //   leftCount  = #{x < target} = LowerBound(target)
        // Exclude equality at both ranks, as in Davidson (2023), §4,
        // algorithm steps 4/6 and the right-rejecting remark.
        var n = sortedDistribution.Length;
        var rightCount = n - UpperBound(sortedDistribution, target);
        var leftCount = LowerBound(sortedDistribution, target);

        var length = (decimal)n;
        var right = rightCount / length;
        var left = leftCount / length;

        return tail switch
        {
            PValueTail.RightTail => right,
            PValueTail.LeftTail => left,
            // min(2·min(right, left), 1) — the standard bilateral p-value
            // under symmetry. Equality is excluded from both tails.
            PValueTail.TwoSided => Math.Min(2m * Math.Min(right, left), 1m),
            _ => throw new ArgumentOutOfRangeException(
                nameof(tail), tail, "Unknown PValueTail value."),
        };
    }

    /// <summary>
    /// Index of the first element of the ascending-sorted <paramref name="sorted"/> that is
    /// <c>&gt;= target</c> — equivalently the count of elements strictly less than
    /// <paramref name="target"/>. Resolves ties by walking back to the first equal element,
    /// so the result is stable regardless of which duplicate
    /// <see cref="Array.BinarySearch{T}(T[], T)"/> lands on.
    /// </summary>
    private static int LowerBound(decimal[] sorted, decimal target)
    {
        var idx = Array.BinarySearch(sorted, target);
        if (idx < 0)
        {
            // Not found: ~idx is the insertion point = index of the first element greater
            // than target = count of elements less than target (none equal it).
            return ~idx;
        }

        // Found: BinarySearch may land on any element equal to target. Walk to the first.
        while (idx > 0 && sorted[idx - 1] == target)
        {
            idx--;
        }

        return idx;
    }

    /// <summary>
    /// Index one past the last element of the ascending-sorted <paramref name="sorted"/> that
    /// is <c>&lt;= target</c> — equivalently the count of elements less than or equal to
    /// <paramref name="target"/>. Resolves ties by walking past the last equal element.
    /// </summary>
    private static int UpperBound(decimal[] sorted, decimal target)
    {
        var idx = Array.BinarySearch(sorted, target);
        if (idx < 0)
        {
            // Not found: ~idx is the insertion point = index of the first element greater
            // than target = count of elements <= target (none equal it).
            return ~idx;
        }

        // Found: walk past every element equal to target to reach the first strictly-greater.
        while (idx < sorted.Length && sorted[idx] == target)
        {
            idx++;
        }

        return idx;
    }
}
