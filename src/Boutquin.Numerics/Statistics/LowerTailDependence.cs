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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Result of a lower-tail-dependence estimation.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
/// <param name="Lambda">
/// Estimated lower-tail-dependence coefficient at the primary (first) quantile.
/// This finite-threshold sample estimate does not establish the population limit
/// as the quantile approaches zero. In particular, a zero estimate can reflect no
/// observations below the chosen threshold and does not establish asymptotic independence.
/// </param>
/// <param name="Threshold">
/// Pseudo-observation thresholds used for each quantile level.
/// Length equals the number of quantiles in <see cref="TailOptions{T}"/>.
/// The constructor snapshots the input, and each property access returns a copy.
/// </param>
/// <param name="JointCount">
/// Number of paired observations that jointly fell below each threshold.
/// Parallel to <see cref="Threshold"/>. The constructor snapshots the input,
/// and each property access returns a copy.
/// </param>
public sealed record TailResult<T>(T Lambda, T[] Threshold, int[] JointCount)
    where T : IFloatingPoint<T>
{
    private T[] _threshold = Snapshot(Threshold);
    private int[] _jointCount = Snapshot(JointCount);

    /// <summary>
    /// Pseudo-observation thresholds used for each quantile level, returned as a
    /// fresh array so callers cannot mutate the completed result.
    /// </summary>
    public T[] Threshold
    {
        get => (T[])_threshold.Clone();
        init => _threshold = Snapshot(value);
    }

    /// <summary>
    /// Joint lower-tail observation counts parallel to <see cref="Threshold"/>,
    /// returned as a fresh array so callers cannot mutate the completed result.
    /// </summary>
    public int[] JointCount
    {
        get => (int[])_jointCount.Clone();
        init => _jointCount = Snapshot(value);
    }

    private static TElement[] Snapshot<TElement>(TElement[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return (TElement[])values.Clone();
    }
}

/// <summary>
/// Selects a finite-threshold lower-tail summary. Both members use the joint
/// exceedance count and differ in its denominator: the realized marginal count
/// or <c>n·q</c>. Neither summary alone establishes a population tail limit or
/// a consistency result as the caller's threshold approaches zero.
/// </summary>
public enum LowerTailDependenceEstimator
{
    /// <summary>
    /// Conditional estimator
    /// <c>λ̂(q) = #{u_i ≤ q AND v_i ≤ q} / #{u_i ≤ q}</c> — the joint-exceedance
    /// count divided by the <em>realized</em> marginal-exceedance count; the library
    /// default. At a fixed <c>q</c> this is a conditional sample proportion,
    /// not a consistency guarantee for the population lower-tail limit. It is
    /// not the finite upper-tail χ(u) logarithmic ratio of Coles, Heffernan &amp;
    /// Tawn (1999), §3.3.1.
    /// </summary>
    Conditional = 0,

    /// <summary>
    /// Joe (1997) / Nelsen (2006) empirical-copula estimator
    /// <c>λ̂(q) = #{u_i ≤ q AND v_i ≤ q} / (n·q)</c> — the joint-exceedance count
    /// divided by the <em>expected</em> marginal-exceedance count <c>n·q</c>.
    /// References: Joe (1997), <i>Multivariate Models and Dependence Concepts</i>;
    /// Nelsen (2006), <i>An Introduction to Copulas</i>, 2nd ed.
    /// </summary>
    EmpiricalCopula = 1,
}

/// <summary>
/// Options controlling lower-tail-dependence estimation.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
/// <param name="Quantiles">
/// Quantile levels at which to estimate tail dependence. Defaults to
/// <c>[0.05, 0.10, 0.15, 0.20]</c> when <see langword="null"/>.
/// </param>
/// <param name="UseGarchFilter">
/// When <see langword="true"/>, the series would be pre-whitened by a
/// GARCH(1,1) filter before rank-transformation. GARCH pre-whitening is not
/// yet implemented: <see cref="LowerTailDependence{T}.Estimate"/> and
/// <see cref="LowerTailDependence{T}.EstimateMatrix"/> throw
/// <see cref="NotSupportedException"/> when this is <see langword="true"/>,
/// rather than silently returning the unfiltered estimate. Leave at the
/// default <see langword="false"/> until pre-whitening ships.
/// </param>
/// <param name="Estimator">
/// Finite-sample estimator selection. Defaults to
/// <see cref="LowerTailDependenceEstimator.Conditional"/> (the CHT joint/marginal
/// form), so existing callers' results are unchanged. See
/// <see cref="LowerTailDependenceEstimator"/> for the alternative empirical-copula
/// form and the citations for each.
/// </param>
public sealed record TailOptions<T>(
    T[]? Quantiles = null,
    bool UseGarchFilter = false,
    LowerTailDependenceEstimator Estimator = LowerTailDependenceEstimator.Conditional)
    where T : IFloatingPoint<T>
{
    private T[]? _quantiles = Quantiles is null ? null : (T[])Quantiles.Clone();

    /// <summary>Quantile levels returned as a copy of the stored values.</summary>
    public T[]? Quantiles
    {
        get => _quantiles is null ? null : (T[])_quantiles.Clone();
        init => _quantiles = value is null ? null : (T[])value.Clone();
    }
}

/// <summary>
/// Empirical lower-tail-dependence estimator based on pseudo-observations
/// (empirical copula approach).
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{T}"/>. No transcendental functions
/// are required.
/// </para>
/// <para>
/// The estimator converts each series to pseudo-observations u = r/(n+1) where
/// r is the ascending rank (1..n; tied values receive the average rank). For a
/// given quantile level q, two finite-threshold summaries are selectable through
/// <see cref="TailOptions{T}.Estimator"/>. They share the joint count J and
/// differ in their denominator:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///     <b>Conditional (default,
///     <see cref="LowerTailDependenceEstimator.Conditional"/>):</b>
///     <c>λ̂(q) = J / #{u_i ≤ q}</c>. This conditions on the first series and
///     can differ when the two series are exchanged, especially with ties.
///     An empty conditioning set returns zero as a local boundary policy.
///     </description>
///   </item>
///   <item>
///     <description>
///     <b>EmpiricalCopula
///     (<see cref="LowerTailDependenceEstimator.EmpiricalCopula"/>):</b>
///     <c>λ̂(q) = J / (n·q)</c>. This summary is symmetric but can exceed one
///     on the finite rank grid; it is not clipped. The average-rank grid
///     <c>r/(n+1)</c> is this library's convention, distinct from the
///     <c>i/n</c> empirical-copula grid in Nelsen (2006).
///     </description>
///   </item>
/// </list>
/// <para>
/// Nelsen (2006), <i>An Introduction to Copulas</i>, 2nd ed., Definition 5.4.1
/// and Theorem 5.4.2, defines the population lower-tail limit using a
/// conditional probability and, for continuous uniform margins, C(q,q)/q.
/// These finite-threshold summaries do not by themselves establish that limit
/// or a consistency result. Coles, Heffernan &amp; Tawn (1999), §3.3.1,
/// concerns upper-tail dependence; its finite χ(u) is a logarithmic ratio
/// and is not either formula implemented here.
/// </para>
/// <para>
/// With a nonempty conditioning set, the two summaries coincide when its
/// realized count equals n·q. They can also coincide when J is zero.
/// The value at the primary quantile (index 0) is returned as
/// <see cref="TailResult{T}.Lambda"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class LowerTailDependence<T>
    where T : IFloatingPoint<T>
{
    private static readonly T[] s_defaultQuantiles =
    [
        T.CreateChecked(0.05),
        T.CreateChecked(0.10),
        T.CreateChecked(0.15),
        T.CreateChecked(0.20),
    ];

    /// <summary>
    /// Estimates the lower-tail-dependence coefficient between two series.
    /// </summary>
    /// <param name="x">First series of raw observations.</param>
    /// <param name="y">Second series of raw observations (must be the same length).</param>
    /// <param name="options">Estimation options; defaults are used when <see langword="null"/>.</param>
    /// <returns>
    /// A <see cref="TailResult{T}"/> containing the primary lambda and
    /// per-quantile threshold / joint-count arrays.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="x"/> and <paramref name="y"/> have different
    /// lengths, when fewer than 4 observations are provided, or when
    /// <paramref name="options"/> supplies an empty <c>Quantiles</c> array.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A quantile is non-finite or outside the open interval (0, 1), or the estimator
    /// enumeration value is undefined.
    /// </exception>
    public static TailResult<T> Estimate(
        ReadOnlySpan<T> x,
        ReadOnlySpan<T> y,
        TailOptions<T>? options = null)
    {
        ValidateOptions(options);

        if (x.Length != y.Length)
        {
            throw new ArgumentException(
                "x and y must have the same length.",
                nameof(y));
        }

        if (x.Length < 4)
        {
            throw new ArgumentException(
                $"Insufficient observations: at least 4 required, got {x.Length}.",
                nameof(x));
        }

        for (var i = 0; i < x.Length; i++)
        {
            if (!T.IsFinite(x[i]))
            {
                throw new ArgumentException(
                    "x must contain only finite values (no NaN or infinity).",
                    nameof(x));
            }

            if (!T.IsFinite(y[i]))
            {
                throw new ArgumentException(
                    "y must contain only finite values (no NaN or infinity).",
                    nameof(y));
            }
        }

        var quantiles = options?.Quantiles ?? s_defaultQuantiles;

        var n = x.Length;

        // Compute pseudo-observations: u = rank / (n+1). The (n+1) divisor is
        // formed inside ComputePseudoObservations rather than here, because
        // for T = Half, T.CreateChecked(n + 1) silently becomes +Infinity
        // once n exceeds 65,504 (F-L1-03 / F-L5-02).
        var u = ComputePseudoObservations(x, n);
        var v = ComputePseudoObservations(y, n);

        var thresholds = new T[quantiles.Length];
        var jointCounts = new int[quantiles.Length];

        for (var qi = 0; qi < quantiles.Length; qi++)
        {
            var q = quantiles[qi];
            thresholds[qi] = q;

            var marginalCount = 0;
            var jointCount = 0;
            for (var i = 0; i < n; i++)
            {
                if (u[i] <= q)
                {
                    marginalCount++;
                    if (v[i] <= q)
                    {
                        jointCount++;
                    }
                }
            }

            jointCounts[qi] = jointCount;
        }

        // Lambda at primary quantile (index 0). The two estimators share the joint
        // exceedance count (numerator) and differ only in the denominator:
        //   Conditional (CHT):     λ̂ = joint / #{u_i ≤ q}   (realized marginal count)
        //   EmpiricalCopula (Joe): λ̂ = joint / (n·q)         (expected marginal count)
        var primaryQ = quantiles[0];
        var primaryMarginal = 0;
        for (var i = 0; i < n; i++)
        {
            if (u[i] <= primaryQ)
            {
                primaryMarginal++;
            }
        }

        var estimator = options?.Estimator ?? LowerTailDependenceEstimator.Conditional;
        T lambda;
        if (typeof(T) == typeof(Half))
        {
            // Both counts can exceed Half's finite range. Convert only the final
            // ratio, retaining the integer counts until division in double.
            var denominator = estimator == LowerTailDependenceEstimator.EmpiricalCopula
                ? n * double.CreateChecked(primaryQ)
                : primaryMarginal;
            lambda = denominator == 0
                ? T.Zero
                : T.CreateChecked(jointCounts[0] / denominator);
        }
        else if (estimator == LowerTailDependenceEstimator.EmpiricalCopula)
        {
            // Local finite-rank empirical-copula summary: denominator n·q.
            var expectedMarginal = T.CreateChecked(n) * primaryQ;
            lambda = expectedMarginal == T.Zero
                ? T.Zero
                : T.CreateChecked(jointCounts[0]) / expectedMarginal;
        }
        else
        {
            // Conditional summary: denominator is the first marginal count.
            lambda = primaryMarginal == 0
                ? T.Zero
                : T.CreateChecked(jointCounts[0]) / T.CreateChecked(primaryMarginal);
        }

        return new TailResult<T>(lambda, thresholds, jointCounts);
    }

    /// <summary>
    /// Estimates a lower-tail-dependence matrix for a collection of series.
    /// Diagonal entries are set to 1 by matrix policy, rather than evaluating
    /// the finite-threshold self-pair summary. Off-diagonal entries are the pairwise
    /// <see cref="TailResult{T}.Lambda"/> values. The default conditional estimator
    /// is directional when ties produce different realized marginal counts, so
    /// entry (i, j) conditions on column i and entry (j, i) on column j.
    /// The empirical-copula estimator remains symmetric.
    /// </summary>
    /// <param name="columns">
    /// Array of series. All series must have the same length.
    /// </param>
    /// <param name="options">Options applied to each pairwise estimate.</param>
    /// <returns>A <c>k × k</c> matrix of lambda values.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="columns"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="columns"/> is empty, contains a null, non-finite,
    /// or too-short series, or when any series has a different length from the first.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A quantile is non-finite or outside the open interval (0, 1), or the estimator
    /// enumeration value is undefined.
    /// </exception>
    public static T[,] EstimateMatrix(T[][] columns, TailOptions<T>? options = null)
    {
        ArgumentNullException.ThrowIfNull(columns);
        ValidateOptions(options);

        var k = columns.Length;
        if (k == 0)
        {
            throw new ArgumentException(
                "columns must contain at least one series.",
                nameof(columns));
        }

        var expectedLength = -1;
        for (var columnIndex = 0; columnIndex < k; columnIndex++)
        {
            var column = columns[columnIndex];
            if (column is null)
            {
                throw new ArgumentException(
                    $"Column at index {columnIndex} must not be null.", nameof(columns));
            }

            if (expectedLength < 0)
            {
                expectedLength = column.Length;
                if (expectedLength < 4)
                {
                    throw new ArgumentException(
                        $"Insufficient observations: at least 4 required, got {expectedLength}.",
                        nameof(columns));
                }
            }
            else if (column.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"All columns must have the same length; column 0 has {expectedLength} " +
                    $"observations but column {columnIndex} has {column.Length}.",
                    nameof(columns));
            }

            for (var rowIndex = 0; rowIndex < column.Length; rowIndex++)
            {
                if (!T.IsFinite(column[rowIndex]))
                {
                    throw new ArgumentException(
                        $"Column {columnIndex} entry at index {rowIndex} must be finite.",
                        nameof(columns));
                }
            }
        }

        var result = new T[k, k];

        for (var i = 0; i < k; i++)
        {
            result[i, i] = T.One;
            for (var j = i + 1; j < k; j++)
            {
                result[i, j] = Estimate(columns[i], columns[j], options).Lambda;
                result[j, i] = Estimate(columns[j], columns[i], options).Lambda;
            }
        }

        return result;
    }

    private static void ValidateOptions(TailOptions<T>? options)
    {
        if (options?.UseGarchFilter == true)
        {
            throw new NotSupportedException(
                "UseGarchFilter is not yet supported. Passing UseGarchFilter = true previously " +
                "resulted in the option being silently ignored (the unfiltered estimate was " +
                "returned); estimation now fails fast instead of silently degrading.");
        }

        if (options is null)
        {
            return;
        }

        if (!Enum.IsDefined(typeof(LowerTailDependenceEstimator), options.Estimator))
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), options.Estimator, "Estimator is not a defined value.");
        }

        var quantiles = options.Quantiles;
        if (quantiles is null)
        {
            return;
        }

        if (quantiles.Length == 0)
        {
#pragma warning disable CA2208 // Name the offending nested property for caller diagnostics.
            throw new ArgumentException(
                "options.Quantiles must contain at least one quantile.",
                nameof(TailOptions<T>.Quantiles));
#pragma warning restore CA2208
        }

        for (var i = 0; i < quantiles.Length; i++)
        {
            var quantile = quantiles[i];
            if (!T.IsFinite(quantile) || quantile <= T.Zero || quantile >= T.One)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(options), quantile,
                    $"Quantile at index {i} must be finite and strictly between 0 and 1.");
            }
        }
    }

    private static readonly T s_two = T.CreateChecked(2);

    // Sum of the first and last 1-based ranks of the tie group [start, end]. Formed in long: for a series of
    // 2^30 or more observations the top-ranked group's sum reaches 2^31 and wraps in int.
    internal static long RankSum(int start, int end) => (long)start + end + 2;

    private static T[] ComputePseudoObservations(ReadOnlySpan<T> series, int n)
    {
        // Sort indices by ascending value to assign ranks. Tied values receive the
        // AVERAGE of the ranks they span (mirrors RankCorrelation.AverageRanks), so
        // the pseudo-observation is invariant to the arbitrary ordering of equal
        // keys — preventing a tied block from arbitrarily straddling a quantile.
        var indexed = new (T Value, int Index)[n];
        for (var i = 0; i < n; i++)
        {
            indexed[i] = (series[i], i);
        }

        Array.Sort(indexed, (a, b) => a.Value.CompareTo(b.Value));

        // Half cannot represent n + 1 (nor the rank sums start+end+2) once n
        // exceeds 65,504 — T.CreateChecked silently returns +Infinity rather
        // than throwing. Every rank at or below that range would then map to
        // u = 0/Infinity = 0 (a false lower-tail exceedance at every
        // quantile), and every rank above it to Infinity/Infinity = NaN
        // (F-L1-03 / F-L5-02). Route the ratio itself through double, which
        // represents every rank and count used here exactly, and round only
        // the final ratio.
        var isHalf = typeof(T) == typeof(Half);

        var pseudoObs = new T[n];
        var start = 0;
        while (start < n)
        {
            var end = start;
            while (end + 1 < n && indexed[end + 1].Value == indexed[start].Value)
            {
                end++;
            }

            // Average 1-based rank for the tie group [start, end].
            T u;
            if (isHalf)
            {
                var avgRankDouble = RankSum(start, end) / 2.0;
                u = T.CreateChecked(avgRankDouble / (n + 1));
            }
            else
            {
                var avgRank = T.CreateChecked(RankSum(start, end)) / s_two;
                u = avgRank / T.CreateChecked(n + 1);
            }

            for (var k = start; k <= end; k++)
            {
                pseudoObs[indexed[k].Index] = u;
            }

            start = end + 1;
        }

        return pseudoObs;
    }
}
