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
using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Random;

namespace Boutquin.Numerics.Statistics;

/// <summary>Lam's nonparametric eigenvalue-regularized covariance estimator (NERCOME).</summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> A+sqrt. Matrix arithmetic stays in T; Half inputs widen to double.</para>
/// <para>Assumes a known zero mean: each subset uses sum(x x-transpose)/subset size.
/// Each random split uses its own eigenbasis and the full reconstructed matrices are averaged
/// (Lam 2016, Annals of Statistics 44(3), 928–953, equations 4.5–4.6).
/// The default chooses the split by minimizing the squared Frobenius norm of the averaged
/// residual matrix (equation 4.8), over the seven candidates in equation 4.9.</para>
/// <para>Library policies: candidate sizes are floored, clamped to [1,n-1], then deduplicated
/// in published order; ties choose the first candidate. Fifty distinct first subsets are used
/// (the paper's simulation choice), capped at the number of possible subsets. A fresh
/// Xoshiro256** stream seeded with zero makes each call reproducible. Fisher–Yates permutations
/// are rejected when their first subset has already occurred. Explicit split fractions select
/// a fixed split size, still with replicated averaging. No asymptotic guarantee is asserted for
/// these finite-sample rounding or seed policies.</para>
/// <para>Raw observations are divided by a maximum absolute magnitude before estimation. Columns that
/// share a nonzero row are divided by one common maximum; columns that never share a nonzero row have an
/// exactly decoupled second-moment block in every split, so each such group of columns is divided by its own
/// maximum and a group far below the rest keeps its block instead of underflowing at the global scale. This
/// scale-only conditioning avoids overflow or underflow without centering observations.
/// Candidate comparison uses a scaled Frobenius norm of the averaged residual, with the same
/// minimizer as its square. A common headroom factor is retained while comparing candidates.
/// The returned covariance is rescaled in T.
/// Columns that share rows keep one common scale (the eigenvectors of D·M·D are not D times those of M), so
/// when a column's own maximum is so far below its group's that its second moments would fall below the full
/// precision of T at unit placement, every group maximum is placed instead at the largest power of two whose
/// square leaves the moments, projector sums and reconstruction finite; the first-subset moment is then a
/// graded matrix. Jacobi's relative-accuracy bounds depend on the scaled condition number of a
/// positive-definite block and on eigenvalue separation for its eigenvectors; singular subset moments
/// have no such guarantee. Decimal eigenvector entries near its absolute resolution can lose relative
/// digits, which can propagate through reconstruction. This placement is a common positive factor and
/// is undone, together with the group scales, on the returned entries.
/// Results outside the finite range of T throw <see cref="OverflowException"/>.</para>
/// </remarks>
/// <typeparam name="T">Floating-point type supporting arithmetic and the native square-root dispatcher.</typeparam>
public sealed class NercomeCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : struct, IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_fullPrecisionFloor = FullPrecisionFloor();
    private static readonly T s_maxValue = LargestFinite();
    private readonly bool _selectSplit;
    private readonly int _replications;
    private readonly ulong _seed;

    private readonly T _splitFraction;

    /// <summary>Creates a NERCOME estimator using the published seven-candidate split selector.</summary>
    public NercomeCovarianceEstimator()
        : this(null, 50, 0)
    {
    }

    /// <summary>
    /// Creates a NERCOME estimator with the specified split fraction.
    /// </summary>
    /// <param name="splitFraction">
    /// Fraction of rows allocated to the first half (eigenvector estimation).
    /// Must lie in (0, 1); <see langword="null"/> (including the <see langword="default"/> literal)
    /// selects the published seven-candidate grid.
    /// </param>
    public NercomeCovarianceEstimator(T? splitFraction)
        : this(splitFraction, 50, 0)
    {
    }

    /// <summary>Creates an estimator with a reproducible permutation policy.</summary>
    /// <param name="splitFraction">Fixed split fraction, or null for the published candidate grid.</param>
    /// <param name="replications">Positive number of distinct first subsets, capped by their total count.</param>
    /// <param name="seed">Xoshiro256** seed, reset at every estimation call.</param>
    public NercomeCovarianceEstimator(T? splitFraction, int replications, ulong seed)
    {
        var fraction = splitFraction ?? T.CreateChecked(0.5);

        if (!T.IsFinite(fraction) || fraction <= T.Zero || fraction >= T.One)
        {
            throw new ArgumentOutOfRangeException(
                nameof(splitFraction),
                fraction,
                "Split fraction must lie strictly between 0 and 1.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(replications);
        _splitFraction = fraction;
        _selectSplit = splitFraction is null;
        _replications = replications;
        _seed = seed;
    }

    /// <summary>Explicit split fraction, or 0.5 as a compatibility value when automatic selection is active.</summary>
    public T SplitFraction => _splitFraction;

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        if (typeof(T) == typeof(Half))
        {
            return EnsureFiniteCovariance(CovarianceHelpers<T>.EstimateHalfShrinkage(returns,
                new NercomeCovarianceEstimator<double>(_selectSplit ? null : double.CreateChecked(_splitFraction), _replications, _seed)));
        }

        var count = returns.GetLength(0);
        var dimension = returns.GetLength(1);
        var scale = T.Zero;
        foreach (var value in returns)
        {
            scale = T.Max(scale, T.Abs(value));
        }

        if (scale == T.Zero)
        {
            return new T[dimension, dimension];
        }

        var columnScale = ColumnScales(returns);
        var relativeScale = new T[dimension];
        var grouped = false;
        for (var j = 0; j < dimension; j++)
        {
            relativeScale[j] = columnScale[j] / scale;
            grouped |= relativeScale[j] != T.One;
        }

        var headroom = Headroom(returns, columnScale, dimension);
        var normalized = new T[count, dimension];
        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                normalized[i, j] = headroom is { } top
                    ? ToHeadroom(returns[i, j], columnScale[j], top)
                    : returns[i, j] / columnScale[j];
            }
        }

        returns = normalized;
        var candidates = _selectSplit
            ? CandidateSizes(count)
            : new[] { Math.Clamp(int.CreateChecked(_splitFraction * T.CreateChecked(count)), 1, count - 1) };
        IRandomSource random = new Xoshiro256StarStarRandomSource(_seed);
        T[,]? best = null;
        var bestScore = T.Zero;
        foreach (var size in candidates)
        {
            var (estimate, score) = headroom is not null || grouped
                ? AverageSplitsWeighted(returns, size, random, grouped ? relativeScale : null)
                : AverageSplits(returns, size, random);
            if (best is null || score < bestScore)
            {
                best = estimate;
                bestScore = score;
            }
        }

        // Each entry returns to the original units through the scales of its own two columns. With one
        // common scale this is the plain (entry . scale) . scale restore.
        var restored = new T[dimension, dimension];
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                restored[i, j] = headroom is { } top
                    ? FromHeadroom(FromHeadroom(best![i, j], columnScale[i], top), columnScale[j], top)
                    : (best![i, j] * columnScale[i]) * columnScale[j];
            }
        }

        return EnsureFiniteCovariance(restored);
    }

    /// <summary>
    /// Returns the magnitude at which to place each conditioning group's maximum when dividing by that
    /// maximum would leave some column's second moments below the full precision of T, or null when it
    /// does not. A column whose largest magnitude is r times its group's maximum has second moments of
    /// order r^2 at unit placement; columns that share rows cannot be conditioned separately (the
    /// eigenvectors of D·M·D are not D times those of M), so when r^2 falls below the smallest
    /// full-precision magnitude the group is placed at the largest power of two H for which every
    /// moment (at most H^2), the projector sums and the reconstruction (at most 4·N·H^2, the bound
    /// that keeps the eigensolver's unguarded rotation in range) stay finite. Placement is a common
    /// positive factor, so the estimator's degree-2 homogeneity returns the result unchanged.
    /// </summary>
    private static T? Headroom(T[,] returns, T[] columnScale, int dimension)
    {
        var count = returns.GetLength(0);
        var needed = false;
        for (var j = 0; j < dimension && !needed; j++)
        {
            var columnMaximum = T.Zero;
            for (var i = 0; i < count; i++)
            {
                columnMaximum = T.Max(columnMaximum, T.Abs(returns[i, j]));
            }

            var relative = columnMaximum / columnScale[j];
            needed = columnMaximum != T.Zero && relative * relative < s_fullPrecisionFloor;
        }

        if (!needed)
        {
            return null;
        }

        var limit = s_maxValue / T.CreateChecked(64 * Math.Max(dimension, 1));
        var top = T.One;
        while ((top * s_two) * (top * s_two) <= limit)
        {
            top *= s_two;
        }

        return top;
    }

    // value · top / scale. The forms keep every intermediate in range (|value| <= scale, so
    // |value · top| < top^2) and never round a small decimal value onto its absolute grid before
    // the multiplication.
    private static T ToHeadroom(T value, T scale, T top)
        => scale >= top ? value / (scale / top) : (value * top) / scale;

    // value · scale / top, the inverse placement. The ratio top / scale is used only while it is
    // finite in T (decimal throws past its range); below that the result is at the bottom of the range
    // and the ordered form (value / top) · scale keeps an IEEE value representable.
    private static T FromHeadroom(T value, T scale, T top)
    {
        if (scale >= top)
        {
            return value * (scale / top);
        }

        return scale >= top / s_maxValue ? value / (top / scale) : (value / top) * scale;
    }

    // Largest finite value of T. Decimal saturates to its maximum, but an IEEE conversion above the
    // target's range rounds to infinity, so the IEEE maxima are tried from the widest down.
    private static T LargestFinite()
    {
        foreach (var candidate in (double[])[double.MaxValue, float.MaxValue, (double)Half.MaxValue])
        {
            var value = T.CreateSaturating(candidate);
            if (T.IsFinite(value))
            {
                return value;
            }
        }

        return T.One;
    }

    // Smallest magnitude carrying the full precision of T: the smallest positive value over machine
    // epsilon (the smallest normal value for IEEE types, about one for decimal).
    private static T FullPrecisionFloor()
    {
        var epsilon = T.One;
        while (T.One + (epsilon / s_two) != T.One)
        {
            epsilon /= s_two;
        }

        var smallest = T.One;
        while (smallest / s_two != T.Zero)
        {
            smallest /= s_two;
        }

        return smallest / epsilon;
    }

    /// <summary>
    /// Per-column conditioning scale: the maximum absolute observation over the group of columns that are
    /// linked by sharing a nonzero row (an all-zero column forms its own group and uses one).
    /// </summary>
    private static T[] ColumnScales(T[,] returns)
    {
        var count = returns.GetLength(0);
        var dimension = returns.GetLength(1);
        var parent = new int[dimension];
        for (var j = 0; j < dimension; j++)
        {
            parent[j] = j;
        }

        int Find(int node)
        {
            while (parent[node] != node)
            {
                parent[node] = parent[parent[node]];
                node = parent[node];
            }

            return node;
        }

        for (var i = 0; i < count; i++)
        {
            var first = -1;
            for (var j = 0; j < dimension; j++)
            {
                if (returns[i, j] == T.Zero)
                {
                    continue;
                }

                if (first < 0)
                {
                    first = j;
                }
                else
                {
                    parent[Find(j)] = Find(first);
                }
            }
        }

        var groupScale = new T[dimension];
        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                var root = Find(j);
                groupScale[root] = T.Max(groupScale[root], T.Abs(returns[i, j]));
            }
        }

        var result = new T[dimension];
        for (var j = 0; j < dimension; j++)
        {
            var groupMaximum = groupScale[Find(j)];
            result[j] = groupMaximum == T.Zero ? T.One : groupMaximum;
        }

        return result;
    }

    private static T[,] EnsureFiniteCovariance(T[,] covariance)
    {
        foreach (var value in covariance)
        {
            if (!T.IsFinite(value))
            {
                throw new OverflowException("The estimated covariance is outside the finite range of the working type.");
            }
        }

        return covariance;
    }

    private static int[] CandidateSizes(int count)
    {
        var root = Math.Sqrt(count);
        double[] locations = [2 * root, .2 * count, .4 * count, .6 * count, .8 * count, count - 2.5 * root, count - 1.5 * root];
        return locations.Select(x => Math.Clamp((int)Math.Floor(x), 1, count - 1)).Distinct().ToArray();
    }

    // Unweighted entry: the selection score of observations that share one scale.
    private (T[,] Estimate, T Score) AverageSplits(T[,] observations, int size, IRandomSource random)
        => AverageSplitsWeighted(observations, size, random, null);

    private (T[,] Estimate, T Score) AverageSplitsWeighted(T[,] observations, int size, IRandomSource random, T[]? relativeScale)
    {
        var count = observations.GetLength(0);
        var dimension = observations.GetLength(1);
        var repetitions = _replications;
        // C(n,m), capped before overflow; only a bound up to the requested count is needed.
        var combinations = 1.0;
        for (var k = 1; k <= Math.Min(size, count - size) && combinations < repetitions; k++)
        {
            combinations *= (count - k + 1.0) / k;
        }

        repetitions = (int)Math.Min(repetitions, Math.Round(combinations));
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var sum = new T[dimension, dimension];
        var residual = new T[dimension, dimension];
        while (seen.Count < repetitions)
        {
            var order = Enumerable.Range(0, count).ToArray();
            for (var k = count - 1; k > 0; k--)
            {
                var other = random.NextInt(k + 1);
                (order[k], order[other]) = (order[other], order[k]);
            }

            var key = string.Join(",", order.Take(size).Order());
            if (!seen.Add(key))
            {
                continue;
            }

            var first = SecondMoment(observations, order, 0, size);
            var second = SecondMoment(observations, order, size, count - size);
            var vectors = JacobiEigenDecomposition<T>.Decompose(first).Vectors;
            var diagonal = new T[dimension];
            for (var k = 0; k < dimension; k++)
            {
                for (var i = 0; i < dimension; i++)
                {
                    for (var j = 0; j < dimension; j++)
                    {
                        diagonal[k] += vectors[i, k] * second[i, j] * vectors[j, k];
                    }
                }

                diagonal[k] = T.Max(T.Zero, diagonal[k]);
            }

            var reconstructed = CovarianceHelpers<T>.ReconstructFromEigen(diagonal, vectors);
            for (var i = 0; i < dimension; i++)
            {
                for (var j = 0; j < dimension; j++)
                {
                    sum[i, j] += reconstructed[i, j] / T.CreateChecked(repetitions);
                    residual[i, j] += (reconstructed[i, j] - second[i, j]) / T.CreateChecked(repetitions);
                }
            }
        }

        // Comparing the norm has the same minimizer as comparing its square. Keep the common
        // headroom factor: undoing it before squaring can turn every candidate into a false zero tie.
        // A group conditioned by its own maximum still enters at its true relative weight.
        var magnitude = T.Zero;
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                var weighted = relativeScale is null ? residual[i, j] : (residual[i, j] * relativeScale[i]) * relativeScale[j];
                residual[i, j] = weighted;
                magnitude = T.Max(magnitude, T.Abs(weighted));
            }

            for (var j = i + 1; j < dimension; j++)
            {
                sum[i, j] = sum[j, i] = (sum[i, j] + sum[j, i]) / s_two;
            }
        }

        var score = T.Zero;
        if (magnitude > T.Zero)
        {
            foreach (var entry in residual)
            {
                var relative = entry / magnitude;
                score += relative * relative;
            }

            score = magnitude * NumericPrecision<T>.Sqrt(score);
        }

        return (sum, score);
    }

    private static T[,] SecondMoment(T[,] observations, int[] order, int start, int count)
    {
        var dimension = observations.GetLength(1);
        var result = new T[dimension, dimension];
        for (var i = 0; i < dimension; i++)
        {
            for (var j = 0; j < dimension; j++)
            {
                for (var k = start; k < start + count; k++)
                {
                    result[i, j] += observations[order[k], i] * observations[order[k], j] / T.CreateChecked(count);
                }
            }
        }

        return result;
    }
}

/// <summary>
/// Decimal facade for Lam's nonparametric eigenvalue-regularized covariance estimator.
/// The parameterless constructor selects the published split grid; an explicit split
/// fraction selects replicated estimation at that split size. Both modes assume known
/// zero mean and average full covariance matrices in their own split-specific eigenbases.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Lam, C. (2016). "Nonparametric eigenvalue-regularized precision
/// or covariance matrix estimator." Annals of Statistics, 44(3), 928-953.
/// </para>
/// </remarks>
[Obsolete(
    "Use NercomeCovarianceEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class NercomeCovarianceEstimator : ICovarianceEstimator
{
    private readonly NercomeCovarianceEstimator<decimal> _inner;

    /// <summary>Creates a NERCOME estimator with automatic split selection.</summary>
    public NercomeCovarianceEstimator()
    {
        _inner = new NercomeCovarianceEstimator<decimal>();
    }

    /// <summary>
    /// Creates a NERCOME estimator with the specified split fraction.
    /// </summary>
    /// <param name="splitFraction">
    /// Fraction of rows allocated to the first half (eigenvector estimation).
    /// Must lie in (0, 1); defaults to 0.5.
    /// </param>
    public NercomeCovarianceEstimator(decimal splitFraction = 0.5m)
    {
        _inner = new NercomeCovarianceEstimator<decimal>(splitFraction);
    }

    /// <summary>Split fraction used to partition observations.</summary>
    public decimal SplitFraction => _inner.SplitFraction;

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
