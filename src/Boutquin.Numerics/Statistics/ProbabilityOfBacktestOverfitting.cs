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
/// Result of a Combinatorially-Symmetric Cross-Validation (CSCV) run for
/// the Probability of Backtest Overfitting.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="Pbo">Estimated probability that the in-sample best strategy underperforms the median out-of-sample.</param>
/// <param name="LogitMedian">Median logit of OOS rank — the central tendency of the overfitting distribution.</param>
/// <param name="LogitValues">Per-fold logit values; the empirical distribution behind the PBO.</param>
public sealed record PboResult<T>(T Pbo, T LogitMedian, IReadOnlyList<T> LogitValues)
    where T : IFloatingPoint<T>
{
    private IReadOnlyList<T> _logitValues = Snapshot(LogitValues);

    /// <summary>
    /// Per-fold logit values in ascending order. Constructor and initializer inputs
    /// are snapshotted, and the returned view cannot be cast to a mutable array.
    /// </summary>
    public IReadOnlyList<T> LogitValues
    {
        get => _logitValues;
        init => _logitValues = Snapshot(value);
    }

    private static IReadOnlyList<T> Snapshot(IReadOnlyList<T> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return Array.AsReadOnly(values.ToArray());
    }
}

/// <summary>
/// Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
/// Cross-Validation (CSCV). Splits a panel of <c>N</c> strategy return series
/// × <c>T</c> time observations into <c>S</c> balanced blocks and exhausts all
/// <c>C(S, S/2)</c> half-half splits. For each split the S/2 training blocks are
/// joined into one in-sample set and the rest into the out-of-sample set, the
/// Sharpe ratio is computed on each joined set, and the in-sample winner is
/// identified; PBO is the proportion of splits in which that winner ranks at or
/// below the out-of-sample median.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H., Borwein, J. M., López de Prado, M. &amp; Zhu,
/// Q. J. (2014). "The Probability of Backtest Overfitting." Journal of
/// Computational Finance, 20(4), 39–70.
/// </para>
/// <para>
/// PBO complements the Deflated Sharpe Ratio: DSR adjusts a single observed
/// Sharpe for the search process that produced it, while PBO measures the
/// generalization rate of the search procedure itself. <c>PBO &gt; 0.5</c>
/// signals that the search is anti-predictive — picking the in-sample winner
/// is worse than random selection out-of-sample.
/// </para>
/// <para>
/// Logit aggregation: the reported <c>Pbo</c> is the empirical probability
/// that the winner's OOS rank falls below the median, but <c>LogitMedian</c>
/// and <c>LogitValues</c> are computed on the per-fold logit
/// <c>ln(r / (1 − r))</c> of the rank <c>r ∈ (0, 1)</c>. Bailey et al. advocate
/// the logit form because a heavy-tailed empirical distribution of logits
/// signals systematic overfitting even when the summary proportion is near
/// 0.5, and because the logit is approximately normal under the null,
/// enabling downstream inference. For <c>N</c> strategies, the integer OOS
/// rank is mapped to <c>r = rank / (N + 1)</c>. Even the best and worst ranks
/// therefore remain strictly inside the open unit interval, so their logits
/// are finite. Strategies tied with the winner out-of-sample share the average
/// rank of their tie group, and a constant return series has Sharpe ratio zero.
/// </para>
/// <para>
/// Scaling: CSCV is O(C(S, S/2) · N · T). For <c>S = 16</c> the fold count is
/// 12,870 — computationally tractable. For <c>S = 20</c> it is 184,756 and
/// still feasible; for larger <c>S</c> consider subsampling folds or fixing
/// <c>S</c> based on the shortest structural block the data supports.
/// </para>
/// <para>
/// Tier A with transcendental tail: arithmetic body, scalar Log for logit.
/// </para>
/// </remarks>
public static class ProbabilityOfBacktestOverfitting<T>
    where T : IFloatingPoint<T>
{
    private const long MaximumFoldCount = 1_000_000;

    /// <summary>
    /// Computes the PBO using Combinatorially-Symmetric Cross-Validation.
    /// </summary>
    /// <param name="returns">Panel of N strategy return series, T observations each. <c>returns[i, t]</c> = strategy i at time t.</param>
    /// <param name="splitCount">Number of equal-size time blocks S. Must be even and divide T evenly.</param>
    /// <returns>PBO estimate, logit median, and the empirical distribution of logits.</returns>
    public static PboResult<T> Compute(T[,] returns, int splitCount = 16)
    {
        ArgumentNullException.ThrowIfNull(returns);
        var n = returns.GetLength(0);
        var t = returns.GetLength(1);
        if (n < 2)
        {
            throw new ArgumentException("Need at least 2 strategies.", nameof(returns));
        }

        if (splitCount < 2 || splitCount % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(splitCount), splitCount, "Split count must be a positive even integer.");
        }

        if (t % splitCount != 0)
        {
            throw new ArgumentException(
                $"Time length ({t}) must be evenly divisible by splitCount ({splitCount}).",
                nameof(returns));
        }

        var blockSize = t / splitCount;

        if (blockSize < 2)
        {
            throw new ArgumentException(
                $"Each block must contain at least 2 observations for Sharpe computation " +
                $"(got blockSize = {blockSize}: returns length = {t}, splitCount = {splitCount}).",
                nameof(splitCount));
        }

        for (var strategy = 0; strategy < n; strategy++)
        {
            for (var observation = 0; observation < t; observation++)
            {
                if (!T.IsFinite(returns[strategy, observation]))
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(returns),
                        returns[strategy, observation],
                        "Strategy returns must be finite.");
                }
            }
        }

        var halfSize = splitCount / 2;
        if (FoldCountExceedsLimit(splitCount, halfSize))
        {
            throw new ArgumentOutOfRangeException(
                nameof(splitCount),
                splitCount,
                $"Split count produces more than {MaximumFoldCount:N0} CSCV folds.");
        }

        // Half cannot represent long joined sample counts or accumulated moments.
        // Widen the panel once and retain the wider Sharpes through ranking.
        if (typeof(T) == typeof(Half))
        {
            var widened = new double[n, t];
            for (var strategy = 0; strategy < n; strategy++)
            {
                for (var observation = 0; observation < t; observation++)
                {
                    widened[strategy, observation] = double.CreateChecked(returns[strategy, observation]);
                }
            }

            var result = ProbabilityOfBacktestOverfitting<double>.Compute(widened, splitCount);
            var halfLogits = new T[result.LogitValues.Count];
            for (var i = 0; i < halfLogits.Length; i++)
            {
                halfLogits[i] = T.CreateChecked(result.LogitValues[i]);
            }

            return new PboResult<T>(T.CreateChecked(result.Pbo), T.CreateChecked(result.LogitMedian), halfLogits);
        }

        var combos = Combinations(splitCount, halfSize);
        var logits = new T[combos.Count];

        // Reused per-fold scratch (allocated once per Compute, fully overwritten before each
        // fold's read, never escapes): the OOS Sharpe vector the in-sample winner is ranked in.
        var oosSharpes = new SharpeKey[n];

        for (var c = 0; c < combos.Count; c++)
        {
            var inSampleBlocks = combos[c];
            var outSampleBlocks = Complement(inSampleBlocks, splitCount);

            // Bailey et al. (2014) §2 steps (a)-(d): the training set J joins the S/2
            // selected blocks into one T/2 × N matrix and the test set J̄ joins the rest;
            // the Sharpe ratio is computed on each column of the joined sets, not averaged
            // over per-block Sharpe ratios. The IS winner is the first column with the
            // highest joined in-sample Sharpe.
            var isWinner = 0;
            var isWinnerSharpe = JoinedSharpe(returns, 0, inSampleBlocks, blockSize);
            for (var i = 1; i < n; i++)
            {
                var sharpe = JoinedSharpe(returns, i, inSampleBlocks, blockSize);
                if (sharpe.CompareTo(isWinnerSharpe) > 0)
                {
                    isWinnerSharpe = sharpe;
                    isWinner = i;
                }
            }

            for (var i = 0; i < n; i++)
            {
                oosSharpes[i] = JoinedSharpe(returns, i, outSampleBlocks, blockSize);
            }

            // Ascending OOS rank of the in-sample winner. Strategies tied with the winner
            // out-of-sample share the average (mid) rank of their tie group, so an exact
            // duplicate of the winner neither lowers nor raises its rank; the rank is kept
            // doubled so it stays an integer: 2·rank = 2·below + tied + 1.
            var below = 0;
            var tied = 0;
            var winnerOos = oosSharpes[isWinner];
            for (var i = 0; i < n; i++)
            {
                if (oosSharpes[i].CompareTo(winnerOos) < 0)
                {
                    below++;
                }
                else if (oosSharpes[i].CompareTo(winnerOos) == 0)
                {
                    tied++;
                }
            }

            var doubledRank = (2 * below) + tied + 1;
            // Preserve double's existing rounding while forming lower-precision
            // logits from the rank before Half can round the top rank to one.
            // 2·rank / 2·(N + 1) rounds identically to rank / (N + 1) for an integer rank.
            var omegaDbl = typeof(T) == typeof(double)
                ? double.CreateChecked(T.CreateChecked(doubledRank) / (T.CreateChecked(2) * T.CreateChecked(n + 1)))
                : doubledRank / (2.0 * (n + 1));
            var logitDbl = Math.Log(omegaDbl / (1.0 - omegaDbl));
            logits[c] = T.CreateChecked(logitDbl);
        }

        // Bailey et al. (2014) define PBO as the rate at which the in-sample
        // winner ranks at or below the out-of-sample median (logit <= 0). A winner
        // sitting exactly at the median (logit == 0, i.e. omega = 0.5 — reachable
        // for any odd strategy count) is included, matching the published estimator.
        var negativeCount = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] <= T.Zero)
            {
                negativeCount++;
            }
        }

        var pbo = typeof(T) == typeof(Half)
            ? T.CreateChecked((double)negativeCount / logits.Length)
            : T.CreateChecked(negativeCount) / T.CreateChecked(logits.Length);

        var sortedLogits = (T[])logits.Clone();
        Array.Sort(sortedLogits);
        var mid = sortedLogits.Length / 2;
        var median = sortedLogits.Length % 2 == 1
            ? sortedLogits[mid]
            : (sortedLogits[mid - 1] + sortedLogits[mid]) * T.CreateChecked(0.5);

        return new PboResult<T>(pbo, median, sortedLogits);
    }

    /// <summary>
    /// Sharpe ratio (mean over sample standard deviation) of one strategy's returns on
    /// the rows of the given blocks, joined in their original order.
    /// </summary>
    private static SharpeKey JoinedSharpe(T[,] returns, int strategy, int[] blocks, int blockSize)
    {
        var length = blocks.Length * blockSize;
        var first = returns[strategy, blocks[0] * blockSize];
        var minimum = first;
        var maximum = first;
        for (var k = 0; k < blocks.Length; k++)
        {
            var start = blocks[k] * blockSize;
            for (var i = 0; i < blockSize; i++)
            {
                var value = returns[strategy, start + i];
                minimum = T.Min(minimum, value);
                maximum = T.Max(maximum, value);
            }
        }

        // A constant column has zero standard deviation and therefore Sharpe 0. Test it
        // exactly: the summed mean of equal values can miss the value by an ulp, which
        // would turn the deviations into rounding residue and the Sharpe into ~1e16.
        if (minimum == maximum)
        {
            return new SharpeKey(T.Zero, 0);
        }

        // Remove a common location before scaling: dividing raw levels by their maximum rounds
        // a large shared level into a few ulps of 1 and erases the representable spread, which
        // turns distinct Sharpe ratios into false ties (decimal 7e28 + {0, 2, 4, 6}). With one
        // sign throughout, differences from the first return cannot overflow. A mixed-sign
        // column has a spread comparable to its range, so it is normalized about zero, where
        // opposite-sign differences cannot overflow either.
        var anchor = minimum >= T.Zero || maximum <= T.Zero ? first : T.Zero;
        var scale = T.Max(maximum - anchor, anchor - minimum);

        // Sharpe is dimensionless: divide both moments by a common positive scale
        // before squaring, so small but nonconstant returns cannot lose their variance
        // solely because their original squared units underflow.
        var sum = T.Zero;
        for (var k = 0; k < blocks.Length; k++)
        {
            var start = blocks[k] * blockSize;
            for (var i = 0; i < blockSize; i++)
            {
                sum += (returns[strategy, start + i] - anchor) / scale;
            }
        }

        var offsetMean = sum / T.CreateChecked(length);

        var ssq = T.Zero;
        for (var k = 0; k < blocks.Length; k++)
        {
            var start = blocks[k] * blockSize;
            for (var i = 0; i < blockSize; i++)
            {
                var dev = (returns[strategy, start + i] - anchor) / scale - offsetMean;
                ssq += dev * dev;
            }
        }

        var sd = NumericPrecision<T>.Sqrt(ssq / T.CreateChecked(length - 1));
        // Mean and standard deviation are both in units of scale, with the anchor's level
        // kept in the numerator.
        if (sd == T.Zero)
        {
            return new SharpeKey(T.Zero, 0);
        }
        try
        {
            var value = (anchor / scale + offsetMean) / sd;
            if (T.IsFinite(value))
            {
                return new SharpeKey(value, 0);
            }
        }
        catch (OverflowException)
        {
            // A Sharpe used only for ranking need not fit in the result type.
        }

        var mean = anchor + offsetMean * scale;
        if (mean == T.Zero)
        {
            return new SharpeKey(T.Zero, 0);
        }
        var meanExponent = ScaledArithmetic<T>.ExponentOfMagnitude(T.Abs(mean));
        var scaleExponent = ScaledArithmetic<T>.ExponentOfMagnitude(scale);
        var mantissa = ScaledArithmetic<T>.Rescale(mean, -meanExponent)
            / ScaledArithmetic<T>.Rescale(scale, -scaleExponent) / sd;
        return new SharpeKey(mantissa, meanExponent - scaleExponent);
    }

    // Comparison-only representation value * radix^exponent. Ordinary finite
    // Sharpes keep exponent zero and retain their existing arithmetic and ties.
    private readonly record struct SharpeKey(T Value, int Exponent) : IComparable<SharpeKey>
    {
        public int CompareTo(SharpeKey other)
        {
            if (Exponent == other.Exponent || Value == T.Zero || other.Value == T.Zero
                || T.IsNegative(Value) != T.IsNegative(other.Value))
            {
                return Value.CompareTo(other.Value);
            }
            var leftScale = ScaledArithmetic<T>.ExponentOfMagnitude(T.Abs(Value));
            var rightScale = ScaledArithmetic<T>.ExponentOfMagnitude(T.Abs(other.Value));
            var order = (Exponent + leftScale).CompareTo(other.Exponent + rightScale);
            if (order != 0)
            {
                return T.IsNegative(Value) ? -order : order;
            }
            return ScaledArithmetic<T>.Rescale(Value, -leftScale)
                .CompareTo(ScaledArithmetic<T>.Rescale(other.Value, -rightScale));
        }
    }

    private static List<int[]> Combinations(int n, int k)
    {
        var result = new List<int[]>();
        var current = new int[k];
        Enumerate(0, 0, n, k, current, result);
        return result;
    }

    private static bool FoldCountExceedsLimit(int n, int k)
    {
        var count = 1L;
        k = Math.Min(k, n - k);
        for (var i = 1; i <= k; i++)
        {
            var factor = n - k + i;
            if (count > MaximumFoldCount * i / factor)
            {
                return true;
            }

            count = count * factor / i;
        }

        return count > MaximumFoldCount;
    }

    private static void Enumerate(int start, int depth, int n, int k, int[] current, List<int[]> result)
    {
        if (depth == k)
        {
            result.Add((int[])current.Clone());
            return;
        }

        for (var i = start; i <= n - (k - depth); i++)
        {
            current[depth] = i;
            Enumerate(i + 1, depth + 1, n, k, current, result);
        }
    }

    private static int[] Complement(int[] subset, int n)
    {
        var setLookup = new bool[n];
        for (var i = 0; i < subset.Length; i++)
        {
            setLookup[subset[i]] = true;
        }

        var result = new int[n - subset.Length];
        var idx = 0;
        for (var i = 0; i < n; i++)
        {
            if (!setLookup[i])
            {
                result[idx++] = i;
            }
        }

        return result;
    }
}

/// <summary>
/// Result of a Combinatorially-Symmetric Cross-Validation (CSCV) run for
/// the Probability of Backtest Overfitting.
/// </summary>
/// <param name="Pbo">Estimated probability that the in-sample best strategy underperforms the median out-of-sample.</param>
/// <param name="LogitMedian">Median logit of OOS rank — the central tendency of the overfitting distribution.</param>
/// <param name="LogitValues">Per-fold logit values; the empirical distribution behind the PBO.</param>
public sealed record PboResult(decimal Pbo, decimal LogitMedian, IReadOnlyList<decimal> LogitValues)
{
    private IReadOnlyList<decimal> _logitValues = Snapshot(LogitValues);

    /// <summary>
    /// Per-fold logit values in ascending order. Constructor and initializer inputs
    /// are snapshotted, and the returned view cannot be cast to a mutable array.
    /// </summary>
    public IReadOnlyList<decimal> LogitValues
    {
        get => _logitValues;
        init => _logitValues = Snapshot(value);
    }

    private static IReadOnlyList<decimal> Snapshot(IReadOnlyList<decimal> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return Array.AsReadOnly(values.ToArray());
    }
}

/// <summary>
/// Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
/// Cross-Validation (CSCV). Splits a panel of <c>N</c> strategy return series
/// × <c>T</c> time observations into <c>S</c> balanced blocks and exhausts all
/// <c>C(S, S/2)</c> half-half splits. For each split the S/2 training blocks are
/// joined into one in-sample set and the rest into the out-of-sample set, the
/// Sharpe ratio is computed on each joined set, and the in-sample winner is
/// identified; PBO is the proportion of splits in which that winner ranks at or
/// below the out-of-sample median.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H., Borwein, J. M., López de Prado, M. &amp; Zhu,
/// Q. J. (2014). "The Probability of Backtest Overfitting." Journal of
/// Computational Finance, 20(4), 39–70.
/// </para>
/// <para>
/// PBO complements the Deflated Sharpe Ratio: DSR adjusts a single observed
/// Sharpe for the search process that produced it, while PBO measures the
/// generalization rate of the search procedure itself. <c>PBO &gt; 0.5</c>
/// signals that the search is anti-predictive — picking the in-sample winner
/// is worse than random selection out-of-sample.
/// </para>
/// <para>
/// Logit aggregation: the reported <c>Pbo</c> is the empirical probability
/// that the winner's OOS rank falls below the median, but <c>LogitMedian</c>
/// and <c>LogitValues</c> are computed on the per-fold logit
/// <c>ln(r / (1 − r))</c> of the rank <c>r ∈ (0, 1)</c>. Bailey et al. advocate
/// the logit form because a heavy-tailed empirical distribution of logits
/// signals systematic overfitting even when the summary proportion is near
/// 0.5, and because the logit is approximately normal under the null,
/// enabling downstream inference. For <c>N</c> strategies, the integer OOS
/// rank is mapped to <c>r = rank / (N + 1)</c>. Even the best and worst ranks
/// therefore remain strictly inside the open unit interval, so their logits
/// are finite. Strategies tied with the winner out-of-sample share the average
/// rank of their tie group, and a constant return series has Sharpe ratio zero.
/// </para>
/// <para>
/// Scaling: CSCV is O(C(S, S/2) · N · T). For <c>S = 16</c> the fold count is
/// 12,870 — computationally tractable. For <c>S = 20</c> it is 184,756 and
/// still feasible; for larger <c>S</c> consider subsampling folds or fixing
/// <c>S</c> based on the shortest structural block the data supports.
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="ProbabilityOfBacktestOverfitting{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
[Obsolete(
    "Use ProbabilityOfBacktestOverfitting<decimal> instead (the facade preserves the original decimal precision; use <double> deliberately if you want IEEE precision). This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class ProbabilityOfBacktestOverfitting
{
    /// <summary>
    /// Computes the PBO using Combinatorially-Symmetric Cross-Validation.
    /// </summary>
    /// <param name="returns">Panel of N strategy return series, T observations each. <c>returns[i, t]</c> = strategy i at time t.</param>
    /// <param name="splitCount">Number of equal-size time blocks S. Must be even and divide T evenly.</param>
    /// <returns>PBO estimate, logit median, and the empirical distribution of logits.</returns>
    public static PboResult Compute(decimal[,] returns, int splitCount = 16)
    {
        var result = ProbabilityOfBacktestOverfitting<decimal>.Compute(returns, splitCount);
        return new PboResult(result.Pbo, result.LogitMedian, result.LogitValues);
    }
}
