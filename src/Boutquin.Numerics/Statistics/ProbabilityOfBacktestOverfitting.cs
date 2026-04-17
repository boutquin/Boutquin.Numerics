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
    where T : IFloatingPoint<T>;

/// <summary>
/// Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
/// Cross-Validation (CSCV). Splits a panel of <c>N</c> strategy return series
/// × <c>T</c> time observations into <c>S</c> balanced blocks and exhausts all
/// <c>C(S, S/2)</c> half-half splits. For each split the in-sample winner (by
/// Sharpe) is identified; PBO is the proportion of splits in which that winner
/// ranks below the out-of-sample median.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H., Borwein, J. M., López de Prado, M. &amp; Zhu,
/// Q. J. (2014). "The Probability of Backtest Overfitting." Journal of
/// Computational Finance, 20(4), 39–70. arXiv:1109.0776.
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
/// enabling downstream inference. Ranks at the extremes are clamped to
/// <c>[1 / (N + 1), N / (N + 1)]</c> so the logit never diverges on ties at
/// the best or worst position.
/// </para>
/// <para>
/// Scaling: CSCV is O(C(S, S/2) · N · T). For <c>S = 16</c> the fold count is
/// 12,870 — computationally tractable. For <c>S = 20</c> it is 184,756 and
/// still feasible; for larger <c>S</c> consider subsampling folds or fixing
/// <c>S</c> based on the shortest structural block the data supports
/// (blocks must remain long enough for meaningful per-block Sharpe estimates).
/// </para>
/// <para>
/// Tier A with transcendental tail: arithmetic body, scalar Log for logit.
/// </para>
/// </remarks>
public static class ProbabilityOfBacktestOverfitting<T>
    where T : IFloatingPoint<T>
{
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
        var halfSize = splitCount / 2;

        var blockSharpe = new T[n, splitCount];
        for (var i = 0; i < n; i++)
        {
            for (var b = 0; b < splitCount; b++)
            {
                blockSharpe[i, b] = Sharpe(returns, i, b * blockSize, blockSize);
            }
        }

        var combos = Combinations(splitCount, halfSize);
        var logits = new T[combos.Count];

        for (var c = 0; c < combos.Count; c++)
        {
            var inSampleBlocks = combos[c];
            var outSampleBlocks = Complement(inSampleBlocks, splitCount);

            var isWinner = 0;
            var firstSum = T.Zero;
            for (var k = 0; k < inSampleBlocks.Length; k++)
            {
                firstSum += blockSharpe[0, inSampleBlocks[k]];
            }

            var isWinnerSharpe = firstSum / T.CreateChecked(inSampleBlocks.Length);
            for (var i = 1; i < n; i++)
            {
                var sum = T.Zero;
                for (var k = 0; k < inSampleBlocks.Length; k++)
                {
                    sum += blockSharpe[i, inSampleBlocks[k]];
                }

                var avg = sum / T.CreateChecked(inSampleBlocks.Length);
                if (avg > isWinnerSharpe)
                {
                    isWinnerSharpe = avg;
                    isWinner = i;
                }
            }

            var oosSharpes = new T[n];
            for (var i = 0; i < n; i++)
            {
                var sum = T.Zero;
                for (var k = 0; k < outSampleBlocks.Length; k++)
                {
                    sum += blockSharpe[i, outSampleBlocks[k]];
                }

                oosSharpes[i] = sum / T.CreateChecked(outSampleBlocks.Length);
            }

            var sortedIdx = Enumerable.Range(0, n).OrderBy(i => oosSharpes[i]).ToArray();
            var rank = Array.IndexOf(sortedIdx, isWinner) + 1;
            var omega = T.CreateChecked(rank) / T.CreateChecked(n + 1);

            if (omega <= T.Zero)
            {
                omega = T.One / T.CreateChecked(2 * (n + 1));
            }
            else if (omega >= T.One)
            {
                omega = T.One - T.One / T.CreateChecked(2 * (n + 1));
            }

            var omegaDbl = double.CreateChecked(omega);
            var logitDbl = Math.Log(omegaDbl / (1.0 - omegaDbl));
            logits[c] = T.CreateChecked(logitDbl);
        }

        var negativeCount = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] <= T.Zero)
            {
                negativeCount++;
            }
        }

        var pbo = T.CreateChecked(negativeCount) / T.CreateChecked(logits.Length);

        var sortedLogits = (T[])logits.Clone();
        Array.Sort(sortedLogits);
        var median = sortedLogits[sortedLogits.Length / 2];

        return new PboResult<T>(pbo, median, sortedLogits);
    }

    private static T Sharpe(T[,] returns, int strategy, int start, int length)
    {
        var sum = T.Zero;
        for (var i = 0; i < length; i++)
        {
            sum += returns[strategy, start + i];
        }

        var mean = sum / T.CreateChecked(length);

        var ssq = T.Zero;
        for (var i = 0; i < length; i++)
        {
            var dev = returns[strategy, start + i] - mean;
            ssq += dev * dev;
        }

        var sd = NumericPrecision<T>.Sqrt(ssq / T.CreateChecked(length - 1));
        return sd == T.Zero ? T.Zero : mean / sd;
    }

    private static List<int[]> Combinations(int n, int k)
    {
        var result = new List<int[]>();
        var current = new int[k];
        Enumerate(0, 0, n, k, current, result);
        return result;
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
public sealed record PboResult(decimal Pbo, decimal LogitMedian, IReadOnlyList<decimal> LogitValues);

/// <summary>
/// Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
/// Cross-Validation (CSCV). Splits a panel of <c>N</c> strategy return series
/// × <c>T</c> time observations into <c>S</c> balanced blocks and exhausts all
/// <c>C(S, S/2)</c> half-half splits. For each split the in-sample winner (by
/// Sharpe) is identified; PBO is the proportion of splits in which that winner
/// ranks below the out-of-sample median.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Bailey, D. H., Borwein, J. M., López de Prado, M. &amp; Zhu,
/// Q. J. (2014). "The Probability of Backtest Overfitting." Journal of
/// Computational Finance, 20(4), 39–70. arXiv:1109.0776.
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
/// enabling downstream inference. Ranks at the extremes are clamped to
/// <c>[1 / (N + 1), N / (N + 1)]</c> so the logit never diverges on ties at
/// the best or worst position.
/// </para>
/// <para>
/// Scaling: CSCV is O(C(S, S/2) · N · T). For <c>S = 16</c> the fold count is
/// 12,870 — computationally tractable. For <c>S = 20</c> it is 184,756 and
/// still feasible; for larger <c>S</c> consider subsampling folds or fixing
/// <c>S</c> based on the shortest structural block the data supports
/// (blocks must remain long enough for meaningful per-block Sharpe estimates).
/// </para>
/// <para>
/// Legacy facade: delegates to <see cref="ProbabilityOfBacktestOverfitting{T}"/> at <c>T = decimal</c>.
/// </para>
/// </remarks>
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
