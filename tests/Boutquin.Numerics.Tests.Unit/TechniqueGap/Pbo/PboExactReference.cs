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
using Boutquin.Numerics.Tests.Unit.TechniqueGap.Kernels;

namespace Boutquin.Numerics.Tests.Unit.TechniqueGap.Pbo;

/// <summary>
/// The logits of the probability of backtest overfitting from exact rational arithmetic on the
/// stored panel: for every half-half split, the first strategy with the largest exact
/// in-sample Sharpe ratio, and its exact out-of-sample rank with ties averaged. A column that
/// is constant on a joined set has Sharpe ratio zero there, as the library documents. The rank
/// is turned into a logit by the library's own expression, so two logit lists differ only
/// where a winner or a rank differs.
/// </summary>
internal static class PboExactReference
{
    /// <summary>The doubled out-of-sample rank of the in-sample winner, for every split.</summary>
    internal static int[] DoubledRanks<T>(T[,] returns, int splitCount)
        where T : IFloatingPoint<T>
    {
        var n = returns.GetLength(0);
        var t = returns.GetLength(1);
        var blockSize = t / splitCount;

        // Sum and sum of squares of each strategy on each block.
        var sums = new ExactRational[n, splitCount];
        var squares = new ExactRational[n, splitCount];
        for (var i = 0; i < n; i++)
        {
            for (var b = 0; b < splitCount; b++)
            {
                var sum = ExactRational.Zero;
                var square = ExactRational.Zero;
                for (var k = 0; k < blockSize; k++)
                {
                    var value = ExactRational.From(returns[i, (b * blockSize) + k]);
                    sum += value;
                    square += value * value;
                }

                sums[i, b] = sum;
                squares[i, b] = square;
            }
        }

        var length = ExactRational.From((long)(splitCount / 2) * blockSize);
        var ranks = new List<int>();
        var inSample = new SharpeKey[n];
        var outSample = new SharpeKey[n];
        for (var mask = 0; mask < 1 << splitCount; mask++)
        {
            if (BitOperations.PopCount((uint)mask) != splitCount / 2)
            {
                continue;
            }

            for (var i = 0; i < n; i++)
            {
                inSample[i] = Key(sums, squares, i, mask, splitCount, length);
                outSample[i] = Key(sums, squares, i, ~mask, splitCount, length);
            }

            var winner = 0;
            for (var i = 1; i < n; i++)
            {
                if (inSample[i].CompareTo(inSample[winner]) > 0)
                {
                    winner = i;
                }
            }

            var below = 0;
            var tied = 0;
            for (var i = 0; i < n; i++)
            {
                var order = outSample[i].CompareTo(outSample[winner]);
                if (order < 0)
                {
                    below++;
                }
                else if (order == 0)
                {
                    tied++;
                }
            }

            ranks.Add((2 * below) + tied + 1);
        }

        return [.. ranks];
    }

    /// <summary>The logits of those ranks in ascending order, as <c>double</c> values.</summary>
    internal static double[] Logits<T>(T[,] returns, int splitCount)
        where T : IFloatingPoint<T>
    {
        var n = returns.GetLength(0);
        var logits = DoubledRanks(returns, splitCount)
            .Select(rank =>
            {
                var omega = rank / (2.0 * (n + 1));
                return Math.Log(omega / (1.0 - omega));
            })
            .ToArray();
        Array.Sort(logits);
        return logits;
    }

    /// <summary>
    /// The least number of splits whose logit differs between two sorted logit lists of one
    /// panel: half the distance between their histograms.
    /// </summary>
    internal static int SplitsOff(IReadOnlyList<double> actual, IReadOnlyList<double> exact)
    {
        var counts = new Dictionary<long, int>();
        foreach (var value in actual)
        {
            var key = BitConverter.DoubleToInt64Bits(value);
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        foreach (var value in exact)
        {
            var key = BitConverter.DoubleToInt64Bits(value);
            counts[key] = counts.GetValueOrDefault(key) - 1;
        }

        return counts.Values.Sum(Math.Abs) / 2;
    }

    private static SharpeKey Key(
        ExactRational[,] sums, ExactRational[,] squares, int strategy, int mask, int splitCount, ExactRational length)
    {
        var sum = ExactRational.Zero;
        var square = ExactRational.Zero;
        for (var b = 0; b < splitCount; b++)
        {
            if ((mask & (1 << b)) != 0)
            {
                sum += sums[strategy, b];
                square += squares[strategy, b];
            }
        }

        // length · Σx² − (Σx)² is length² times the population variance: zero for a constant set.
        var spread = (length * square) - (sum * sum);
        return new SharpeKey(sum, spread);
    }

    // A Sharpe ratio up to the positive factor that every strategy of a joined set shares:
    // sum / sqrt(spread), and zero where the spread is zero.
    private readonly record struct SharpeKey(ExactRational Sum, ExactRational Spread) : IComparable<SharpeKey>
    {
        private int Sign => Spread.Numerator.IsZero ? 0 : Sum.Numerator.Sign;

        public int CompareTo(SharpeKey other)
        {
            if (Sign != other.Sign)
            {
                return Sign.CompareTo(other.Sign);
            }

            if (Sign == 0)
            {
                return 0;
            }

            // Same sign: compare the squares, sum² / spread, and reverse for negative ratios.
            var order = (Sum * Sum * other.Spread).CompareTo(other.Sum * other.Sum * Spread);
            return Sign > 0 ? order : -order;
        }
    }
}
