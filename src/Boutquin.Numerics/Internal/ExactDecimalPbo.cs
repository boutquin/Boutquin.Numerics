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

#if NET11_0_OR_GREATER
using System.Numerics;
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.Internal;

/// <summary>Ranks decimal Sharpe ratios without division or square-root rounding.</summary>
internal static class ExactDecimalPbo
{
    internal static PboResult<decimal> Compute(decimal[,] returns, int splitCount, out long work)
    {
        var strategies = returns.GetLength(0);
        var blockSize = returns.GetLength(1) / splitCount;
        var length = (splitCount / 2) * blockSize;
        var sums = new BigInteger[strategies, splitCount];
        var squares = new BigInteger[strategies, splitCount];
        work = 0;
        for (var i = 0; i < strategies; i++)
        {
            for (var b = 0; b < splitCount; b++)
            {
                for (var k = 0; k < blockSize; k++)
                {
                    var value = ScaledInteger(returns[i, (b * blockSize) + k]);
                    sums[i, b] += value;
                    squares[i, b] += value * value;
                    work++;
                }
            }
        }

        var logits = new List<decimal>();
        var training = new Key[strategies];
        var testing = new Key[strategies];
        for (var mask = 0; mask < 1 << splitCount; mask++)
        {
            if (BitOperations.PopCount((uint)mask) != splitCount / 2)
            {
                continue;
            }

            for (var i = 0; i < strategies; i++)
            {
                var inSum = BigInteger.Zero;
                var outSum = BigInteger.Zero;
                var inSquares = BigInteger.Zero;
                var outSquares = BigInteger.Zero;
                for (var b = 0; b < splitCount; b++)
                {
                    if ((mask & (1 << b)) != 0)
                    {
                        inSum += sums[i, b];
                        inSquares += squares[i, b];
                    }
                    else
                    {
                        outSum += sums[i, b];
                        outSquares += squares[i, b];
                    }

                    work += 2;
                }

                training[i] = new Key(inSum, (length * inSquares) - (inSum * inSum));
                testing[i] = new Key(outSum, (length * outSquares) - (outSum * outSum));
            }

            var winner = 0;
            for (var i = 1; i < strategies; i++)
            {
                if (training[i].CompareTo(training[winner]) > 0)
                {
                    winner = i;
                }
            }

            var below = 0;
            var tied = 0;
            for (var i = 0; i < strategies; i++)
            {
                var order = testing[i].CompareTo(testing[winner]);
                below += order < 0 ? 1 : 0;
                tied += order == 0 ? 1 : 0;
            }

            var omega = ((2 * below) + tied + 1) / (2.0 * (strategies + 1));
            logits.Add((decimal)Math.Log(omega / (1 - omega)));
        }

        logits.Sort();
        var probability = (decimal)logits.Count(value => value <= 0) / logits.Count;
        var middle = logits.Count / 2;
        var median = logits.Count % 2 == 1 ? logits[middle] : (logits[middle - 1] + logits[middle]) / 2;
        return new PboResult<decimal>(probability, median, logits);
    }

    private static BigInteger ScaledInteger(decimal value)
    {
        Span<int> bits = stackalloc int[4];
        decimal.GetBits(value, bits);
        var magnitude = (new BigInteger((uint)bits[2]) << 64)
            | (new BigInteger((uint)bits[1]) << 32) | (uint)bits[0];
        magnitude *= BigInteger.Pow(10, 28 - ((bits[3] >> 16) & 0xff));
        return bits[3] < 0 ? -magnitude : magnitude;
    }

    private readonly record struct Key(BigInteger Sum, BigInteger Variance) : IComparable<Key>
    {
        private int Sign => Variance.IsZero ? 0 : Sum.Sign;

        public int CompareTo(Key other)
        {
            var signOrder = Sign.CompareTo(other.Sign);
            if (signOrder != 0 || Sign == 0)
            {
                return signOrder;
            }

            return Sign * (Sum * Sum * other.Variance).CompareTo(other.Sum * other.Sum * Variance);
        }
    }
}
#endif
