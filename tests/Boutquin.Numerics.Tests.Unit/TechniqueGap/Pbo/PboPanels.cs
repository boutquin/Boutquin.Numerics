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

namespace Boutquin.Numerics.Tests.Unit.TechniqueGap.Pbo;

/// <summary>One panel of a population: strategies by observations, with its number of blocks.</summary>
internal sealed record PboPanel(string Kind, double[,] Returns, int SplitCount);

/// <summary>
/// The panels of the PBO tests: a design population, a held-out population with other seeds,
/// sizes and scales, and witnesses chosen for the mechanism (the fold statistics come from
/// merged block moments in place of passes over the joined observations).
/// </summary>
internal static class PboPanels
{
    /// <summary>The kinds of panel of the two random populations.</summary>
    private static readonly string[] s_kinds =
    [
        "plain", "offset", "scales", "near-ties", "exact-ties", "constant", "quiet-block", "outlier-block", "mixed-sign-offset",
    ];

    internal static IEnumerable<PboPanel> Design() => Population(20261130, panels: 90, maxStrategies: 8, splits: [4, 6, 8], maxBlock: 12, scale: 1.0);

    internal static IEnumerable<PboPanel> HeldOut() => Population(977_311, panels: 90, maxStrategies: 11, splits: [4, 8, 10], maxBlock: 9, scale: 3.0e-7);

    private static IEnumerable<PboPanel> Population(int seed, int panels, int maxStrategies, int[] splits, int maxBlock, double scale)
    {
        var random = new System.Random(seed);
        for (var p = 0; p < panels; p++)
        {
            var kind = s_kinds[p % s_kinds.Length];
            var n = 2 + random.Next(maxStrategies - 1);
            var splitCount = splits[random.Next(splits.Length)];
            var block = 2 + random.Next(maxBlock - 1);
            yield return new PboPanel(kind, Build(random, kind, n, splitCount, block, scale), splitCount);
        }
    }

    private static double Noise(System.Random random) => (random.NextDouble() + random.NextDouble() + random.NextDouble() - 1.5) * 0.02;

    private static double[,] Build(System.Random random, string kind, int n, int splitCount, int block, double scale)
    {
        var t = splitCount * block;
        var returns = new double[n, t];
        for (var i = 0; i < n; i++)
        {
            var drift = (random.NextDouble() - 0.4) * 0.004;
            for (var k = 0; k < t; k++)
            {
                returns[i, k] = scale * (drift + Noise(random));
            }
        }

        switch (kind)
        {
            case "offset":
                // A level far above the spread, one sign throughout.
                for (var i = 0; i < n; i++)
                {
                    var level = scale * Math.Pow(10, 2 + random.Next(7));
                    for (var k = 0; k < t; k++)
                    {
                        returns[i, k] += level;
                    }
                }

                break;
            case "scales":
                // Each strategy in its own unit, twelve orders of magnitude apart at most.
                for (var i = 0; i < n; i++)
                {
                    var unit = Math.Pow(10, random.Next(-6, 7));
                    for (var k = 0; k < t; k++)
                    {
                        returns[i, k] *= unit;
                    }
                }

                break;
            case "near-ties":
                // Every second strategy is the one before it with a few values moved by a
                // relative 1e-13 to 1e-9: Sharpe ratios that differ far below the spread.
                for (var i = 1; i < n; i += 2)
                {
                    for (var k = 0; k < t; k++)
                    {
                        returns[i, k] = returns[i - 1, k];
                    }

                    for (var moved = 0; moved < 3; moved++)
                    {
                        var k = random.Next(t);
                        returns[i, k] *= 1 + ((random.NextDouble() - 0.5) * Math.Pow(10, -9 - random.Next(5)));
                    }
                }

                break;
            case "exact-ties":
                // Copies, and copies in another unit by a power of two: equal Sharpe ratios.
                for (var i = 1; i < n; i += 2)
                {
                    var unit = i % 4 == 1 ? 1.0 : 8.0;
                    for (var k = 0; k < t; k++)
                    {
                        returns[i, k] = unit * returns[i - 1, k];
                    }
                }

                break;
            case "constant":
                for (var k = 0; k < t; k++)
                {
                    returns[n - 1, k] = scale * 0.01;
                }

                // And one strategy that is constant on one block only.
                for (var k = 0; k < block; k++)
                {
                    returns[0, k] = scale * 0.003;
                }

                break;
            case "quiet-block":
                // One block of one strategy moves a fraction of the rest: fractions on both
                // sides of any fixed threshold between 1e-2 and 1e-9.
                for (var i = 0; i < n; i += 2)
                {
                    var quiet = random.Next(splitCount);
                    var fraction = Math.Pow(10, -2 - (7 * random.NextDouble()));
                    for (var k = 0; k < block; k++)
                    {
                        returns[i, (quiet * block) + k] *= fraction;
                    }
                }

                break;
            case "outlier-block":
                // One block of one strategy is a million times the rest, then its mirror image.
                for (var i = 0; i < n; i += 2)
                {
                    var loud = random.Next(splitCount - 1);
                    for (var k = 0; k < block; k++)
                    {
                        returns[i, (loud * block) + k] *= 1e6;
                        returns[i, ((loud + 1) * block) + k] = -returns[i, (loud * block) + k];
                    }
                }

                break;
            case "mixed-sign-offset":
                // A large positive level in some blocks and a large negative one in others.
                for (var i = 0; i < n; i++)
                {
                    var level = scale * Math.Pow(10, 1 + random.Next(5));
                    for (var b = 0; b < splitCount; b++)
                    {
                        var sign = random.Next(2) == 0 ? 1.0 : -1.0;
                        for (var k = 0; k < block; k++)
                        {
                            returns[i, (b * block) + k] += sign * level;
                        }
                    }
                }

                break;
        }

        return returns;
    }

    /// <summary>
    /// Witnesses by mechanism: cancellation across scales, a dominant term that cancels later,
    /// values on the two sides of a scaling threshold, and Sharpe ratios that differ by parts
    /// in 1e12.
    /// </summary>
    internal static IEnumerable<PboPanel> Witnesses()
    {
        // A level of 1e8 with a spread of 1e-4, in four strategies that differ in the spread only.
        var level = new double[4, 24];
        for (var i = 0; i < 4; i++)
        {
            for (var k = 0; k < 24; k++)
            {
                level[i, k] = 1e8 + (1e-4 * (((k * (i + 3)) % 7) - 3 + (0.1 * i)));
            }
        }

        yield return new PboPanel("level-1e8-spread-1e-4", level, 4);

        // A dominant block that the next block cancels: the joined mean is the small rest.
        var cancelling = new double[3, 32];
        for (var i = 0; i < 3; i++)
        {
            for (var k = 0; k < 32; k++)
            {
                var small = 0.01 * (((k * (2 * i + 5)) % 11) - 4.5 + i);
                cancelling[i, k] = k / 4 == 2 ? 1e9 + small : k / 4 == 3 ? -1e9 + small : small;
            }
        }

        yield return new PboPanel("dominant-block-cancels", cancelling, 8);

        // Blocks whose spread is just under and just over 1e-4 of the spread of the series.
        for (var side = 0; side < 2; side++)
        {
            var fraction = side == 0 ? 0.9e-4 : 1.1e-4;
            var threshold = new double[3, 24];
            for (var i = 0; i < 3; i++)
            {
                for (var k = 0; k < 24; k++)
                {
                    var value = (((k * (i + 2)) % 5) - 2) + (0.01 * i * (k % 3));
                    threshold[i, k] = k / 6 == 1 ? value * fraction : value;
                }
            }

            yield return new PboPanel(side == 0 ? "block-spread-0.9e-4" : "block-spread-1.1e-4", threshold, 4);
        }

        // Two strategies whose Sharpe ratios differ by a part in 1e12 on every joined set.
        var close = new double[3, 32];
        for (var k = 0; k < 32; k++)
        {
            var value = ((k * 7) % 13) - 5.5;
            close[0, k] = value;
            close[1, k] = value + (1e-12 * (k % 2 == 0 ? 1 : -0.5));
            close[2, k] = 0.3 * value * (k % 3);
        }

        yield return new PboPanel("sharpe-differs-1e-12", close, 8);

        // A series with values of opposite signs at 1e150 and a quiet half at 1e-150.
        var range = new double[2, 16];
        for (var k = 0; k < 16; k++)
        {
            var unit = k < 8 ? 1e150 : 1e-150;
            range[0, k] = unit * (((k * 3) % 5) - 2);
            range[1, k] = unit * (((k * 5) % 7) - 2.5);
        }

        yield return new PboPanel("range-1e150-and-1e-150", range, 4);
    }

    /// <summary>The panel in another type, value by value.</summary>
    internal static TTarget[,] To<TTarget>(double[,] returns)
        where TTarget : System.Numerics.IFloatingPoint<TTarget>
    {
        var converted = new TTarget[returns.GetLength(0), returns.GetLength(1)];
        for (var i = 0; i < returns.GetLength(0); i++)
        {
            for (var k = 0; k < returns.GetLength(1); k++)
            {
                converted[i, k] = TTarget.CreateChecked(returns[i, k]);
            }
        }

        return converted;
    }
}
