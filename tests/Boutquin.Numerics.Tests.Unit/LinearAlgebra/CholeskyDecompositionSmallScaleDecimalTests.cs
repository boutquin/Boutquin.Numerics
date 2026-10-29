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

using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

/// <summary>
/// The default stopping tolerance of the pivoted Cholesky decomposition on small-scale <see cref="decimal"/> inputs.
/// Below magnitude one, <see cref="decimal"/> rounds to an absolute grid of <c>1e-28</c>, so <c>n·u·max(diag(a))</c>
/// falls under that grid and would round to exactly zero: the absolute-zero cutoff the default exists to replace. An
/// exactly rank-deficient matrix must stop at its true rank, as it does at <see cref="double"/>.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class CholeskyDecompositionSmallScaleDecimalTests
{
    // B·B′ is computed exactly in decimal (entries on the 0.01 grid, products on the 1e-4 grid), so A has rank exactly 2.
    private static readonly decimal[,] s_rankTwoFactor =
    {
        { 0.07m, 0.04m },
        { 0.04m, -0.02m },
        { -0.09m, -0.01m },
        { -0.05m, -0.05m },
    };

    [Fact]
    public void DecomposePivoted_SmallScaleExactlyRankDeficientDecimal_StopsAtTheTrueRank()
    {
        var a = Gram(s_rankTwoFactor);

        Assert.Equal(2, CholeskyDecomposition<decimal>.DecomposePivoted(a).Rank);
#pragma warning disable BNU0001 // The legacy facade's 0m default must request the same rule.
        Assert.Equal(2, CholeskyDecomposition.DecomposePivoted(a).Rank);
#pragma warning restore BNU0001
    }

    // The seeded family of the case above: rank-r Gram matrices of factors on the 0.01 grid, n from 3 to 8. Each is
    // exactly of rank at most r in decimal arithmetic, a bound known independently of the decomposition.
    [Theory]
    [InlineData(20261024)]
    [InlineData(20261025)]
    [InlineData(20261026)]
    public void DecomposePivoted_SeededSmallScaleRankDeficientDecimal_StopsAtTheTrueRank(int seed)
    {
        var rng = new System.Random(seed);
        var spurious = new List<string>();
        for (var trial = 0; trial < 200; trial++)
        {
            var n = rng.Next(3, 9);
            var rank = rng.Next(1, Math.Min(4, n));
            var factor = RandomFactor(n, rank, rng);
            var actual = CholeskyDecomposition<decimal>.DecomposePivoted(Gram(factor)).Rank;
            if (actual > rank)
            {
                spurious.Add($"trial {trial}: n={n} true rank {rank}, reported {actual}");
            }
        }

        Assert.True(spurious.Count == 0, string.Join(Environment.NewLine, spurious));
    }

    // The default must not swallow a genuine small pivot: full-rank small-scale Gram matrices keep every pivot.
    [Fact]
    public void DecomposePivoted_SmallScaleFullRankDecimal_KeepsEveryPivot()
    {
        var rng = new System.Random(20261027);
        for (var trial = 0; trial < 100; trial++)
        {
            // A lower-triangular factor with a nonzero diagonal: its Gram matrix has rank exactly n.
            var n = rng.Next(2, 6);
            var factor = RandomFactor(n, n, rng);
            for (var i = 0; i < n; i++)
            {
                factor[i, i] = rng.Next(1, 10) * (rng.Next(0, 2) == 0 ? -1 : 1) / 100m;
                for (var k = i + 1; k < n; k++)
                {
                    factor[i, k] = 0m;
                }
            }

            Assert.Equal(n, CholeskyDecomposition<decimal>.DecomposePivoted(Gram(factor)).Rank);
        }
    }

    [Fact]
    public void DefaultTolerance_SmallScaleDecimal_IsPositive()
    {
        // maxDiag 0.0004: 3·u·0.0004 lies far below decimal's grid.
        var a = new decimal[,]
        {
            { 0.0004m, 0.0001m, 0m },
            { 0.0001m, 0.0003m, 0.0001m },
            { 0m, 0.0001m, 0.0002m },
        };

        Assert.True(CholeskyDecomposition<decimal>.DefaultTolerance(a) > 0m);
    }

    // The decimal default is Higham's rule plus n units of decimal's absolute grid: n·u·max(diag(a)) + n·1e-28, where
    // u = 0.5e-28 is half decimal's spacing at one. With even n and an integer max diagonal the products are exact, so the
    // value is pinned to the grid unit: a tolerance ten times, or 10^8 times, larger (or smaller) fails.
    [Theory]
    [InlineData(2, 1)]
    [InlineData(2, 1000)]
    [InlineData(4, 4)]
    [InlineData(6, 1)]
    [InlineData(8, 1000)]
    public void DefaultTolerance_Decimal_IsHighamsRulePlusNUnitsOfTheAbsoluteGrid(int n, int maxDiag)
    {
        var a = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            a[i, i] = (i == 0) ? maxDiag : 1m / (i + 1);
        }

        // u·n = (n/2) grid units for even n: 0.5e-28 itself is not a decimal.
        var grid = 1e-28m;
        var expected = ((n / 2) * grid * maxDiag) + (n * grid);

        Assert.Equal(expected, CholeskyDecomposition<decimal>.DefaultTolerance(a));
    }

    // A genuine pivot below any scale a coarser grid could resolve must keep its rank: the tolerance only has to clear the
    // rounding noise of the grid (n·1e-28), not the pivot. Pivots are k·1e-28 with k fixed, so the true rank is known
    // without the decomposition: the pivots at or below n·1e-28 are noise (the default tolerance is at least that), the
    // pivots from 3n·1e-28 up are genuine.
    [Fact]
    public void DecomposePivoted_DecimalPivotsBetweenTheGridNoiseAndAnEnlargedTolerance_KeepTheirRank()
    {
        // n = 2 (tolerance about 2e-28 to 3e-28): a 5e-28 pivot is genuine; 1e-24 and 1e-21 sit far above the grid.
        Assert.Equal(2, CholeskyDecomposition<decimal>.DecomposePivoted(Diagonal(1e-3m, 5e-28m)).Rank);
        Assert.Equal(2, CholeskyDecomposition<decimal>.DecomposePivoted(Diagonal(1e-3m, 1e-24m)).Rank);
        Assert.Equal(2, CholeskyDecomposition<decimal>.DecomposePivoted(Diagonal(1m, 1e-25m)).Rank);
        Assert.Equal(3, CholeskyDecomposition<decimal>.DecomposePivoted(Diagonal(1e-4m, 1e-4m, 1e-22m)).Rank);

        // Noise at the grid is dropped.
        Assert.Equal(1, CholeskyDecomposition<decimal>.DecomposePivoted(Diagonal(1e-3m, 1e-28m)).Rank);
    }

    [Theory]
    [InlineData(20261029)]
    [InlineData(20261030)]
    [InlineData(20261031)]
    public void DecomposePivoted_SeededDecimalDiagonalWithPivotsOnTheGrid_ReportsTheKnownRank(int seed)
    {
        var rng = new System.Random(seed);
        var wrong = new List<string>();
        for (var trial = 0; trial < 200; trial++)
        {
            var n = rng.Next(2, 9);
            var pivots = new decimal[n];
            var genuine = 0;
            for (var i = 0; i < n; i++)
            {
                if (i == 0)
                {
                    pivots[i] = rng.Next(1, 10) / 10m;
                    genuine++;
                }
                else if (rng.Next(0, 3) == 0)
                {
                    // Noise: fewer than n grid units, below any default tolerance.
                    pivots[i] = rng.Next(1, n) * 1e-28m;
                }
                else
                {
                    // Genuine: from 3n grid units (about 3e-28 to 2.4e-27) up to 1e-21, a log-uniform magnitude.
                    var exponent = rng.Next(0, 8);
                    pivots[i] = (rng.Next(3 * n, 10 * n) * 1e-28m) * (decimal)Math.Pow(10, exponent);
                    genuine++;
                }
            }

            var actual = CholeskyDecomposition<decimal>.DecomposePivoted(Diagonal(pivots)).Rank;
            if (actual != genuine)
            {
                wrong.Add($"trial {trial}: n={n} pivots [{string.Join(", ", pivots)}] true rank {genuine}, reported {actual}");
            }
        }

        Assert.True(wrong.Count == 0, string.Join(Environment.NewLine, wrong.Take(5)));
    }

    // The IEEE default is Higham's rule n·u·max(diag(a)) with u = 2^-53, to the bit.
    [Fact]
    public void DefaultTolerance_Double_IsHighamsRelativeRuleToTheBit()
    {
        var rng = new System.Random(20261028);
        for (var trial = 0; trial < 50; trial++)
        {
            var n = rng.Next(1, 12);
            var a = new double[n, n];
            var maxDiag = 0.0;
            for (var i = 0; i < n; i++)
            {
                a[i, i] = Math.Pow(10, rng.Next(-300, 300)) * rng.NextDouble();
                maxDiag = Math.Max(maxDiag, a[i, i]);
            }

            var expected = (n * Math.Pow(2, -52) / 2) * maxDiag;
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(CholeskyDecomposition<double>.DefaultTolerance(a)));
        }
    }

    private static decimal[,] Diagonal(params decimal[] pivots)
    {
        var a = new decimal[pivots.Length, pivots.Length];
        for (var i = 0; i < pivots.Length; i++)
        {
            a[i, i] = pivots[i];
        }

        return a;
    }

    private static decimal[,] RandomFactor(int n, int rank, System.Random rng)
    {
        var factor = new decimal[n, rank];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < rank; k++)
            {
                factor[i, k] = rng.Next(-9, 10) / 100m;
            }
        }

        return factor;
    }

    private static decimal[,] Gram(decimal[,] factor)
    {
        var n = factor.GetLength(0);
        var a = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = 0m;
                for (var k = 0; k < factor.GetLength(1); k++)
                {
                    sum += factor[i, k] * factor[j, k];
                }

                a[i, j] = sum;
            }
        }

        return a;
    }
}
