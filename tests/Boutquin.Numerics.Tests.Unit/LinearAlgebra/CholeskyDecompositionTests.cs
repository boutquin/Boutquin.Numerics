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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class CholeskyDecompositionTests
{
    [Fact]
    public void Decompose_3x3PositiveDefinite_ReturnsLowerTriangular()
    {
        // A = L * L^T where L = [[2,0,0],[6,1,0],[-8,5,3]]
        // A = [[4,12,-16],[12,37,-43],[-16,-43,98]]
        var a = new decimal[,]
        {
            { 4, 12, -16 },
            { 12, 37, -43 },
            { -16, -43, 98 },
        };

        var lower = CholeskyDecomposition.Decompose(a);

        // T = decimal: the factor here is exact (perfect squares / integers), so assert to the
        // decimal precision regime (~1e-20) rather than the double-cast 1e-10 used previously.
        const decimal tol = 1e-20m;
        lower[0, 0].Should().BeApproximately(2m, tol);
        lower[1, 0].Should().BeApproximately(6m, tol);
        lower[1, 1].Should().BeApproximately(1m, tol);
        lower[2, 0].Should().BeApproximately(-8m, tol);
        lower[2, 1].Should().BeApproximately(5m, tol);
        lower[2, 2].Should().BeApproximately(3m, tol);

        // Upper triangle should be zero.
        lower[0, 1].Should().Be(0m);
        lower[0, 2].Should().Be(0m);
        lower[1, 2].Should().Be(0m);
    }

    [Fact]
    public void Solve_3x3System_ReturnsCorrectSolution()
    {
        var a = new decimal[,]
        {
            { 4, 12, -16 },
            { 12, 37, -43 },
            { -16, -43, 98 },
        };

        var lower = CholeskyDecomposition.Decompose(a);

        // A * x = b where b = A * [1,2,3]
        var b = new decimal[]
        {
            4 * 1 + 12 * 2 + -16 * 3,
            12 * 1 + 37 * 2 + -43 * 3,
            -16 * 1 + -43 * 2 + 98 * 3,
        };

        var x = CholeskyDecomposition.Solve(lower, b);

        // T = decimal: tighten from the previous double-cast 1e-8 to the decimal regime (~1e-20).
        x[0].Should().BeApproximately(1m, 1e-20m);
        x[1].Should().BeApproximately(2m, 1e-20m);
        x[2].Should().BeApproximately(3m, 1e-20m);
    }

    [Fact]
    public void Decompose_NonPositiveDefinite_Throws()
    {
        var a = new decimal[,] { { -1, 0 }, { 0, 1 } };
        var act = () => CholeskyDecomposition.Decompose(a);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Decompose_NonFiniteDiagonal_Throws()
    {
        var a = new double[,] { { double.NaN } };

        var act = () => CholeskyDecomposition<double>.Decompose(a);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DecomposePivoted_NonFiniteDiagonal_Throws()
    {
        var a = new double[,] { { double.NaN } };

        var act = () => CholeskyDecomposition<double>.DecomposePivoted(a);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Decompose_NonFiniteLowerEntry_Throws()
    {
        var a = new double[,] { { 1, 0 }, { double.NaN, 1 } };

        var act = () => CholeskyDecomposition<double>.Decompose(a);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DecomposePivoted_NonFiniteTolerance_Throws()
    {
        var a = new double[,] { { 1 } };

        var act = () => CholeskyDecomposition<double>.DecomposePivoted(a, double.NaN, null);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .Which.ParamName.Should().Be("tolerance");
    }

    [Fact]
    public void DecomposePivoted_OverflowedSchurDiagonal_Throws()
    {
        var a = new double[,] { { 1, 1e308 }, { 1e308, 1 } };

        var act = () => CholeskyDecomposition<double>.DecomposePivoted(a, 0, maxRank: 1);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void DecomposePivoted_SemiDefiniteMatrix_ReturnsReducedRank()
    {
        // Rank-1 matrix: [1,1,1]^T * [1,1,1] = [[1,1,1],[1,1,1],[1,1,1]]
        var a = new decimal[,]
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 },
        };

        var result = CholeskyDecomposition.DecomposePivoted(a, tolerance: 1e-10m);

        result.Rank.Should().Be(1);
        result.Permutation.Should().HaveCount(3);
    }

    [Fact]
    public void DecomposePivoted_GenericRankDeficientFactor_HasNByRankShapeAndReconstructs()
    {
        var a = new double[,]
        {
            { 1, 0, 1 },
            { 0, 4, 0 },
            { 1, 0, 1 },
        };

        var result = CholeskyDecomposition<double>.DecomposePivoted(a);
        var lower = result.Lower;
        var permutation = result.Permutation;

        result.Rank.Should().Be(2);
        lower.GetLength(0).Should().Be(3);
        lower.GetLength(1).Should().Be(result.Rank);
        permutation.Should().BeEquivalentTo(new[] { 0, 1, 2 });

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var reconstructed = 0.0;
                for (var k = 0; k < result.Rank; k++)
                {
                    reconstructed += lower[i, k] * lower[j, k];
                }

                reconstructed.Should().BeApproximately(a[permutation[i], permutation[j]], 1e-12);
            }
        }
    }

    [Fact]
    public void DecomposePivoted_DecimalRankDeficientFactor_HasNByRankShapeAndReconstructs()
    {
        var a = new decimal[,]
        {
            { 1, 0, 1 },
            { 0, 4, 0 },
            { 1, 0, 1 },
        };

        var result = CholeskyDecomposition<decimal>.DecomposePivoted(a);
        var lower = result.Lower;
        var permutation = result.Permutation;

        result.Rank.Should().Be(2);
        lower.GetLength(0).Should().Be(3);
        lower.GetLength(1).Should().Be(result.Rank);

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var reconstructed = 0m;
                for (var k = 0; k < result.Rank; k++)
                {
                    reconstructed += lower[i, k] * lower[j, k];
                }

                reconstructed.Should().Be(a[permutation[i], permutation[j]]);
            }
        }
    }

    [Fact]
    public void DecomposePivoted_ZeroRankFactor_HasNoColumns()
    {
        var a = new double[,] { { 4, 0 }, { 0, 9 } };

        var capped = CholeskyDecomposition<double>.DecomposePivoted(a, 0, maxRank: 0);
        var zero = CholeskyDecomposition<double>.DecomposePivoted(new double[2, 2]);

        capped.Rank.Should().Be(0);
        capped.Lower.GetLength(0).Should().Be(2);
        capped.Lower.GetLength(1).Should().Be(0);
        zero.Rank.Should().Be(0);
        zero.Lower.GetLength(0).Should().Be(2);
        zero.Lower.GetLength(1).Should().Be(0);
    }

    [Fact]
    public void DecomposePivoted_EarlyTermination_RespectsMaxRank()
    {
        var a = new decimal[,]
        {
            { 4, 2, 1 },
            { 2, 5, 3 },
            { 1, 3, 6 },
        };

        var result = CholeskyDecomposition.DecomposePivoted(a, maxRank: 2);

        result.Rank.Should().Be(2);
        result.Lower.GetLength(1).Should().Be(2);
    }

    [Fact]
    public void DecomposePivoted_FullRank_MatchesUnpivoted()
    {
        var a = new decimal[,]
        {
            { 4, 2, 1 },
            { 2, 5, 3 },
            { 1, 3, 6 },
        };

        var result = CholeskyDecomposition.DecomposePivoted(a);

        result.Rank.Should().Be(3);
        result.Lower.GetLength(0).Should().Be(3);
        result.Lower.GetLength(1).Should().Be(result.Rank);

        // Verify L * L^T reconstructs P^T * A * P.
        var n = 3;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = 0m;
                for (var k = 0; k < result.Rank; k++)
                {
                    sum += result.Lower[i, k] * result.Lower[j, k];
                }

                // Compare against permuted A. T = decimal: tighten from the previous double-cast
                // 1e-6 to the decimal regime — the pivoted factor uses NumericPrecision<decimal>.Sqrt
                // (Newton-Raphson), which converges to ~1e-20 here.
                var pi = result.Permutation[i];
                var pj = result.Permutation[j];
                Math.Abs(sum - a[pi, pj]).Should().BeLessThan(1e-20m,
                    $"Reconstruction mismatch at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void DecomposePivoted_DefaultOverload_UsesHighamRelativeTolerance_NotAbsoluteZero()
    {
        // A = B*B^T for B (4x2): exact mathematical rank 2, but double-precision Cholesky
        // accumulates rounding residue at the third pivot (~3e-8, verified below to be nonzero
        // and far above double's own noise floor) that an absolute zero-tolerance default
        // wrongly admits as a genuine extra pivot. Higham's (1990) eq. 4.3 relative rule
        // (ε = n·u applied to the first pivot) correctly stops at rank 2.
        double[,] b =
        {
            { 1, 2 },
            { 3, 1 },
            { 2, -1 },
            { 0.5, 4 },
        };
        var a = new double[4, 4];
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                a[i, j] = (b[i, 0] * b[j, 0]) + (b[i, 1] * b[j, 1]);
            }
        }

        var result = CholeskyDecomposition<double>.DecomposePivoted(a);

        result.Rank.Should().Be(2,
            "the default overload must apply Higham's relative stopping rule, not an absolute " +
            "zero tolerance that admits floating-point rounding residue as a spurious extra pivot");
    }

    [Fact]
    public void DecomposePivoted_DecimalDefaultTolerance_UsesHighamRelativeTolerance_NotAbsoluteZero()
    {
        // L*L^T with an intentionally tiny (1e-14) third-diagonal contribution: the Schur
        // complement at the third pivot reduces EXACTLY (decimal arithmetic is exact for these
        // operands) to 1e-28m — noise-level relative to the matrix scale (max diagonal 16), but
        // strictly greater than an absolute-zero cutoff. Higham's default
        // (n·u·max diag, computed from decimal's own
        // machine epsilon) must reject it.
        var l = new decimal[,]
        {
            { 4m, 0m, 0m },
            { 2m, 3m, 0m },
            { 1m, 0.5m, 0.00000000000001m },
        };
        var a = new decimal[3, 3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var sum = 0m;
                for (var k = 0; k < 3; k++)
                {
                    sum += l[i, k] * l[j, k];
                }

                a[i, j] = sum;
            }
        }

        var result = CholeskyDecomposition<decimal>.DecomposePivoted(a);

        result.Rank.Should().Be(2,
            "the decimal default must use Higham's relative stopping rule, " +
            "not a literal absolute-zero cutoff that admits noise-level residue as a real pivot");
    }
}
