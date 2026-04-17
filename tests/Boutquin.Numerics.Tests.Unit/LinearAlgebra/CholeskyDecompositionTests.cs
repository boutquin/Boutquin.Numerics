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

        ((double)lower[0, 0]).Should().BeApproximately(2.0, 1e-10);
        ((double)lower[1, 0]).Should().BeApproximately(6.0, 1e-10);
        ((double)lower[1, 1]).Should().BeApproximately(1.0, 1e-10);
        ((double)lower[2, 0]).Should().BeApproximately(-8.0, 1e-10);
        ((double)lower[2, 1]).Should().BeApproximately(5.0, 1e-10);
        ((double)lower[2, 2]).Should().BeApproximately(3.0, 1e-10);

        // Upper triangle should be zero.
        ((double)lower[0, 1]).Should().BeApproximately(0.0, 1e-10);
        ((double)lower[0, 2]).Should().BeApproximately(0.0, 1e-10);
        ((double)lower[1, 2]).Should().BeApproximately(0.0, 1e-10);
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

        ((double)x[0]).Should().BeApproximately(1.0, 1e-8);
        ((double)x[1]).Should().BeApproximately(2.0, 1e-8);
        ((double)x[2]).Should().BeApproximately(3.0, 1e-8);
    }

    [Fact]
    public void Decompose_NonPositiveDefinite_Throws()
    {
        var a = new decimal[,] { { -1, 0 }, { 0, 1 } };
        var act = () => CholeskyDecomposition.Decompose(a);
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
    public void DecomposePivoted_EarlyTermination_RespectsMaxRank()
    {
        var a = new decimal[,]
        {
            { 4, 2, 1 },
            { 2, 5, 3 },
            { 1, 3, 6 },
        };

        var result = CholeskyDecomposition.DecomposePivoted(a, maxRank: 2);

        result.Rank.Should().BeInRange(1, 2);
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

                // Compare against permuted A.
                var pi = result.Permutation[i];
                var pj = result.Permutation[j];
                ((double)Math.Abs(sum - a[pi, pj])).Should().BeLessThan(1e-6,
                    $"Reconstruction mismatch at [{i},{j}]");
            }
        }
    }
}
