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

public sealed class C2IndependentBoundaryTests
{
    [Fact]
    public void PivotedCholesky_EqualLargestDiagonals_SelectsFirstActiveIndex()
    {
        // Equal diagonal Gram vectors e0 and e1 have no ordering ambiguity in
        // Higham's smallest-active-index rule. A >= selector incorrectly chooses 1.
        var result = CholeskyDecomposition<double>.DecomposePivoted(
            new[,] { { 4d, 0d, 0d }, { 0d, 4d, 0d }, { 0d, 0d, 1d } }, 0d, 1);
        result.Permutation.Should().Equal(0, 1, 2);
        result.Lower[0, 0].Should().Be(2d);
    }

    [Fact]
    public void PivotedCholesky_ResidualEqualsCutoff_StopsBeforePivot()
    {
        // All arithmetic is exactly binary representable: after accepting 1,
        // the independent second coordinate has residual variance 1/4.
        var result = CholeskyDecomposition<double>.DecomposePivoted(
            new[,] { { 1d, 0d }, { 0d, .25d } }, .25d, null);
        result.Rank.Should().Be(1);
        result.Lower.GetLength(1).Should().Be(1);
    }

    [Fact]
    public void PivotedCholesky_OneStep_RetainsIndependentSchurResidual()
    {
        // A=[[4,2],[2,2]]. Conditioning the second Gram vector on the first
        // leaves squared length 2-(2*2)/4=1. This is not a complete factorization.
        var result = CholeskyDecomposition<double>.DecomposePivoted(
            new[,] { { 4d, 2d }, { 2d, 2d } }, 0d, 1);
        result.Rank.Should().Be(1);
        result.Permutation.Should().Equal(0, 1);
        result.Lower[0, 0].Should().Be(2d);
        result.Lower[1, 0].Should().Be(1d);
        (2d - result.Lower[1, 0] * result.Lower[1, 0]).Should().Be(1d);
    }

    [Fact]
    public void GaussianElimination_ColumnMaximumEqualsFloor_IsEligible()
    {
        // Independent diagonal system, exact decimal solution (3,7). The
        // maximum in column 1 equals 1e-14 times the original matrix maximum.
        decimal[]? result = null;
        FluentActions.Invoking(() => result = GaussianElimination<decimal>.Solve(
            new[,] { { 1m, 0m }, { 0m, 1e-14m } }, [3m, 7e-14m], 17))
            .Should().NotThrow("equality at the declared pivot floor is eligible");
        result.Should().Equal(3m, 7m);
    }
}
