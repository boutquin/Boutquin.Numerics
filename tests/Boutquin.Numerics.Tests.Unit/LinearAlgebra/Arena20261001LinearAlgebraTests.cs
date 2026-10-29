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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class Arena20261001LinearAlgebraTests
{
    [Fact]
    public void EigenClip_GridFloor_PreservesAlreadyPsdMatrix()
    {
        const decimal v = .0000000000000000000000000001m;
        NearestPsdProjection<decimal>.EigenClip(new[,] { { v, v }, { v, v } })[0, 1].Should().Be(v);
        var e = double.Epsilon;
        NearestPsdProjection<double>.EigenClip(new[,] { { e, e }, { e, e } })[0, 1].Should().Be(e);
    }

    [Fact]
    public void IsPsd_GridFloor_RejectsIndefiniteMatrixAtZeroTolerance()
    {
        const decimal v = .0000000000000000000000000001m;
        NearestPsdProjection<decimal>.IsPsd(new[,] { { 0m, v }, { v, 0m } }, 0).Should().BeFalse();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EigenClip_PositiveFloor_UsesMaximumEigenvalueFloor(int variant)
    {
        double[,] input = variant switch
        {
            0 => new[,] { { .25, 0 }, { 0, 1d } },
            1 => new[,] { { .625, .375 }, { .375, .625 } },
            2 => new double[2, 2],
            _ => new[,] { { .01, 0 }, { 0, .02 } },
        };
        var output = NearestPsdProjection<double>.EigenClip(input, .5);
        output[0, 0].Should().BeApproximately(variant == 1 ? .75 : .5, 1e-14);
        output[1, 1].Should().BeApproximately(variant == 0 ? 1 : variant == 1 ? .75 : .5, 1e-14);
        output[0, 1].Should().BeApproximately(variant == 1 ? .25 : 0, 1e-14);
    }

    [Fact]
    public void Higham_NegativeIterations_RejectsArgument()
    {
        Action act = () => NearestPsdProjection<double>.Higham(new[,] { { 1d } }, -1, 1e-12);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxIterations");
    }

    [Fact]
    public void Pca_HalfThreshold_DoesNotRoundRequestDown()
    {
        var result = PrincipalComponentAnalysis<Half>.Decompose(new[,] { { (Half)1, (Half)0 }, { (Half)0, (Half)1 } });
        result.NumComponentsForExplainedVariance(.50001).Should().Be(2);
    }

    [Fact]
    public void Cholesky_DecimalSolveOverflow_UsesFamilyFailureContract()
    {
        var factor = CholeskyDecomposition<decimal>.Decompose(new[,] { { .01m } });
        Action act = () => CholeskyDecomposition<decimal>.Solve(factor, [decimal.MaxValue]);
        act.Should().Throw<InvalidOperationException>().WithInnerException<OverflowException>();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pca_DecimalAccumulation_PreservesRepresentableResult(bool standardize)
    {
        decimal[,] input = standardize ? new[,] { { -1e20m }, { 0m }, { 1e20m } }
            : new[,] { { -1e14m }, { -1e14m }, { -1e14m }, { -1e14m }, { 1e14m }, { 1e14m }, { 1e14m }, { 1e14m } };
        PcaResult? result = null;
        var act = () => result = PrincipalComponentAnalysis.FromReturns((ReturnsMatrix)input, standardize);
        act.Should().NotThrow("the expected covariance and standard deviation are representable");
        var expected = standardize ? 1m : 1e28m / 7m * 8m;
        (result!.Eigenvalues[0] / expected).Should().BeApproximately(1m, 1e-25m);
    }
}
