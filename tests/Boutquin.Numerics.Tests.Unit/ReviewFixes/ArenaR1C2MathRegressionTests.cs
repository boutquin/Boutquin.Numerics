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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1C2MathRegressionTests
{
    [Fact]
    public void Jacobi_HalfOffDiagonalSquareUnderflows_StillDiagonalizesMatrix()
    {
        var diagonal = Half.CreateChecked(0.00020003318786621094);
        var offDiagonal = Half.CreateChecked(0.00010001659393310547);
        Half[,] matrix = { { diagonal, offDiagonal }, { offDiagonal, diagonal } };

        var result = JacobiEigenDecomposition<Half>.Decompose(matrix);

        result.Values.Should().Equal(
            Half.CreateChecked(0.0002999305725097656),
            Half.CreateChecked(0.00010001659393310547));
    }

    [Fact]
    public void Jacobi_OppositeLargeDiagonals_DoesNotOverflowRotationDifference()
    {
        double[,] matrix = { { -1e308, 1e307 }, { 1e307, 1e308 } };

        var result = JacobiEigenDecomposition<double>.Decompose(matrix);

        (result.Values[0] / 1e308).Should().BeApproximately(Math.Sqrt(1.01), 1e-14);
        (result.Values[1] / 1e308).Should().BeApproximately(-Math.Sqrt(1.01), 1e-14);
    }

    [Fact]
    public void Higham_PositiveInfinityTolerance_IsRejected()
    {
        double[,] matrix = { { 1.0, 2.0 }, { 2.0, 1.0 } };

        var act = () => NearestPsdProjection<double>.Higham(matrix, 100, double.PositiveInfinity);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("tolerance");
    }

    [Fact]
    public void EigenClip_LargeFiniteSymmetricPsdMatrix_RemainsFinite()
    {
        double[,] matrix = { { 1e308, 1e308 }, { 1e308, 1e308 } };

        var result = NearestPsdProjection<double>.EigenClip(matrix);

        result[0, 0].Should().BeApproximately(1e308, 1e293);
        result[0, 1].Should().BeApproximately(1e308, 1e293);
        result[1, 0].Should().BeApproximately(1e308, 1e293);
        result[1, 1].Should().BeApproximately(1e308, 1e293);
    }

    [Fact]
    public void Pca_LargeFiniteEigenvalues_HaveNormalizedExplainedVariance()
    {
        double[,] covariance = { { 1e308, 0.0 }, { 0.0, 1e308 } };

        var result = PrincipalComponentAnalysis<double>.Decompose(covariance);

        result.ExplainedVarianceRatio.Should().Equal(0.5, 0.5);
        result.CumulativeExplainedVariance.Should().Equal(0.5, 1.0);
    }
}
