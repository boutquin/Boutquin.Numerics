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

public sealed class ArenaR4B05LinearAlgebraTests
{
    [Fact]
    public void Jacobi_EqualLargeDecimalDiagonals_PreservesLiveRotation()
    {
        const decimal diagonal = 1e20m;
        const decimal pivot = 1e-10m;
        EigenResult<decimal>? result = null;

        var act = () => result = JacobiEigenDecomposition<decimal>.Decompose(
            new decimal[,] { { diagonal, pivot }, { pivot, diagonal } });

        act.Should().NotThrow();
        decimal.Abs(result!.Vectors[0, 0]).Should().BeApproximately(0.7071067811865475244008443621m, 1e-12m);
    }

    [Fact]
    public void Jacobi_EqualWideDecimalDiagonals_PreservesLiveRotation()
    {
        const decimal diagonal = 1e28m;
        const decimal pivot = 1e-10m;
        EigenResult<decimal>? result = null;

        var act = () => result = JacobiEigenDecomposition<decimal>.Decompose(
            new decimal[,] { { diagonal, pivot }, { pivot, diagonal } });

        act.Should().NotThrow();
        decimal.Abs(result!.Vectors[0, 0]).Should().BeApproximately(0.7071067811865475244008443621m, 1e-12m);
    }

    [Fact]
    public void EigenClip_FloorGreaterThanTinyScale_ClipsWithoutOverflow()
    {
        decimal[,]? result = null;
        var act = () => result = NearestPsdProjection<decimal>.EigenClip(
            new decimal[,] { { 1e-28m } }, 10m);

        act.Should().NotThrow();
        result![0, 0].Should().Be(10m);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void CholeskySolve_NonFiniteRightHandSide_IsRejected(double invalid)
    {
        var act = () => CholeskyDecomposition<double>.Solve(new double[,] { { 1.0 } }, [invalid]);

        act.Should().Throw<ArgumentException>().WithParameterName("b");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CholeskySolve_NonFiniteUsedFactor_IsRejected(bool diagonal)
    {
        var lower = new double[,] { { 1, 0 }, { 0, 1 } };
        lower[1, diagonal ? 1 : 0] = double.NaN;
        var act = () => CholeskyDecomposition<double>.Solve(lower, [1.0, 2.0]);

        act.Should().Throw<ArgumentException>().WithParameterName("lower");
    }

    [Fact]
    public void CholeskySolve_FiniteInputsWithUnrepresentableSolution_Throws()
    {
        var lower = CholeskyDecomposition<double>.Decompose(new double[,] { { 1e-320 } });
        var act = () => CholeskyDecomposition<double>.Solve(lower, [1.0]);

        act.Should().Throw<InvalidOperationException>().WithMessage("*non-finite*");
    }
}
