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

using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.LinearAlgebra;
using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class ArenaSuccessorC2Tests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Monotone_MixedWidths_PreservesBoundedHermiteValue(bool cached)
    {
        double[] xs = [0, 1e-310, 1];
        double[] ys = [0, 1, 1];
        var result = cached ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(5e-311)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(5e-311, xs, ys);
        result.Should().BeApproximately(.625, 1e-12);
    }

    [Fact]
    public void Convex_MixedWidths_PreservesBoundedIntegral()
    {
        MonotoneConvexInterpolator<double>.Instance.Interpolate(5e-311, [0, 1e-310, 1], [0, 1, 1])
            .Should().BeApproximately(.5, 1e-12);
    }

    [Fact]
    public void NaturalCubic_UnrepresentableCoefficients_RefusesSuccess()
    {
        Action act = () => _ = new CubicSplineInterpolator<double>([0, 1e-310, 1], [0, 1, 0]);
        act.Should().Throw<InvalidOperationException>().WithMessage("*coefficients*");
    }

    [Fact]
    public void NaturalCubic_MixedWidths_ReproducesLine()
    {
        Action act = () => new CubicSplineInterpolator<double>([0, 1e-310, 1], [0, 1e-310, 1])
            .Interpolate(.5).Should().BeApproximately(.5, 1e-14);
        act.Should().NotThrow();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CollapsingCoordinates_RejectBeforeReturningWrongKnot(int kind)
    {
        double[] xs = [0, 1e-300, 2e-300, 1e308];
        double[] ys = [0, 1, 2, 3];
        Action act = () => _ = kind switch
        {
            0 => new CubicSplineInterpolator<double>(xs, ys).Interpolate(xs[1]),
            1 => MonotoneCubicInterpolator<double>.Instance.Interpolate(xs[1], xs, ys),
            2 => new MonotoneCubicSpline<double>(xs, ys).Interpolate(xs[1]),
            _ => MonotoneConvexInterpolator<double>.Instance.Interpolate(xs[1], xs, ys),
        };
        act.Should().Throw<InvalidOperationException>().WithMessage("*distinct*");
    }

    [Fact]
    public void Jacobi_UnrepresentableSpectrum_RejectsNumericalExhaustion()
    {
        var v = double.MaxValue;
        Action act = () => JacobiEigenDecomposition<double>.Decompose(new[,] { { v, v }, { v, v } });
        act.Should().Throw<InvalidOperationException>().WithMessage("*finite*");
    }

    [Fact]
    public void Jacobi_OffDiagonalNorm_IncludesBothTriangles()
    {
        var result = JacobiEigenDecomposition<double>.Decompose(new[,] { { 1d, 8e-16 }, { 8e-16, 1d } });
        Math.Abs(result.Vectors[0, 1]).Should().BeGreaterThan(.7);
    }

    [Fact]
    public void Cholesky_DefaultCutoff_UsesUnitRoundoff()
    {
        CholeskyDecomposition<double>.DecomposePivoted(new[,] { { 1d, 0 }, { 0, 3e-16 } }).Rank.Should().Be(2);
    }

    [Fact]
    public void Pca_TinyDecimalVariance_Standardizes()
    {
        Action act = () => PrincipalComponentAnalysis.FromReturns(new[,] { { -1e-20m }, { 0m }, { 1e-20m } }, true)
            .Eigenvalues[0].Should().Be(1m);
        act.Should().NotThrow();
    }

    [Fact]
    public void Pca_LegacyThreshold_DoesNotRoundRequestDown()
    {
        var result = PrincipalComponentAnalysis.Decompose(new[,] { { 1m, 0m }, { 0m, 1m } });
        result.NumComponentsForExplainedVariance(Math.BitIncrement(.5)).Should().Be(2);
    }

    [Fact]
    public void Ols_HugeResponse_PreservesFiniteExactFit()
    {
        Action act = () =>
        {
            var result = OrdinaryLeastSquares<double>.Fit(new[,] { { 1d }, { 1d } }, [1e308, 1e308], false);
            (result.Coefficients[0] / 1e308).Should().BeApproximately(1, 1e-14);
            result.ResidualSumOfSquares.Should().Be(0);
        };
        act.Should().NotThrow();
    }

    [Fact]
    public void Ols_UnrepresentableCoefficient_RefusesSuccess()
    {
        Action act = () => _ = OrdinaryLeastSquares<double>.Fit(new[,] { { .1 }, { .1 } }, [1e308, 1e308], false);
        act.Should().Throw<InvalidOperationException>();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Monotone_HalfMixedWidths_PreservesHermiteValue(bool cached)
    {
        Half[] xs = [(Half)0, (Half).00001, (Half)1];
        Half[] ys = [(Half)0, (Half)1, (Half)1];
        var query = (Half).000005;
        var u = (double)query / (double)xs[1];
        var expected = 3 * u * u - 2 * u * u * u + (u * u * u - 2 * u * u + u) * (1 + (double)xs[1]);
        var value = cached ? new MonotoneCubicSpline<Half>(xs, ys).Interpolate(query)
            : MonotoneCubicInterpolator<Half>.Instance.Interpolate(query, xs, ys);
        ((double)value).Should().BeApproximately(expected, .003);
    }
}
