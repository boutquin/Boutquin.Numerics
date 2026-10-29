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
using Boutquin.Numerics.LinearAlgebra.Internal;
using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class Arena20261003C2RegressionTests
{
    [Fact]
    public void R4_Cholesky_FiniteSolutionSurvivesOverflowingNumerator()
    {
        Action solve = () =>
        {
            var x = CholeskyDecomposition<double>.Solve(new[,] { { 1e100, 0d }, { 1e100, 1e100 } }, [1e308, -1e308]);
            (x[0] / 1e108).Should().BeApproximately(3, 1e-14);
            (x[1] / 1e108).Should().BeApproximately(-2, 1e-14);
        };
        solve.Should().NotThrow();
    }

    [Fact]
    public void R4_Gaussian_CommonScalePreservesSolution()
    {
        Action solve = () => GaussianElimination<double>.Solve(new[,] { { 1e308, 0d }, { 1e308, 1e308 } }, [1e308, -1e308], 42)
            .Should().Equal(1d, -2d);
        solve.Should().NotThrow();
    }

    [Fact]
    public void R4_EigenClip_IndependentSmallDiagonalSurvives()
    {
        var result = NearestPsdProjection<decimal>.EigenClip(new[,] { { 1e28m, 0m }, { 0m, 1e-28m } });
        result[1, 1].Should().Be(1e-28m);
    }

    [Fact]
    public void R4_Pca_TranslationPreservesCovariance()
    {
        PrincipalComponentAnalysis.FromReturns(new[,] { { 1e28m }, { 1e28m + 1m } })
            .Eigenvalues[0].Should().Be(.5m);
    }

    [Fact]
    public void R4_MonotoneCubic_LocalSmallSegmentSurvives()
    {
        var value = MonotoneCubicInterpolator<double>.Instance.Interpolate(.5, [0, 1, 2], [0, 1e-100, 1e308]);
        (value / 1e-100).Should().BeApproximately(.125, 1e-14);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void R4_MonotoneCubic_WeightedLargeTangentPreservesInteriorKnot(bool cached)
    {
        double[] xs = [0, 1, 2], ys = [0, 1e-100, 1.5e308];
        var value = cached
            ? new MonotoneCubicSpline<double>(xs, ys).Interpolate(1)
            : MonotoneCubicInterpolator<double>.Instance.Interpolate(1, xs, ys);
        value.Should().Be(1e-100);
    }

    [Fact]
    public void R4_MonotoneSpline_LocalSmallSegmentSurvives()
    {
        var value = new MonotoneCubicSpline<double>([0, 1, 2], [0, 1e-100, 1e308]).Interpolate(.5);
        (value / 1e-100).Should().BeApproximately(.125, 1e-14);
    }

    [Fact]
    public void R4_MonotoneConvex_LocalSmallSegmentSurvives()
    {
        var value = MonotoneConvexInterpolator<double>.Instance.Interpolate(.5, [0, 1, 2], [0, 1e-100, 1e308]);
        (value / 1e-100).Should().BeApproximately(.25, 1e-14);
    }

    [Fact]
    public void R4_PcaProjection_WeightedCenterSurvives()
    {
        var fit = new PcaResult<double>
        {
            Eigenvalues = [1],
            Eigenvectors = new[,] { { .5 }, { Math.Sqrt(.75) } },
            ExplainedVarianceRatio = [1],
            CumulativeExplainedVariance = [1],
            Mean = [-1e308, 0],
        };
        (fit.Project(new[,] { { 1e308, 0d } }, 1)[0, 0] / 1e308).Should().BeApproximately(1, 1e-14);
    }

    [Fact]
    public void R4_PcaProjection_CancellationSurvives()
    {
        var v = 1 / Math.Sqrt(6);
        var fit = new PcaResult<double>
        {
            Eigenvalues = [1],
            Eigenvectors = new[,] { { v }, { v }, { v }, { v }, { v }, { v } },
            ExplainedVarianceRatio = [1],
            CumulativeExplainedVariance = [1],
            Mean = [],
        };
        (fit.Project(new[,] { { 1e308, 1e308, 1e308, 1e308, 1e308, -1e308 } }, 1)[0, 0] / 1e308)
            .Should().BeApproximately(4 / Math.Sqrt(6), 1e-14);
    }

    [Fact]
    public void R4_PcaProjection_DecimalWeightedCenterSurvives()
    {
        var fit = new PcaResult
        {
            Eigenvalues = [1],
            Eigenvectors = new[,] { { .6m }, { .8m } },
            ExplainedVarianceRatio = [1],
            CumulativeExplainedVariance = [1],
            Mean = [-6e28m, 0],
        };
        Action project = () => fit.Project(new[,] { { 6e28m, 0m } }, 1)[0, 0].Should().Be(7.2e28m);
        project.Should().NotThrow();
    }

    [Fact]
    public void R4_QrUpper_FiniteQuotientSurvives()
    {
        Action solve = () => HouseholderQr<double>.SolveUpperTriangular(new[,] { { 0d, 1e308 }, { 0d, 0d } },
            [1e308, 1e308], [1e308, -1e308], 2, [0, 1]).Should().Equal(2d, -1d);
        solve.Should().NotThrow();
    }

    [Fact]
    public void R4_QrTranspose_FiniteQuotientSurvives()
    {
        HouseholderQr<double>.SolveUpperTriangularTranspose(new[,] { { 0d, 1e308 }, { 0d, 0d } },
            [1e308, 1e308], [1e308, -1e308], 2, [0, 1]).Should().Equal(1d, -2d);
    }

    [Fact]
    public void Ols_LargeFloatReflectorAndResponse_PreservesExactFit()
    {
        var designScale = MathF.ScaleB(1, 40);
        var responseScale = MathF.ScaleB(1, 88);
        Action fit = () => OrdinaryLeastSquares<float>.Fit(
            new[,] { { 3 * designScale }, { 4 * designScale } },
            [3 * responseScale, 4 * responseScale], false)
            .Coefficients[0].Should().Be(MathF.ScaleB(1, 48));
        fit.Should().NotThrow();
    }

    [Fact]
    public void NaturalSpline_DecimalQuantumAndWideInterval_ReproducesLine()
    {
        decimal[] knots = [0, 1e-28m, 10];
        Action interpolate = () => new CubicSplineInterpolator<decimal>(knots, knots)
            .Interpolate(1m).Should().Be(1m);
        interpolate.Should().NotThrow();
    }

    [Fact]
    public void StandardizedPca_UnrepresentableRawDeviation_PreservesCorrelation()
    {
        decimal[,] data = { { -6e28m }, { 6e28m }, { 6e28m }, { 6e28m } };
        Action fit = () => PrincipalComponentAnalysis.FromReturns(data, true)
            .Eigenvalues[0].Should().Be(1m);
        fit.Should().NotThrow();
    }

    [Fact]
    public void StandardizedPcaProjection_UnrepresentableRawDeviation_PreservesScores()
    {
        var fit = new PcaResult
        {
            Eigenvalues = [1m],
            Eigenvectors = new[,] { { 1m } },
            ExplainedVarianceRatio = [1m],
            CumulativeExplainedVariance = [1m],
            Mean = [3e28m],
            StandardDeviations = [6e28m],
        };
        Action project = () =>
        {
            var scores = fit.Project(new[,] { { -6e28m }, { 6e28m } }, 1);
            scores[0, 0].Should().Be(-1.5m);
            scores[1, 0].Should().Be(.5m);
        };
        project.Should().NotThrow();
    }

    [Fact]
    public void Jacobi_DecimalUnrepresentableEigenvalue_ThrowsDocumentedException()
    {
        var maximum = decimal.MaxValue;
        Action decompose = () => JacobiEigenDecomposition<decimal>.Decompose(
            new[,] { { maximum, maximum }, { maximum, maximum } });
        decompose.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Higham_ZeroTolerance_ExactFixedPointReturnsIdentity()
    {
        Action act = () => NearestPsdProjection<decimal>.Higham(new[,] { { 1m, 0m }, { 0m, 1m } }, 1, 0m);
        act.Should().NotThrow();
    }

    [Fact]
    public void Higham_LargeDecimalDiagonal_ReturnsIdentity()
    {
        decimal[,]? result = null;
        Action act = () => result = NearestPsdProjection<decimal>.Higham(new[,] { { 1e20m, 0m }, { 0m, 1e20m } });
        act.Should().NotThrow();
        result![0, 0].Should().Be(1m);
        result[0, 1].Should().Be(0m);
        result[1, 1].Should().Be(1m);
    }

    [Fact]
    public void Ols_LargeFiniteDesign_ZeroResponseHasZeroCoefficients()
    {
        Action act = () =>
        {
            var result = OrdinaryLeastSquares<double>.Fit(new[,] { { 1e308, 9e307 }, { 1e308, 8e307 }, { 0d, 1e307 } }, [0d, 0d, 0d], false);
            result.Coefficients.Should().OnlyContain(value => value == 0d);
        };
        act.Should().NotThrow();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void MixedDecimalWidths_PreserveNonflatWideInterval(int kind)
    {
        decimal[] xs = [0m, 1e-28m, 10m];
        decimal[] ys = [0m, 1m, 2m];
        var actual = kind switch
        {
            0 => MonotoneCubicInterpolator<decimal>.Instance.Interpolate(2.5m, xs, ys),
            1 => new MonotoneCubicSpline<decimal>(xs, ys).Interpolate(2.5m),
            _ => MonotoneConvexInterpolator<decimal>.Instance.Interpolate(2.5m, xs, ys),
        };
        // Limiting wide-interval polynomial: 2-(1-u)^3 (cubic), 1+2u-u^2 (convex).
        actual.Should().BeApproximately(kind == 2 ? 1.4375m : 1.578125m, 1e-25m);
    }
}
