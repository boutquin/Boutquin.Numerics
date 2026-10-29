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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1C4RegressionTests
{
    private const decimal LargeDecimal = 39614081257132168796771975167m;

    [Fact]
    public void SampleCovariance_ConstantFiniteDoubleAtLargeOffset_IsZero()
    {
        var returns = new double[,] { { 1e308 }, { 1e308 }, { 1e308 } };

        var covariance = new SampleCovarianceEstimator<double>().Estimate(returns);

        covariance[0, 0].Should().Be(0.0);
    }

    [Fact]
    public void SampleCovariance_ConstantFiniteDecimalAtLargeOffset_IsZero()
    {
        var returns = new decimal[,] { { LargeDecimal }, { LargeDecimal }, { LargeDecimal } };

        var covariance = new SampleCovarianceEstimator<decimal>().Estimate(returns);

        covariance[0, 0].Should().Be(0m);
    }

    [Fact]
    public void WelfordMoments_LargeDecimalUnitSpacedTriple_HasExactSampleVariance()
    {
        var moments = WelfordMoments<decimal>.Compute(
            new decimal[] { LargeDecimal - 1m, LargeDecimal, LargeDecimal + 1m });

        moments.Mean.Should().Be(LargeDecimal);
        moments.Variance.Should().Be(1m);
    }

    [Fact]
    public void GraphicalLasso_ConvergenceAndSolution_WithRelativeTolerance_AreScaleEquivariant()
    {
        var covariance = new double[,]
        {
            { 1.0, 0.7, 0.2 },
            { 0.7, 1.5, 0.4 },
            { 0.2, 0.4, 1.2 },
        };
        const double scale = 1e12;
        var scaled = Scale(covariance, scale);

        const double tolerance = 1e-8;
        var baseline = GraphicalLassoSolver<double>.Solve(covariance, 0.1,
            new GraphicalLassoOptions<double>(Tolerance: tolerance));
        // Both deltaW and offdiagonal S scale together; relative tolerance is unchanged.
        var scaledResult = GraphicalLassoSolver<double>.Solve(scaled, 0.1 * scale,
            new GraphicalLassoOptions<double>(Tolerance: tolerance));

        scaledResult.Iterations.Should().Be(baseline.Iterations);
        MaxPrecisionError(baseline.Precision, scaledResult.Precision, scale)
            .Should().BeLessThan(1e-10);
    }

    [Fact]
    public void GraphicalLasso_BicSelection_IsScaleEquivariant()
    {
        var covariance = new double[,]
        {
            { 1.0, 0.7, 0.2 },
            { 0.7, 1.5, 0.4 },
            { 0.2, 0.4, 1.2 },
        };
        var lambdaGrid = new[] { 0.01, 0.1, 0.4 };
        const double scale = 1e12;
        var options = new GraphicalLassoOptions<double>(Tolerance: 1e-16);

        var baseline = GraphicalLassoSolver<double>.SolveWithBic(
            covariance, 200, lambdaGrid, options);
        var scaled = GraphicalLassoSolver<double>.SolveWithBic(
            Scale(covariance, scale),
            200,
            lambdaGrid.Select(value => value * scale).ToArray(),
            options);

        (scaled.Lambda / scale).Should().BeApproximately(baseline.Lambda, 1e-12);
    }

    [Fact]
    public void GraphicalLasso_MaterialSmallScaleAsymmetry_IsRejected()
    {
        var asymmetric = new double[,] { { 1e-8, 5e-9 }, { 0.0, 1e-8 } };

        var act = () => GraphicalLassoSolver<double>.Solve(asymmetric, 0.0);

        act.Should().Throw<ArgumentException>()
            .Where(exception => exception.ParamName == "sampleCovariance")
            .WithMessage("*symmetric*");
    }

    [Fact]
    public void SampleCovariance_CenteredDoubleProducts_DoNotOverflowBeforeDivision()
    {
        var returns = new double[,] { { -1e154 }, { 0.0 }, { 1e154 } };

        var covariance = new SampleCovarianceEstimator<double>().Estimate(returns);

        covariance[0, 0].Should().BeApproximately(1e308, 1e293);
    }

    [Fact]
    public void SampleCovariance_CenteredDecimalProducts_DoNotOverflowBeforeDivision()
    {
        var returns = new decimal[,]
        {
            { -200000000000000m },
            { 0m },
            { 200000000000000m },
        };

        var covariance = new SampleCovarianceEstimator<decimal>().Estimate(returns);

        covariance[0, 0].Should().Be(40000000000000000000000000000m);
    }

    [Fact]
    public void WelfordPearson_LargeIdenticalSeries_HasUnitCorrelation()
    {
        var pearson = new WelfordMoments<double>.Pearson();
        pearson.Add(-1e100, -1e100);
        pearson.Add(1e100, 1e100);

        pearson.Correlation.Should().BeApproximately(1.0, 1e-15);
    }

    [Fact]
    public void GarchForecast_EmptySeries_AreRejectedBeforeIndexing()
    {
        var missingVariance = new GarchFitResult<double>(
            0.0, 0.0, 0.1, 0.1, 0.8, [], [1.0], 0.0, true);
        var missingResidual = new GarchFitResult<double>(
            0.0, 0.0, 0.1, 0.1, 0.8, [1.0], [], 0.0, true);

        var varianceAct = () => missingVariance.ForecastVariance(1);
        var residualAct = () => missingResidual.ForecastVariance(1);

        varianceAct.Should().Throw<ArgumentException>()
            .WithMessage("Conditional variance series must contain at least one value.");
        residualAct.Should().Throw<ArgumentException>()
            .WithMessage("Standardized residual series must contain at least one value.");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void GarchForecast_InvalidHorizon_IsRejected(int horizon)
    {
        var result = new GarchFitResult<double>(
            0.0, 0.0, 0.1, 0.1, 0.8, [1.0], [1.0], 0.0, true);

        var act = () => result.ForecastVariance(horizon);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .Where(exception => exception.ParamName == "horizon");
    }

    [Fact]
    public void PoetFacade_FiniteThresholdOutsideDecimalRange_IsRejectedAsArgument()
    {
#pragma warning disable BNU0001
        var act = () => new PoetCovarianceEstimator(1, double.MaxValue);
#pragma warning restore BNU0001

        act.Should().Throw<ArgumentOutOfRangeException>()
            .Where(exception => exception.ParamName == "thresholdMultiplier");
    }

    [Fact]
    public void ReturnsMatrix_CopiesConstructorInputsAndMaterializedOutputs()
    {
        var input = new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        var matrix = new ReturnsMatrix<double>(input);

        input[0, 0] = 99.0;
        matrix[0, 0].Should().Be(1.0);

        var materialized = matrix.AsTimeByAsset();
        materialized[0, 1] = 77.0;
        matrix[0, 1].Should().Be(2.0);
    }

    [Fact]
    public void ReturnsMatrix_CopiesJaggedConstructorInput()
    {
        var input = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
        var matrix = new ReturnsMatrix<double>(input);

        input[0][0] = 99.0;

        matrix[0, 0].Should().Be(1.0);
    }

    [Fact]
    public void ReturnsMatrix_ZeroAssetColumns_AreRejected()
    {
        var act = () => new ReturnsMatrix<double>(new double[3, 0]);

        act.Should().Throw<ArgumentException>()
            .Where(exception => exception.ParamName == "timeByAsset");
    }

    [Fact]
    public void GarchFitResult_CopiesConstructorInputsAndPropertyOutputs()
    {
        var variances = new[] { 1.0 };
        var residuals = new[] { 1.0 };
        var result = new GarchFitResult<double>(
            0.0, 0.0, 0.1, 0.1, 0.8, variances, residuals, 0.0, true);
        var expected = result.ForecastVariance(1);

        variances[0] = 9.0;
        residuals[0] = 3.0;
        result.ConditionalVariances[0] = 12.0;
        result.StandardizedResiduals[0] = 4.0;

        result.ForecastVariance(1).Should().Be(expected);
    }

    [Fact]
    public void GraphicalLassoRecords_CopyMatrixInputsAndPropertyOutputs()
    {
        var precision = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var covariance = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var result = new GraphicalLassoResult<double>(precision, covariance, 0.1, 1, true);

        precision[0, 0] = 9.0;
        result.Precision[0, 1] = 8.0;

        result.Precision[0, 0].Should().Be(1.0);
        result.Precision[0, 1].Should().Be(0.0);

        var weights = new double[,] { { 0.0, 1.0 }, { 1.0, 0.0 } };
        var options = new GraphicalLassoOptions<double>(Weights: weights);
        weights[0, 1] = weights[1, 0] = 2.0;

        options.Weights![0, 1].Should().Be(1.0);
        options.Weights[0, 1] = 3.0;
        options.Weights[0, 1].Should().Be(1.0);
    }

    [Fact]
    public void WelfordMoments_HalfCountBeyondFiniteRepresentation_ThrowsWithoutMutation()
    {
        var moments = new WelfordMoments<Half>();
        for (var i = 0; i < 65_519; i++)
        {
            moments.Add(Half.Zero);
        }

        var act = () => moments.Add(Half.One);

        act.Should().Throw<InvalidOperationException>();
        moments.Count.Should().Be(65_519);
    }

    [Fact]
    public void WelfordPearson_HalfCountBeyondFiniteRepresentation_ThrowsWithoutMutation()
    {
        var pearson = new WelfordMoments<Half>.Pearson();
        for (var i = 0; i < 65_519; i++)
        {
            var value = (Half)(i % 2);
            pearson.Add(value, value);
        }

        var covarianceBefore = pearson.Covariance;
        var correlationBefore = pearson.Correlation;
        covarianceBefore.Should().BeGreaterThan(Half.Zero);

        var act = () => pearson.Add(Half.One, Half.One);

        act.Should().Throw<InvalidOperationException>();
        pearson.Count.Should().Be(65_519);
        pearson.Covariance.Should().Be(covarianceBefore);
        pearson.Correlation.Should().Be(correlationBefore);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void WelfordMoments_NonFiniteAdd_RejectsWithoutMutation(double invalid)
    {
        var moments = new WelfordMoments<double>();
        moments.Add(1.0);
        moments.Add(3.0);

        var act = () => moments.Add(invalid);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .Where(exception => exception.ParamName == "value");
        moments.Count.Should().Be(2);
        moments.Mean.Should().Be(2.0);
        moments.Variance.Should().Be(2.0);
    }

    [Theory]
    [InlineData(double.NaN, true)]
    [InlineData(double.PositiveInfinity, false)]
    [InlineData(double.NegativeInfinity, true)]
    public void WelfordPearson_NonFinitePair_RejectsWithoutMutation(double invalid, bool invalidX)
    {
        var pearson = new WelfordMoments<double>.Pearson();
        pearson.Add(1.0, 2.0);
        pearson.Add(3.0, 4.0);
        var correlationBefore = pearson.Correlation;

        var act = () => pearson.Add(invalidX ? invalid : 5.0, invalidX ? 6.0 : invalid);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .Where(exception => exception.ParamName == (invalidX ? "x" : "y"));
        pearson.Count.Should().Be(2);
        pearson.Covariance.Should().Be(2.0);
        pearson.Correlation.Should().Be(correlationBefore);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void SampleSkewness_NonFiniteObservation_IsRejected(double invalid)
    {
        var values = new[] { 1.0, 2.0, invalid };

        var act = () => SampleSkewness<double>.Compute(values);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void NercomeSmallSample_UsesKnownZeroMeanSecondMoments()
    {
        decimal[,] returns = { { 2, 1 }, { 2, 1 }, { 2, 1 } };
        var actual = new NercomeCovarianceEstimator().Estimate(returns);
        actual[0, 0].Should().BeApproximately(4m, 1e-20m);
        actual[0, 1].Should().BeApproximately(2m, 1e-20m);
        actual[1, 1].Should().BeApproximately(1m, 1e-20m);
    }

    [Fact]
    public void EwmaCovariance_IsPositiveSemidefinite()
    {
        var returns = new decimal[,]
        {
            { 0.01m, 0.02m, -0.01m },
            { 0.03m, 0.01m, 0.00m },
            { -0.02m, 0.04m, 0.02m },
            { 0.00m, -0.01m, 0.03m },
        };

        var covariance = new ExponentiallyWeightedCovarianceEstimator(0.94m).Estimate(returns);

        NearestPsdProjection<decimal>.IsPsd(covariance).Should().BeTrue();
    }

    [Fact]
    public void TracyWidomDenoisedCovariance_IsPositiveSemidefinite()
    {
        var returns = new decimal[,]
        {
            { 0.01m, 0.02m, -0.01m },
            { 0.03m, 0.01m, 0.00m },
            { -0.02m, 0.04m, 0.02m },
            { 0.00m, -0.01m, 0.03m },
            { 0.02m, 0.00m, 0.01m },
            { -0.01m, 0.03m, 0.02m },
        };

        var covariance = new TracyWidomDenoisedCovarianceEstimator().Estimate(returns);

        NearestPsdProjection<decimal>.IsPsd(covariance).Should().BeTrue();
    }

    private static double[,] Scale(double[,] matrix, double factor)
    {
        var scaled = new double[matrix.GetLength(0), matrix.GetLength(1)];
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                scaled[i, j] = matrix[i, j] * factor;
            }
        }

        return scaled;
    }

    private static double MaxPrecisionError(double[,] baseline, double[,] scaled, double scale)
    {
        var error = 0.0;
        for (var i = 0; i < baseline.GetLength(0); i++)
        {
            for (var j = 0; j < baseline.GetLength(1); j++)
            {
                error = Math.Max(error, Math.Abs(baseline[i, j] - (scaled[i, j] * scale)));
            }
        }

        return error;
    }
}
