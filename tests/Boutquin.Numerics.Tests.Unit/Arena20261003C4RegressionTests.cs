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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class Arena20261003C4RegressionTests
{
    [Fact]
    public void Round4_GtDownside_IgnoresPositiveMagnitude()
    {
        var result = GtInspiredScore<double>.Compute([-1, 1e300], 0, 1e-6, 1);
        result.DownsideDeviation.Should().BeApproximately(Math.Sqrt(0.5), 1e-15);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Round4_Shrinkage_PreservesSmallColumnVariance(bool singleFactor)
    {
        double[,] panel = { { 1e100, 1e-100 }, { -1e100, -1e-100 }, { 1e100, -1e-100 }, { -1e100, 1e-100 } };
        ICovarianceEstimator<double> estimator = singleFactor
            ? new LedoitWolfSingleFactorEstimator<double>()
            : new LedoitWolfConstantCorrelationEstimator<double>();
        estimator.Estimate(panel)[1, 1].Should().BeApproximately(4e-200 / 3, 1e-214);
    }

    [Fact]
    public void Round4_GarchHalf_WeightedInnovationRemainsFinite()
    {
        var observations = Enumerable.Range(0, 20).Select(i => (Half)(i == 0 ? 270 : i % 2 == 0 ? 1 : -1)).ToArray();
        GarchFitResult<Half>? result = null;
        var fit = () => result = GarchOneOneEstimator<Half>.Fit(observations,
            new GarchOptions<Half>(MaxIterations: 40) { FixedZeroMean = true });
        fit.Should().NotThrow("weighted innovations and the initial likelihood are representable");
        result!.ConditionalVariances.Should().OnlyContain(h => Half.IsFinite(h) && h > (Half)0);
        Half.IsFinite(result.LogLikelihood).Should().BeTrue();
    }

    [Fact]
    public void Round4_MinTrl_MedianQuantileHasOneObservation()
    {
        MinimumTrackRecordLength<double>.Compute(1e-320, 0.5, 1).Should().Be(1);
    }

    [Fact]
    public void Round4_GlassoHalf_RejectsFalseFirstSweepConvergence()
    {
        Half[,] covariance =
        {
            { (Half)30000, (Half)20000, (Half)20000 },
            { (Half)20000, (Half)30000, (Half)20000 },
            { (Half)20000, (Half)20000, (Half)30000 },
        };
        var result = GraphicalLassoSolver<Half>.Solve(covariance, (Half)5000,
            new GraphicalLassoOptions<Half> { MaxIterations = 1 });
        result.Converged.Should().BeFalse("first-sweep mean change exceeds the finite stopping threshold");
    }

    [Theory]
    [InlineData(0.0, 1.0)]
    [InlineData(0.5, 2.5e307)]
    public void Forecast_ZeroAlpha_DoesNotEvaluateUnusedInnovation(double beta, double expected)
    {
        var fit = new GarchFitResult<double>(0, 0, 1, 0, beta, [1e308], [1e308], 0, true);
        fit.ForecastVariance(2).Should().Be(expected);
    }

    [Fact]
    public void Forecast_TinyPositiveAlpha_PreservesRepresentableWeightedInnovation()
    {
        var fit = new GarchFitResult<double>(0, 0, 1, 1e-310, 0, [1e308], [2e154], 0, true);
        fit.ForecastVariance(1).Should().BeApproximately(4e306, 4e293);
    }

    [Fact]
    public void Dsr_PositiveVarianceWithUnderflowingQuotient_RetainsSignificance()
    {
        DeflatedSharpeRatio<double>.Compute(1, 1, 1e308, 2, double.BitIncrement(5), 1)
            .PValue.Should().Be(0);
    }

    [Fact]
    public void SampleCovariance_SubnormalProducts_RetainsRepresentableVariance()
    {
        // Exact binary inputs give 2*a*a rounding to the least positive double.
        new SampleCovarianceEstimator<double>().Estimate(new double[,] { { -1.4e-162 }, { 1.4e-162 } })[0, 0]
            .Should().Be(double.Epsilon);
    }

    [Fact]
    public void VarInverse_DecimalOverflow_ReportsNoninvertibleMatrix()
    {
        var action = () => NeweyWestVariance<decimal>.LongRunCovariance(
            new decimal[,] { { 0.00000000000001m, 1m }, { 0m, 0.1m }, { 0.00000000000001m, 0m } },
            new decimal[] { 1m, 1m, 1m }, new decimal[] { 1m, 1m }, 0);
        action.Should().Throw<InvalidOperationException>().WithMessage("*rank-deficient*");
    }

    [Theory]
    [InlineData(0, 0.33623188405797101449)]
    [InlineData(1, 0.37522655519214776085)]
    [InlineData(2, 0.47327320308358828170)]
    [InlineData(3, 0.25861960053348903500)]
    public void CenteredEstimator_TranslatedFractionalMeans_PreservesIndependentMomentOracle(int kind, double expected)
    {
        double[,] rows = { { 0, 0, 2 }, { 0, 2, 0 }, { 0, 0, 0 }, { 2, 4, 0 }, { 0, 2, 2 }, { 2, 0, 4 } };
        var panel = new double[24, 3];
        for (var i = 0; i < 24; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[i, j] = 1e16 + rows[i % 6, j];
            }
        }

        ICovarianceEstimator<double> estimator = kind switch
        {
            0 => new LedoitWolfShrinkageEstimator<double>(),
            1 => new LedoitWolfConstantCorrelationEstimator<double>(),
            2 => new LedoitWolfSingleFactorEstimator<double>(),
            _ => new PoetCovarianceEstimator<double>(1, 0.5),
        };
        // Independent 80-digit centered paper-equation oracle, selected Bessel output convention.
        estimator.Estimate(panel)[0, 1].Should().BeApproximately(expected, 2e-13);
    }

    [Fact]
    public void HalfGlasso_StationaryIdentityConverges()
    {
        Half[,] covariance = { { (Half)1, (Half)0 }, { (Half)0, (Half)1 } };
        var result = GraphicalLassoSolver<Half>.Solve(covariance, (Half)0,
            new GraphicalLassoOptions<Half> { MaxIterations = 3 });
        result.Converged.Should().BeTrue("the exact identity solution is stationary");
        result.Iterations.Should().Be(1);
    }

    [Fact]
    public void GtInspired_OverflowingSignificanceHasFiniteLogarithmicLoss()
    {
        var returns = Enumerable.Range(0, 100).Select(i => (i % 2 + 1) * 1e-100).ToArray();
        var result = GtInspiredScore<double>.Compute(returns, -1e300, 1e-6, 50);
        // Independent 100-digit evaluation of the selected sample-SD/cumulative-R² variant.
        result.Score.Should().BeApproximately(-1.3859909352369063e-91, 3e-104);
    }

    [Fact]
    public void Haircut_CauchyTailRemainsRepresentableAfterStatisticSquareOverflows()
    {
        var result = HaircutSharpe<double>.Compute(1e200, 100, 2, 1);
        var expected = 100 * 2 / Math.PI * Math.Atan(1 / (1e200 * Math.Sqrt(2)));
        result.BonferroniPValue.Should().BeApproximately(expected, expected * 2e-13);
        result.HaircutSharpe.Should().BeApproximately(1e198, 2e185);
    }

    [Fact]
    public void Ewma_OppositeExtremeEndpointsHaveFiniteWeightedVariance()
    {
        var actual = new ExponentiallyWeightedCovarianceEstimator<double>(1e-309)
            .Estimate(new double[,] { { -7e307 }, { 1.3e308 } })[0, 0];
        // Exact two-point identity λ(b-a)²/(1+λ)², evaluated independently at 100 digits.
        actual.Should().BeApproximately(4.000000000000008e307, 2e294);
    }

    [Fact]
    public void Ewma_SmallMixedSignsDoNotLoseRepresentableSubnormalVariance()
    {
        var actual = new ExponentiallyWeightedCovarianceEstimator<double>(.5)
            .Estimate(new double[,] { { -2.5e-162 }, { 2.5e-162 } })[0, 0];
        actual.Should().Be(double.Epsilon);
    }

    [Theory]
    [InlineData(10000)]
    [InlineData(60000)]
    public void HalfBhy_LongHarmonicReductionDoesNotStagnate(int count)
    {
        var values = Enumerable.Repeat((Half)1, count).ToArray();
        values[0] = (Half)1e-6;
        var harmonic = Enumerable.Range(1, count).Sum(i => 1m / i);
        var expected = (Half)((decimal)(double)values[0] * count * harmonic);
        var actual = HaircutSharpe<Half>.AdjustPValues(values).Bhy;
        actual[0].Should().Be(expected);
        actual[^1].Should().Be((Half)1);
    }
}
