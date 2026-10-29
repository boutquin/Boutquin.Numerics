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

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4ContractRegressionTests
{
    [Theory]
    [InlineData(-1, 1, .1, .1, .8)]
    [InlineData(double.NaN, 1, .1, .1, .8)]
    [InlineData(1, double.PositiveInfinity, .1, .1, .8)]
    [InlineData(1, 1, double.NaN, .1, .8)]
    [InlineData(1, 1, .1, -.1, .8)]
    [InlineData(1, 1, .1, .1, double.PositiveInfinity)]
    public void Garch_ForecastRejectsInvalidNumericState(double variance, double residual, double omega, double alpha, double beta)
    {
        var fit = new GarchFitResult<double>(0, 0, omega, alpha, beta, [variance], [residual], 0, true);
        Action act = () => fit.ForecastVariance(1);
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData(2)]
    [InlineData(49)]
    [InlineData(50)]
    public void Gt_DefaultMinimumCount_UsesInclusiveAuthorPenalty(int n)
    {
        var returns = Enumerable.Range(1, n).Select(i => i * .001).ToArray();
        GtInspiredScore<double>.Compute(returns).Score.Should().BeApproximately(999 - n * 899.0 / 50, 1e-10);
    }

    [Fact]
    public void Gt_CustomMinimumCount_IsInclusiveAndValidated()
    {
        double[] returns = [1, 2, 3, 4];
        GtInspiredScore<double>.Compute(returns, 0, 1e-6, 4).Score.Should().Be(100);
        GtInspiredScore<double>.Compute(returns, 0, 1e-6, 8).Score.Should().Be(549.5);
        Action invalid = () => GtInspiredScore<double>.Compute(returns, 0, 1e-6, 0);
        invalid.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GraphicalLasso_ReusesColumnScratchAcrossSweeps()
    {
        double[,] sample = { { 1, .4, .15, .05 }, { .4, 1.2, .3, .1 }, { .15, .3, .9, .25 }, { .05, .1, .25, 1.1 } };
        _ = GraphicalLassoSolver<double>.Solve(sample, .1);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var result = GraphicalLassoSolver<double>.Solve(sample, .1);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        result.Iterations.Should().BeGreaterThan(1);
        allocated.Should().BeLessThan(5000);
    }

    [Fact]
    public void GraphicalLasso_WeightedTwoVariable_MatchesAnalyticKktOptimum()
    {
        double[,] sample = { { 1, .6 }, { .6, 2 } };
        double[,] weights = { { 0, 2 }, { 2, 0 } };
        var result = GraphicalLassoSolver<double>.Solve(sample, .1, new GraphicalLassoOptions<double>(Weights: weights));
        result.Covariance[0, 1].Should().BeApproximately(.4, 1e-10);
        result.Precision[0, 0].Should().BeApproximately(2 / 1.84, 1e-10);
        result.Precision[0, 1].Should().BeApproximately(-.4 / 1.84, 1e-10);
    }

    [Fact]
    public void MinTrl_NonGaussianMonthlyPrimaryExample()
    {
        var result = MinimumTrackRecordLength<double>.Compute(2, 1, -.72, 5.78, .05, 12);
        result.Should().BeApproximately(59.89509868650751, 1e-10);
        (result / 12).Should().BeApproximately(4.991258223875626, 1e-11);
    }

    [Fact]
    public void Garch_HullWorkedForecasts()
    {
        var fit = new GarchFitResult<double>(0, 0, .000002, .13, .86, [.000256], [-.01 / .016], 0, true);
        fit.ForecastVariance(1).Should().BeApproximately(.00023516, 1e-15);
        // Choose the last standardized shock so h_(T+1) equals the displayed current variance.
        const double longRun = .0001391;
        const double persistence = .9714;
        var omega = longRun * (1 - persistence);
        var z = Math.Sqrt((.0003 - omega) / (.0003 * persistence));
        var forecast = new GarchFitResult<double>(0, 0, omega, persistence, 0, [.0003], [z], 0, true);
        forecast.ForecastVariance(11).Should().BeApproximately(.0002594752038161279, 1e-15);
        forecast.ForecastVariance(101).Should().BeApproximately(.00014793824340399752, 1e-15);
        forecast.ForecastVariance(10_001).Should().BeApproximately(longRun, 1e-15);
    }

    [Fact]
    public void Garch_ReturnedLikelihood_IsGaussianAverageOverReturnedState()
    {
        var data = Enumerable.Range(1, 100).Select(i => Math.Sin(i * .37) * .01 + Math.Cos(i * .73) * .02).ToArray();
        var fit = GarchOneOneEstimator<double>.Fit(data);
        var expected = -.5 * fit.ConditionalVariances.Zip(fit.StandardizedResiduals, (h, z) => Math.Log(h) + z * z).Average();
        fit.LogLikelihood.Should().BeApproximately(expected, 1e-9);
    }
}
