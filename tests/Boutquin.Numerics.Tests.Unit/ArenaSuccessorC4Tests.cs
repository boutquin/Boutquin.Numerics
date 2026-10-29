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

[Trait("Tier", "Unit")]
public sealed class ArenaSuccessorC4Tests
{
    [Fact]
    public void GarchAr1Mean_FitsCoefficient_AndReconstructsConditionalResiduals()
    {
        var returns = new double[600];
        uint state = 93479;
        for (var i = 1; i < returns.Length; i++)
        {
            state = unchecked(state * 1664525 + 1013904223);
            returns[i] = .003 + .72 * returns[i - 1] + .02 * ((state >> 8) / 16777216.0 - .5);
        }

        var fit = GarchOneOneEstimator<double>.Fit(returns, new(IncludeAr1Mean: true, MaxIterations: 10000, Tolerance: 1e-9));
        fit.Phi.Should().BeApproximately(.72, .10);
        var residuals = fit.StandardizedResiduals;
        var variances = fit.ConditionalVariances;
        for (var i = 1; i < returns.Length; i++)
        {
            (residuals[i] * Math.Sqrt(variances[i])).Should().BeApproximately(
                returns[i] - fit.Mu - fit.Phi * returns[i - 1], 1e-12);
        }
    }

    [Fact]
    public void DenoisingShrinkage_IsotropicLargeVariance_RemainsFinite()
    {
        var returns = new double[6, 3];
        for (var j = 0; j < 3; j++)
        {
            returns[2 * j, j] = 1.5e154;
            returns[2 * j + 1, j] = -1.5e154;
        }

        var covariance = new DenoisedCovarianceEstimator<double>(true).Estimate(returns);
        for (var j = 0; j < 3; j++)
        {
            (covariance[j, j] / 9e307).Should().BeApproximately(1, 1e-12);
        }
    }

    [Fact]
    public void GarchForecast_FiniteHorizon_DoesNotRequireFiniteLongRun()
    {
        var fit = new GarchFitResult<double>(0, 0, 1e308, .1, .4, [1e308], [1], 0, true);
        (fit.ForecastVariance(2) / 1e308).Should().BeApproximately(1.75, 1e-14);
    }

    [Theory]
    [InlineData(60, 1e308, 9e307, -1e308, 1e308, 299.56468416687574, -5.417201350452168)]
    [InlineData(52, 1e308, -1e307, -1.5e308, 1e-6, 25.319609883015552, -20.531548767677993)]
    public void GtSignificance_OppositeBenchmark_PreservesFiniteQuotient(
        int count, double first, double second, double benchmark, double epsilon, double z, double score)
    {
        var returns = Enumerable.Range(0, count).Select(i => i % 2 == 0 ? first : second).ToArray();
        var result = GtInspiredScore<double>.Compute(returns, benchmark, epsilon);
        result.Z.Should().BeApproximately(z, 1e-10);
        result.Score.Should().BeApproximately(score, 1e-10);
    }

    [Theory]
    [InlineData(200, false)]
    [InlineData(200, true)]
    public void NercomeHalf_RotatedVarianceMayExceedRange_WhenEntriesFit(int amplitude, bool twoRows)
    {
        // The raw moment is amplitude² J: eigenvalue 2*amplitude² exceeds Half,
        // while every returned entry amplitude² is representable.
        var returns = new Half[twoRows ? 2 : 8, 2];
        for (var i = 0; i < returns.GetLength(0); i++)
        {
            returns[i, 0] = returns[i, 1] = (Half)(i % 2 == 0 ? amplitude : -amplitude);
        }

        var expected = (double)(Half)(amplitude * amplitude);
        var result = new NercomeCovarianceEstimator<Half>().Estimate(returns);
        foreach (var value in result)
        {
            ((double)value).Should().Be(expected);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void DsrHalf_HighFrequency_PreservesFiniteAnnualSharpe(int scenario)
    {
        Half[] returns = scenario switch
        {
            0 => [(Half).01, (Half).02, (Half).03, (Half).04],
            1 => [(Half)(-1), (Half)0, (Half)1, (Half)2],
            _ => [(Half).01, (Half)(-.02), (Half).03, (Half).04],
        };
        var trials = scenario == 2 ? 10 : 1;
        var wide = returns.Select(x => (double)x).ToArray();
        var expected = DeflatedSharpeRatio<double>.ComputeFromReturns(wide, trials, 70000);
        DsrResult<Half>? result = null;
        var compute = () => result = DeflatedSharpeRatio<Half>.ComputeFromReturns(returns, trials, 70000);
        compute.Should().NotThrow();
        ((double)result!.DeflatedSharpe).Should().BeApproximately(expected.DeflatedSharpe, 2);
        ((double)result.PValue).Should().BeApproximately(expected.PValue, .002);
    }

    [Fact]
    public void GarchHalf_LongNonconstantSeries_HasRepresentableInitialVariance()
    {
        var returns = Enumerable.Range(0, 70000).Select(i => (Half)(i % 2 == 0 ? 1 : -1)).ToArray();
        GarchFitResult<Half>? result = null;
        var fit = () => result = GarchOneOneEstimator<Half>.Fit(returns, new(MaxIterations: 2));
        fit.Should().NotThrow();
        var firstVariance = (double)result!.ConditionalVariances[0];
        var seed = returns.Select(x => Math.Pow((double)x - (double)result.Mu, 2)).Average();
        var expected = (double)result.Omega + ((double)result.Alpha + (double)result.Beta) * seed;
        firstVariance.Should().BeApproximately(expected, .005);
        Half.IsFinite(result.LogLikelihood).Should().BeTrue();
    }

    [Fact]
    public void GraphicalLasso_ExhaustedInnerCycle_CannotClaimConvergence()
    {
        double[,] covariance = { { 1, .999, .999 }, { .999, 1, .999 }, { .999, .999, 1 } };
        var result = GraphicalLassoSolver<double>.Solve(covariance, 0, new(Tolerance: 1e-8));
        var residual = 0.0;
        var precision = result.Precision;
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var product = 0.0;
                for (var k = 0; k < 3; k++)
                {
                    product += covariance[i, k] * precision[k, j];
                }
                residual = Math.Max(residual, Math.Abs(product - (i == j ? 1 : 0)));
            }
        }

        (result.Converged && residual > 1e-5).Should().BeFalse("convergence must not hide exhausted coordinate descent");
    }

    [Fact]
    public void PoetFacade_RoundedDecimalBoundary_RejectsWithParameterException()
    {
        var boundary = (double)decimal.MaxValue;
        var accept = () => new PoetCovarianceEstimator(thresholdMultiplier: Math.BitDecrement(boundary));
        accept.Should().NotThrow();
        var reject = () => new PoetCovarianceEstimator(thresholdMultiplier: boundary);
        reject.Should().ThrowExactly<ArgumentOutOfRangeException>().WithParameterName("thresholdMultiplier");
    }
}
