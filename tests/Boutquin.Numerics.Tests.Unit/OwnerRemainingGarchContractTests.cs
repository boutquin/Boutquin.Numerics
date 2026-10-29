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

namespace Boutquin.Numerics.Tests.Unit;

public sealed class OwnerRemainingGarchContractTests
{
    [Theory]
    [InlineData(0.0, 0.4, 1.0)]
    [InlineData(0.3, 0.0, 0.275)]
    public void ForecastAcceptsZeroCoefficientBoundary(double alpha, double beta, double expected)
    {
        var fit = new GarchFitResult<double>(0, 0, 0.2, alpha, beta, [2.0], [0.5 / Math.Sqrt(2.0)], 0, true);
        double actual = 0;
        Assert.Null(Record.Exception(() => actual = fit.ForecastVariance(1)));
        Assert.Equal(expected, actual, 12);
    }

    [Fact]
    public void FitCoefficientsAreNonnegative()
    {
        GarchFitResult<double>? fit = null;
        Assert.Null(Record.Exception(() => fit = GarchOneOneEstimator<double>.Fit([1, -2, 3, -4],
            new(MinObservations: 2, MaxIterations: 2) { FixedZeroMean = true, VarianceTargeting = true })));
        Assert.NotNull(fit);
        Assert.InRange(fit.Alpha, 0, 1);
        Assert.InRange(fit.Beta, 0, 1);
    }

    [Fact]
    public void TwoPhaseBudgetMatchesFiveLiteralExpansions()
    {
        // Independent five-step literal trace, recorded in remaining-literal-oracle.json.
        // Two coarse updates followed by three fine updates; every accepted step is expansion.
        var fit = GarchOneOneEstimator<double>.Fit([1, -2, 3, -4],
            new(MinObservations: 2, MaxIterations: 2, Tolerance: 1e-12) { FixedZeroMean = true, VarianceTargeting = true });
        Assert.Equal(0.009907845613728074, fit.Alpha, 12);
        Assert.Equal(0.9569271561665924, fit.Beta, 12);
        Assert.Equal(0.24873748664759665, fit.Omega, 12);
        Assert.Equal(-1.5087685556782162, fit.LogLikelihood, 12);
        Assert.False(fit.Converged);
    }

    [Fact]
    public void HullPublishedCoefficientsHavePrintedLongRunVarianceAndVolatility()
    {
        // Hull Table 23.1 rounded coefficients; convergence after 10,000 periods is below rounding error.
        var fit = new GarchFitResult<double>(0, 0, 0.0000039818, 0.223793, 0.747577, [0.0003], [0.0], 0, true);
        var variance = fit.ForecastVariance(10000);
        Assert.InRange(variance, 0.00013905, 0.00013915);
        Assert.InRange(Math.Sqrt(variance), 0.011785, 0.011795);
    }
}
