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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Regression tests from the 2026-06-20 code review (findings #11, #20, #22).
/// </summary>
public sealed class CovarianceAndDsrFixTests
{
    // --- DSR SE(SR) uses the authoritative observation-count denominator T. ---

    [Fact]
    public void DeflatedSharpe_VarianceDenominatorUsesObservationCount()
    {
        // numTrials = 1 => expectedMaxSharpe = 0, so PValue = 1 - Cdf(sr / se).
        const double sr = 0.5;
        const int tradingDays = 10;   // small N so n vs n-1 is detectable

        // Independent golden (house independent-goldens discipline): PValue is a
        // fixed literal computed by hand outside the production code path — it no
        // longer re-derives the variance formula in-line (that was tautological:
        // any regression to another divisor would move both the test's local
        // recomputation and the production result together, in lockstep, so the
        // test could never fail). The normal CDF here uses math.erf (Python's
        // standard-library implementation), not this project's
        // NormalDistribution<double>.Cdf, so neither half of the computation
        // shares code with the production path.
        // The annualized SR 0.5 is converted to the per-period frequency of the
        // T = 10 observations (1 year at 10 periods/year), and eq. 2 of Bailey &
        // López de Prado (2014) scales by √(T − 1). Cross-checked independently:
        //   >>> import math
        //   >>> sr = 0.5 / math.sqrt(10)                                     # per-period SR
        //   >>> z = sr * math.sqrt(10 - 1) / math.sqrt(1 + 0.5 * sr * sr)    # skew 0, kurt 3
        //   >>> phi = lambda x: 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))
        //   >>> 1.0 - phi(z)
        //   0.31867594411696865
        const double expectedPValue = 0.31867594411696865;

        var result = DeflatedSharpeRatio<double>.Compute(sr, 1, 1.0, 0.0, 3.0, tradingDays);

        result.DeflatedSharpe.Should().BeApproximately(sr, 1e-12);
        result.PValue.Should().BeApproximately(expectedPValue, 1e-9);
    }

    // --- ValidateReturns rejects non-finite input (review finding #20) ---

    [Fact]
    public void SampleCovariance_NaNInput_ThrowsArgumentException()
    {
        var returns = new double[,]
        {
            { 0.01, 0.02 },
            { double.NaN, 0.03 },
            { 0.02, 0.01 },
        };
        var act = () => new SampleCovarianceEstimator<double>().Estimate(returns);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void SampleCovariance_InfinityInput_ThrowsArgumentException()
    {
        var returns = new double[,]
        {
            { 0.01, 0.02 },
            { 0.02, double.PositiveInfinity },
            { 0.02, 0.01 },
        };
        var act = () => new SampleCovarianceEstimator<double>().Estimate(returns);
        act.Should().Throw<ArgumentException>();
    }

    // --- Generic ICovarianceEstimator<T> accepts ReturnsMatrix<T> (review finding #22) ---

    [Fact]
    public void GenericEstimator_ReturnsMatrixOverload_MatchesArrayOverload()
    {
        var matrix = new double[,]
        {
            { 0.01, 0.02, -0.01 },
            { 0.03, -0.01, 0.02 },
            { -0.02, 0.01, 0.00 },
            { 0.01, 0.00, 0.01 },
        };

        ICovarianceEstimator<double> estimator = new SampleCovarianceEstimator<double>();
        var viaArray = estimator.Estimate(matrix);
        var viaReturnsMatrix = estimator.Estimate(new ReturnsMatrix<double>(matrix));

        viaReturnsMatrix.GetLength(0).Should().Be(viaArray.GetLength(0));
        viaReturnsMatrix.GetLength(1).Should().Be(viaArray.GetLength(1));
        for (var i = 0; i < viaArray.GetLength(0); i++)
        {
            for (var j = 0; j < viaArray.GetLength(1); j++)
            {
                viaReturnsMatrix[i, j].Should().Be(viaArray[i, j]);
            }
        }
    }
}
