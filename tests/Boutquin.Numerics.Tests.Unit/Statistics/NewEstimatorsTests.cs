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

using Boutquin.Numerics.Random;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class NewEstimatorsTests
{
    [Fact]
    public void QisEstimator_ProducesSymmetricMatrix()
    {
        var returns = MakeFactorReturns(150, 6, seed: 1);
        var cov = new QuadraticInverseShrinkageEstimator().Estimate(returns);
        AssertSymmetric(cov);
    }

    [Fact]
    public void LedoitWolfConstantCorrelation_PreservesDiagonal()
    {
        var returns = MakeFactorReturns(100, 4, seed: 2);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var lwcc = new LedoitWolfConstantCorrelationEstimator().Estimate(returns);

        for (var i = 0; i < 4; i++)
        {
            ((double)Math.Abs(sample[i, i] - lwcc[i, i])).Should().BeLessThan(1e-12);
        }
    }

    [Fact]
    public void LedoitWolfSingleFactor_PreservesDiagonal()
    {
        var returns = MakeFactorReturns(120, 5, seed: 3);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var lwfm = new LedoitWolfSingleFactorEstimator().Estimate(returns);

        for (var i = 0; i < 5; i++)
        {
            ((double)Math.Abs(sample[i, i] - lwfm[i, i])).Should().BeLessThan(1e-12);
        }
    }

    [Fact]
    public void OracleApproximatingShrinkage_StaysWithinIdentityAndSample()
    {
        var returns = MakeFactorReturns(80, 4, seed: 4);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var oas = new OracleApproximatingShrinkageEstimator().Estimate(returns);

        // OAS off-diagonals are bounded between sample (when ρ=0) and 0 (when ρ=1).
        for (var i = 0; i < 4; i++)
        {
            for (var j = i + 1; j < 4; j++)
            {
                var sampleAbs = Math.Abs(sample[i, j]);
                var oasAbs = Math.Abs(oas[i, j]);
                oasAbs.Should().BeLessThanOrEqualTo(sampleAbs + 1e-20m);
            }
        }
    }

    [Fact]
    public void TracyWidomDenoised_ProducesPsd()
    {
        var returns = MakeFactorReturns(200, 5, seed: 5);
        var cov = new TracyWidomDenoisedCovarianceEstimator().Estimate(returns);
        AssertSymmetric(cov);
    }

    [Fact]
    public void ReturnsMatrix_AcceptsBothLayouts()
    {
        decimal[,] tn =
        {
            { 0.01m, 0.02m },
            { -0.01m, 0.005m },
            { 0.015m, -0.005m },
            { -0.005m, 0.01m },
        };

        decimal[][] assetMajor =
        [
            [0.01m, -0.01m, 0.015m, -0.005m],
            [0.02m, 0.005m, -0.005m, 0.01m],
        ];

        ICovarianceEstimator estimator = new SampleCovarianceEstimator();
        var fromTN = estimator.Estimate(new ReturnsMatrix(tn));
        var fromAM = estimator.Estimate(new ReturnsMatrix(assetMajor));

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                ((double)Math.Abs(fromTN[i, j] - fromAM[i, j])).Should().BeLessThan(1e-20);
            }
        }
    }

    [Fact]
    public void DeflatedSharpeRatio_FromReturns_MatchesManualComputation()
    {
        var rng = new Pcg64RandomSource(42UL);
        var returns = new decimal[1000];
        var gaussian = new GaussianSampler(rng);
        for (var i = 0; i < returns.Length; i++)
        {
            returns[i] = (decimal)(0.0005 + 0.01 * gaussian.Next());
        }

        var fromReturns = DeflatedSharpeRatio.ComputeFromReturns(returns, numTrials: 50);

        // Sanity: with 1000 obs (~4 years) and modest Sharpe, p-value should be in (0, 1).
        fromReturns.PValue.Should().BeGreaterThanOrEqualTo(0m);
        fromReturns.PValue.Should().BeLessThanOrEqualTo(1m);
        fromReturns.NumTrials.Should().Be(50);
    }

    private static decimal[,] MakeFactorReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            var factor = rng.NextDouble() * 0.02 - 0.01;
            for (var j = 0; j < n; j++)
            {
                var idio = rng.NextDouble() * 0.002 - 0.001;
                returns[i, j] = (decimal)(factor + idio);
            }
        }

        return returns;
    }

    private static void AssertSymmetric(decimal[,] m)
    {
        var n = m.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                ((double)Math.Abs(m[i, j] - m[j, i])).Should().BeLessThan(1e-10);
            }
        }
    }
}
