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

public sealed class DenoisedCovarianceEstimatorTests
{
    [Fact]
    public void Estimate_PreservesSymmetryForFactorStructuredReturns()
    {
        var returns = GenerateFactorStructuredReturns(200, 5, seed: 41);
        var cov = new DenoisedCovarianceEstimator().Estimate(returns);

        for (var i = 0; i < 5; i++)
        {
            for (var j = i + 1; j < 5; j++)
            {
                ((double)Math.Abs(cov[i, j] - cov[j, i])).Should().BeLessThan(1e-10);
            }
        }
    }

    [Fact]
    public void Estimate_DelegatesToSampleForFewerThanThreeAssets()
    {
        var returns = GenerateFactorStructuredReturns(50, 2, seed: 7);

        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var denoised = new DenoisedCovarianceEstimator().Estimate(returns);

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                denoised[i, j].Should().Be(sample[i, j]);
            }
        }
    }

    [Fact]
    public void Estimate_PreservesDiagonalVariances()
    {
        // Reconstruction forces unit correlation diagonal, so the resulting
        // covariance diagonal matches the sample variances exactly.
        var returns = GenerateFactorStructuredReturns(200, 5, seed: 13);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var denoised = new DenoisedCovarianceEstimator().Estimate(returns);

        for (var i = 0; i < 5; i++)
        {
            ((double)Math.Abs(sample[i, i] - denoised[i, i])).Should().BeLessThan(1e-10);
        }
    }

    [Fact]
    public void Estimate_WithShrinkage_ProducesDifferentMatrixThanWithout()
    {
        var returns = GenerateFactorStructuredReturns(100, 4, seed: 88);

        var plain = new DenoisedCovarianceEstimator(applyLedoitWolfShrinkage: false).Estimate(returns);
        var shrunk = new DenoisedCovarianceEstimator(applyLedoitWolfShrinkage: true).Estimate(returns);

        // The shrinkage pass shifts off-diagonal entries toward zero.
        Math.Abs(shrunk[0, 1]).Should().BeLessThan(Math.Abs(plain[0, 1]) + 1e-20m);
    }

    /// <summary>
    /// Factor-structured returns: one common factor plus idiosyncratic noise.
    /// Gives a clean signal/noise eigenvalue split that exercises the
    /// Marčenko-Pastur threshold.
    /// </summary>
    private static decimal[,] GenerateFactorStructuredReturns(int t, int n, int seed)
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
}
