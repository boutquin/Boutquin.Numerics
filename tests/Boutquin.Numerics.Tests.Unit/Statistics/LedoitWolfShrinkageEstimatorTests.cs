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

public sealed class LedoitWolfShrinkageEstimatorTests
{
    [Fact]
    public void Estimate_PreservesSymmetryAndDiagonal()
    {
        var returns = GenerateReturns(100, 5, seed: 31);
        var cov = new LedoitWolfShrinkageEstimator().Estimate(returns);
        var n = cov.GetLength(0);

        for (var i = 0; i < n; i++)
        {
            cov[i, i].Should().BeGreaterThan(0m, $"diagonal[{i}] must be positive for a valid covariance");
            for (var j = i + 1; j < n; j++)
            {
                cov[i, j].Should().Be(cov[j, i]);
            }
        }
    }

    [Fact]
    public void Estimate_ShrinksTowardScaledIdentityForNoisySmallSample()
    {
        // Small sample, uncorrelated noise — expect the shrinkage to move
        // off-diagonal entries closer to zero than the sample covariance.
        var returns = GenerateReturns(25, 5, seed: 99);

        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var shrunk = new LedoitWolfShrinkageEstimator().Estimate(returns);

        var sampleOffSum = 0m;
        var shrunkOffSum = 0m;
        for (var i = 0; i < 5; i++)
        {
            for (var j = i + 1; j < 5; j++)
            {
                sampleOffSum += Math.Abs(sample[i, j]);
                shrunkOffSum += Math.Abs(shrunk[i, j]);
            }
        }

        shrunkOffSum.Should().BeLessThan(sampleOffSum);
    }

    [Fact]
    public void Estimate_PreservesTotalVariance()
    {
        // Shrinking toward F = μ·I preserves the trace — both Σ and F have
        // the same trace by construction (μ is the mean of Σ's diagonal).
        var returns = GenerateReturns(80, 4, seed: 2);

        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var shrunk = new LedoitWolfShrinkageEstimator().Estimate(returns);

        var traceSample = sample[0, 0] + sample[1, 1] + sample[2, 2] + sample[3, 3];
        var traceShrunk = shrunk[0, 0] + shrunk[1, 1] + shrunk[2, 2] + shrunk[3, 3];

        Math.Abs(traceSample - traceShrunk).Should().BeLessThan(1e-20m);
    }

    private static decimal[,] GenerateReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                returns[i, j] = (decimal)(rng.NextDouble() * 0.02 - 0.01);
            }
        }

        return returns;
    }
}
