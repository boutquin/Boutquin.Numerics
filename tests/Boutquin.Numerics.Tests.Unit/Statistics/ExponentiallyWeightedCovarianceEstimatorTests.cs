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

public sealed class ExponentiallyWeightedCovarianceEstimatorTests
{
    [Fact]
    public void Constructor_RejectsInvalidLambda()
    {
        Action zero = () => new ExponentiallyWeightedCovarianceEstimator(0m);
        Action one = () => new ExponentiallyWeightedCovarianceEstimator(1m);
        Action negative = () => new ExponentiallyWeightedCovarianceEstimator(-0.1m);

        zero.Should().Throw<ArgumentOutOfRangeException>();
        one.Should().Throw<ArgumentOutOfRangeException>();
        negative.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Estimate_PreservesSymmetryAndPsd()
    {
        var returns = GenerateReturns(200, 3, seed: 11);
        var cov = new ExponentiallyWeightedCovarianceEstimator(0.94m).Estimate(returns);

        // Symmetry.
        for (var i = 0; i < 3; i++)
        {
            for (var j = i + 1; j < 3; j++)
            {
                cov[i, j].Should().Be(cov[j, i]);
            }
        }

        // Non-negative diagonal (weighted variances).
        for (var i = 0; i < 3; i++)
        {
            cov[i, i].Should().BeGreaterThanOrEqualTo(0m);
        }
    }

    [Fact]
    public void Estimate_RecentObservationsWeighMore()
    {
        // Three assets where the most recent 10 observations are uniformly
        // large (magnitude ~0.05) and earlier observations are small
        // (magnitude ~0.001). An EWMA with λ = 0.5 should see substantially
        // larger variance than a simple sample covariance.
        const int t = 30;
        var returns = new decimal[t, 1];
        var rng = new System.Random(7);
        for (var i = 0; i < t - 10; i++)
        {
            returns[i, 0] = (decimal)(rng.NextDouble() * 0.002 - 0.001);
        }

        for (var i = t - 10; i < t; i++)
        {
            returns[i, 0] = (decimal)(rng.NextDouble() * 0.1 - 0.05);
        }

        var ewma = new ExponentiallyWeightedCovarianceEstimator(0.5m).Estimate(returns);
        var sample = new SampleCovarianceEstimator().Estimate(returns);

        // EWMA should favor the recent regime, producing higher variance.
        ewma[0, 0].Should().BeGreaterThan(sample[0, 0]);
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
