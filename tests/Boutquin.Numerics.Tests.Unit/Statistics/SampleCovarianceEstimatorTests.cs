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

public sealed class SampleCovarianceEstimatorTests
{
    [Fact]
    public void Estimate_LargeFiniteDoubleProduct_ReturnsFiniteVariance()
    {
        var returns = new double[1000, 1];
        returns[0, 0] = 1e155;

        var covariance = new SampleCovarianceEstimator<double>().Estimate(returns);

        covariance[0, 0].Should().BeApproximately(1e307, 1e294);
    }

    [Fact]
    public void Estimate_LongHalfSeries_PreservesVarianceBeyondHalfCountRange()
    {
        var returns = new Half[70_000, 1];
        for (var i = 0; i < returns.GetLength(0); i++)
        {
            returns[i, 0] = (Half)(i % 2);
        }

        var covariance = new SampleCovarianceEstimator<Half>().Estimate(returns);

        ((float)covariance[0, 0]).Should().BeApproximately(0.25f, 0.001f);
    }

    [Fact]
    public void Estimate_LongAlternatingHalfSeries_PreservesVariance()
    {
        var returns = new Half[4096, 1];
        for (var i = 0; i < 4096; i++)
        {
            returns[i, 0] = (Half)(i % 2);
        }

        var covariance = new SampleCovarianceEstimator<Half>().Estimate(returns);

        ((float)covariance[0, 0]).Should().BeApproximately(0.25f, 0.001f);
    }

    [Fact]
    public void Estimate_HalfSeriesBeyondExactIntegerSum_PreservesVariance()
    {
        var returns = new Half[5000, 1];
        for (var i = 0; i < 5000; i++)
        {
            returns[i, 0] = (Half)(i % 2);
        }

        var covariance = new SampleCovarianceEstimator<Half>().Estimate(returns);

        ((float)covariance[0, 0]).Should().BeApproximately(0.25f, 0.001f);
    }

    [Fact]
    public void Estimate_UsesUnbiasedDivisor()
    {
        // Two assets, T = 3. Hand-computed expected covariance with N-1 divisor.
        // Asset A: [1, 2, 3]  → mean 2, deviations [-1, 0, 1], sum of squares = 2, var = 2 / (3-1) = 1
        // Asset B: [2, 4, 6]  → mean 4, deviations [-2, 0, 2], sum of squares = 8, var = 8 / 2 = 4
        // Cov(A,B) = ((-1)(-2) + 0 + (1)(2)) / 2 = 4 / 2 = 2
        var returns = new decimal[,]
        {
            { 1m, 2m },
            { 2m, 4m },
            { 3m, 6m },
        };

        var cov = new SampleCovarianceEstimator().Estimate(returns);

        cov[0, 0].Should().Be(1m);
        cov[1, 1].Should().Be(4m);
        cov[0, 1].Should().Be(2m);
        cov[1, 0].Should().Be(2m);
    }

    [Fact]
    public void Estimate_RejectsInsufficientObservations()
    {
        var returns = new decimal[,] { { 1m, 2m } };
        var estimator = new SampleCovarianceEstimator();
        Action act = () => estimator.Estimate(returns);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Estimate_RejectsZeroAssets()
    {
        var returns = new decimal[2, 0];
        var estimator = new SampleCovarianceEstimator();
        Action act = () => estimator.Estimate(returns);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Estimate_PreservesSymmetry()
    {
        var returns = GenerateReturns(50, 4, seed: 17);
        var cov = new SampleCovarianceEstimator().Estimate(returns);
        var n = cov.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                cov[i, j].Should().Be(cov[j, i]);
            }
        }
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
