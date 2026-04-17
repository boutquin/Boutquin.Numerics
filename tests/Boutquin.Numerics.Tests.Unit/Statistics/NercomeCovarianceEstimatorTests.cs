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

public sealed class NercomeCovarianceEstimatorTests
{
    [Fact]
    public void Ctor_RejectsOutOfRangeSplitFraction()
    {
        FluentActions.Invoking(() => new NercomeCovarianceEstimator(0m))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new NercomeCovarianceEstimator(1m))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new NercomeCovarianceEstimator(-0.1m))
            .Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Estimate_ProducesSymmetricMatrix()
    {
        var returns = GenerateReturns(120, 4, seed: 17);
        var cov = new NercomeCovarianceEstimator(0.5m).Estimate(returns);

        var n = cov.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                cov[i, j].Should().Be(cov[j, i]);
            }
        }
    }

    [Fact]
    public void Estimate_ProducesNonNegativeDiagonal()
    {
        var returns = GenerateReturns(150, 5, seed: 31);
        var cov = new NercomeCovarianceEstimator(0.5m).Estimate(returns);

        for (var i = 0; i < cov.GetLength(0); i++)
        {
            cov[i, i].Should().BeGreaterThanOrEqualTo(0m);
        }
    }

    [Fact]
    public void Estimate_SmallSampleFallsBackToSampleCovariance()
    {
        // T = 3 is below the minimum viable split; estimator should degrade
        // gracefully to the sample covariance rather than throw.
        var returns = GenerateReturns(3, 2, seed: 7);
        var cov = new NercomeCovarianceEstimator().Estimate(returns);
        cov.Should().NotBeNull();
        cov.GetLength(0).Should().Be(2);
        cov.GetLength(1).Should().Be(2);
    }

    [Fact]
    public void Estimate_DifferentSplitFractions_ProduceDifferentResults()
    {
        var returns = GenerateReturns(200, 4, seed: 91);
        var cov04 = new NercomeCovarianceEstimator(0.4m).Estimate(returns);
        var cov06 = new NercomeCovarianceEstimator(0.6m).Estimate(returns);

        // The two splits should not produce exactly the same matrix.
        var differ = false;
        for (var i = 0; i < 4 && !differ; i++)
        {
            for (var j = 0; j < 4 && !differ; j++)
            {
                if (cov04[i, j] != cov06[i, j])
                {
                    differ = true;
                }
            }
        }

        differ.Should().BeTrue("different split fractions should yield different estimates");
    }

    [Fact]
    public void SplitFraction_PropertyMatchesConstructor()
    {
        new NercomeCovarianceEstimator(0.42m).SplitFraction.Should().Be(0.42m);
    }

    private static decimal[,] GenerateReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var r = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                r[i, j] = (decimal)(rng.NextDouble() * 0.02 - 0.01);
            }
        }

        return r;
    }
}
