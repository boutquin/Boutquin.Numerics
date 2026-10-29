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

public sealed class CovarianceReductionRegressionTests
{
    [Fact]
    public void SampleCovariance_TwoTranslatedObservations_PreservesVariance()
    {
        const decimal anchor = 10000000000000000000000000000m;
        new SampleCovarianceEstimator<decimal>().Estimate(new[,] { { anchor }, { anchor + 1m } })[0, 0]
            .Should().Be(0.5m);
    }

    [Fact]
    public void SampleCovariance_HalfTwoTranslatedObservations_PreservesVariance()
    {
        ((double)new SampleCovarianceEstimator<Half>().Estimate(new[,] { { (Half)60000 }, { (Half)60032 } })[0, 0])
            .Should().Be(512);
    }

    [Fact]
    public void SampleCovariance_ThreeTranslatedObservations_PreservesVariance()
    {
        const decimal anchor = 100000000000000000000m;
        const decimal step = 0.00000001m;
        new SampleCovarianceEstimator<decimal>().Estimate(new[,] { { anchor }, { anchor }, { anchor + step } })[0, 0]
            .Should().BeApproximately(step * step / 3m, 0.0000000000000000000000000001m);
    }

    [Fact]
    public void SampleCovariance_HalfThreeTranslatedObservations_PreservesVariance()
    {
        ((double)new SampleCovarianceEstimator<Half>().Estimate(new[,] { { (Half)1024 }, { (Half)1024 }, { (Half)1025 } })[0, 0])
            .Should().BeApproximately(1.0 / 3, 0.001);
    }

    [Fact]
    public void Ewma_LongHalfPanel_RespectsBoundedVariance()
    {
        var returns = new Half[70000, 1];
        returns[0, 0] = (Half)100;
        var variance = (double)new ExponentiallyWeightedCovarianceEstimator<Half>((Half)0.9995).Estimate(returns)[0, 0];
        variance.Should().BeInRange(0, 2500);
    }

    [Theory]
    [InlineData(70000)]
    [InlineData(100000)]
    public void Poet_LongHalfPanel_AppliesPositiveThreshold(int count)
    {
        var returns = new Half[count, 3];
        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                returns[i, j] = (Half)((((i * (j + 1)) + j) % 11 - 5) * 0.125);
            }
        }

        var thresholded = new PoetCovarianceEstimator<Half>().Estimate(returns);
        var unthresholded = new PoetCovarianceEstimator<Half>(1, (Half)0).Estimate(returns);
        thresholded[0, 1].Should().NotBe(unthresholded[0, 1]);
    }
}
