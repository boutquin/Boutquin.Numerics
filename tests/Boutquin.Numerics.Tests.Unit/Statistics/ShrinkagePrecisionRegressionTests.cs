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

public sealed class ShrinkagePrecisionRegressionTests
{
    [Fact]
    public void Oas_KnownZeroMean_UnrepresentableRawMomentThrows()
    {
        // Raw second moment is 10^40, beyond decimal's range. Centering would incorrectly return zero.
        var estimator = new OracleApproximatingShrinkageEstimator<decimal>();
        var action = () => estimator.Estimate(new decimal[,] { { 100000000000000000000m }, { 100000000000000000000m } });
        action.Should().Throw<OverflowException>();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Estimate_LargeDecimalTranslation_PreservesCovariance(int kind)
    {
        ICovarianceEstimator<decimal> estimator = kind switch
        {
            0 => new LedoitWolfShrinkageEstimator<decimal>(),
            1 => new LedoitWolfConstantCorrelationEstimator<decimal>(),
            _ => new LedoitWolfSingleFactorEstimator<decimal>(),
        };
        decimal[,] panel = { { -2m, -1m }, { -1m, 2m }, { 1m, -2m }, { 2m, 1m } };
        var baseline = estimator.Estimate(panel);
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                panel[i, j] += 100000000000000000000m;
            }
        }

        var shifted = estimator.Estimate(panel);
        shifted[0, 0].Should().BeApproximately(baseline[0, 0], 0.00000000000000000001m);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Estimate_LongHalfPanel_PreservesFiniteCovariance(int kind)
    {
        ICovarianceEstimator<Half> estimator = kind switch
        {
            0 => new LedoitWolfShrinkageEstimator<Half>(),
            1 => new LedoitWolfConstantCorrelationEstimator<Half>(),
            _ => new LedoitWolfSingleFactorEstimator<Half>(),
        };
        var panel = new Half[70000, 3];
        for (var i = 0; i < 70000; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[i, j] = (Half)(((i * (j + 1) + j) % 11 - 5) * 0.125);
            }
        }

        var covariance = estimator.Estimate(panel);
        // Identity: independent Python formula. Correlation and factor: investigated
        // double-specialization values on the same exactly representable panel.
        var expectedOffDiagonal = kind switch { 0 => 0.07811133507648053, 1 => 0.07811166798778911, _ => 0.0781229886877776 };
        ((double)covariance[0, 1]).Should().BeApproximately(expectedOffDiagonal, 0.0001);
        ((double)covariance[0, 0]).Should().BeApproximately(0.15625, 0.0002);
    }
}
