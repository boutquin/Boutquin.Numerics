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

[Trait("Tier", "Unit")]
public sealed class C3NumericalStabilityTests
{
    [Theory]
    [InlineData(1e-20)]
    [InlineData(-1e-20)]
    public void FisherForward_TinyDouble_RetainsCorrelation(double correlation)
    {
        FisherZTransform<double>.Forward(correlation).Should().Be(correlation);
        FisherZTransform<double>.Inverse(FisherZTransform<double>.Forward(correlation)).Should().Be(correlation);
    }

    [Fact]
    public void FisherForward_TinyFloatAndHalf_RetainsCorrelation()
    {
        FisherZTransform<float>.Forward(1e-10f).Should().Be(1e-10f);
        FisherZTransform<Half>.Forward((Half)0.0001).Should().Be((Half)0.0001);
        FisherZTransform<Half>.Forward((Half)(-0.0001)).Should().Be((Half)(-0.0001));
    }

    [Theory]
    [InlineData(1e-200, 1e-200, 1)]
    [InlineData(1e-200, 1e200, -1)]
    [InlineData(1e-170, 1, 1)]
    public void Pearson_TinyDouble_RetainsDependence(double scaleX, double scaleY, int sign)
    {
        var accumulator = new WelfordMoments<double>.Pearson();
        for (var i = -1; i <= 1; i++)
        {
            accumulator.Add(i * scaleX, sign * i * scaleY);
        }

        accumulator.Correlation.Should().BeApproximately(sign, 2e-15);
    }

    [Fact]
    public void Pearson_TinyDecimal_RetainsDependence()
    {
        var accumulator = new WelfordMoments<decimal>.Pearson();
        for (var i = 1; i <= 3; i++)
        {
            accumulator.Add(i * 0.00000000000000000001m, -i * 0.00000000000000000001m);
        }

        accumulator.Correlation.Should().Be(-1m);
        accumulator.Covariance.Should().Be(0m);
    }
}
