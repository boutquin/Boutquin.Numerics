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

public sealed class ArenaR3B12Tests
{
    [Theory]
    [InlineData(1.0, 3.0, -1.0, 1.0)]
    [InlineData(0.0, 1.0, -0.5, 0.5)]
    public void Pbo_HalfLongJoinedSamplesPreserveStableWinner(double a, double b, double c, double d)
    {
        var panel = new Half[2, 140_000];
        for (var i = 0; i < 140_000; i++)
        {
            panel[0, i] = (Half)(i % 2 == 0 ? a : b);
            panel[1, i] = (Half)(i % 2 == 0 ? c : d);
        }

        var result = ProbabilityOfBacktestOverfitting<Half>.Compute(panel, 2);
        result.Pbo.Should().Be((Half)0);
        result.LogitValues.Should().OnlyContain(x => x == (Half)Math.Log(2));
    }

    [Fact]
    public void NeweyWest_RepresentableVarianceAvoidsIntermediateOverflow()
    {
        var series = Enumerable.Range(0, 100).Select(i => i % 2 == 0 ? 100000000000000m : -100000000000000m).ToArray();
        decimal actual = 0;
        Action compute = () => actual = NeweyWestVariance<decimal>.MeanVariance(series, 0);
        compute.Should().NotThrow();
        actual.Should().Be(100000000000000000000000000m);
        NeweyWestVariance<double>.MeanVariance(new[] { 1e308, 1e308 }, 0).Should().Be(0);
    }

    [Fact]
    public void NeweyWest_FloatTranslationPreservesSmallVariance()
    {
        var series = Enumerable.Range(0, 10_000).Select(i => 1000000f + i % 2).ToArray();
        NeweyWestVariance<float>.MeanVariance(series, 0).Should().BeApproximately(0.000025f, 1e-10f);
    }

    [Fact]
    public void NeweyWest_DecimalTranslationPreservesSmallVariance()
    {
        var series = Enumerable.Range(0, 10).Select(i => 10000000000000000000000000000m + i % 2).ToArray();
        decimal actual = 0;
        Action compute = () => actual = NeweyWestVariance<decimal>.MeanVariance(series, 0);
        compute.Should().NotThrow();
        actual.Should().Be(0.025m);
    }
}
