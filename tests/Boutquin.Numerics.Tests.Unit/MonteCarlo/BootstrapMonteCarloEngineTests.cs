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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class BootstrapMonteCarloEngineTests
{
    [Fact]
    public void Constructor_RejectsNonPositiveCount()
    {
        Action zero = () => BootstrapMonteCarloEngine.FromSeed(0);
        Action negative = () => BootstrapMonteCarloEngine.FromSeed(-1);
        zero.Should().Throw<ArgumentOutOfRangeException>();
        negative.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Run_DeterministicWithSeed()
    {
        decimal[] observations = [0.01m, -0.02m, 0.005m, 0.015m, -0.008m, 0.02m, 0.001m];
        var first = BootstrapMonteCarloEngine.FromSeed(200, 42)
            .Run(observations, Mean);
        var second = BootstrapMonteCarloEngine.FromSeed(200, 42)
            .Run(observations, Mean);

        first.Mean.Should().Be(second.Mean);
        first.Median.Should().Be(second.Median);
        first.Percentile5.Should().Be(second.Percentile5);
        first.Percentile95.Should().Be(second.Percentile95);
    }

    [Fact]
    public void Run_DistributionIsSortedAscending()
    {
        decimal[] observations = [0.01m, -0.02m, 0.005m, 0.015m, -0.008m, 0.02m];
        var result = BootstrapMonteCarloEngine.FromSeed(500, 1)
            .Run(observations, Mean);

        for (var i = 1; i < result.Statistics.Count; i++)
        {
            result.Statistics[i].Should().BeGreaterThanOrEqualTo(result.Statistics[i - 1]);
        }
    }

    [Fact]
    public void Run_PercentilesRespectOrdering()
    {
        decimal[] observations = [0.01m, -0.02m, 0.005m, 0.015m, -0.008m, 0.02m, 0.007m];
        var result = BootstrapMonteCarloEngine.FromSeed(2000, 5)
            .Run(observations, Mean);

        result.Percentile5.Should().BeLessThanOrEqualTo(result.Median);
        result.Median.Should().BeLessThanOrEqualTo(result.Percentile95);
    }

    [Fact]
    public void Run_EmptyObservations_Throws()
    {
        var engine = BootstrapMonteCarloEngine.FromSeed(10, 0);
        Action act = () => engine.Run([], Mean);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Run_MeanApproximatesSampleMean()
    {
        decimal[] observations = [0.01m, 0.02m, 0.03m, 0.04m, 0.05m];
        var sampleMean = Mean(observations);

        var result = BootstrapMonteCarloEngine.FromSeed(5000, 314)
            .Run(observations, Mean);

        Math.Abs(result.Mean - sampleMean).Should().BeLessThan(0.002m);
    }

    private static decimal Mean(decimal[] values)
    {
        var sum = 0m;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
        }

        return sum / values.Length;
    }
}
