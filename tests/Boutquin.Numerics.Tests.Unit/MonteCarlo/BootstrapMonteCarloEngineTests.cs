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
    public void GenericScalarResult_SnapshotsStatisticsOnBothEdges()
    {
        decimal[] input = [1m, 2m];
        var result = new BootstrapMonteCarloResult<decimal>(2, input, 1m, 1m, 2m, 1.5m);
        input[0] = 99m;
        result.Statistics[0].Should().Be(1m);
        (result.Statistics as decimal[]).Should().BeNull();

        var initialized = result with { Statistics = input };
        input[0] = 88m;
        initialized.Statistics[0].Should().Be(99m);
    }

    [Fact]
    public void GenericMultiResult_DeepSnapshotsEveryCollectionOnBothEdges()
    {
        string[] names = ["a"];
        decimal[] row = [1m, 2m];
        decimal[][] rows = [row];
        decimal[] summaries = [3m];
        var result = new MultiStatisticBootstrapResult<decimal>(2, names, rows, summaries, summaries, summaries, summaries);
        names[0] = "b";
        row[0] = 99m;
        summaries[0] = 88m;
        result.Names[0].Should().Be("a");
        result.Statistics[0][0].Should().Be(1m);
        result.Means[0].Should().Be(3m);
        result.Medians[0].Should().Be(3m);
        result.Percentile5s[0].Should().Be(3m);
        result.Percentile95s[0].Should().Be(3m);
        result.Statistics[0][0] = 77m;
        result.Statistics[0][0].Should().Be(1m);
        (result.Names as string[]).Should().BeNull();
        (result.Means as decimal[]).Should().BeNull();
        (result.Medians as decimal[]).Should().BeNull();
        (result.Percentile5s as decimal[]).Should().BeNull();
        (result.Percentile95s as decimal[]).Should().BeNull();
    }

    [Fact]
    public void LegacyScalarResult_SnapshotsStatisticsOnBothEdges()
    {
        decimal[] input = [1m, 2m];
        var result = new BootstrapMonteCarloResult(2, input, 1m, 1m, 2m, 1.5m);
        input[0] = 99m;
        result.Statistics[0].Should().Be(1m);
        (result.Statistics as decimal[]).Should().BeNull();

        var initialized = result with { Statistics = input };
        input[0] = 88m;
        initialized.Statistics[0].Should().Be(99m);
    }

    [Fact]
    public void LegacyMultiResult_DeepSnapshotsEveryCollectionOnBothEdges()
    {
        string[] names = ["a"];
        decimal[] row = [1m, 2m];
        decimal[][] rows = [row];
        decimal[] summaries = [3m];
        var result = new MultiStatisticBootstrapResult(2, names, rows, summaries, summaries, summaries, summaries);
        names[0] = "b";
        row[0] = 99m;
        summaries[0] = 88m;
        result.Names[0].Should().Be("a");
        result.Statistics[0][0].Should().Be(1m);
        result.Means[0].Should().Be(3m);
        result.Medians[0].Should().Be(3m);
        result.Percentile5s[0].Should().Be(3m);
        result.Percentile95s[0].Should().Be(3m);
        result.Statistics[0][0] = 77m;
        result.Statistics[0][0].Should().Be(1m);
        (result.Names as string[]).Should().BeNull();
        (result.Means as decimal[]).Should().BeNull();
        (result.Medians as decimal[]).Should().BeNull();
        (result.Percentile5s as decimal[]).Should().BeNull();
        (result.Percentile95s as decimal[]).Should().BeNull();
    }

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

    [Fact]
    public void Run_HalfConstantStatisticKeepsFiniteMean()
    {
        Half[] observations = [(Half)100];
        var result = BootstrapMonteCarloEngine<Half>.FromSeed(1000, 1)
            .Run(observations, values => values[0]);

        result.Mean.Should().Be((Half)100);
    }

    [Fact]
    public void RunMulti_HalfConstantStatisticKeepsFiniteMean()
    {
        Half[] observations = [(Half)100];
        var result = BootstrapMonteCarloEngine<Half>.FromSeed(1000, 1)
            .RunMulti(observations, 1, (values, output) => output[0] = values[0]);

        result.Means[0].Should().Be((Half)100);
    }

    [Fact]
    public void HalfLongBinaryBootstrapMeanMatchesReturnedDistribution()
    {
        Half[] observations = [(Half)0, (Half)1];
        var scalar = BootstrapMonteCarloEngine<Half>.FromSeed(10_000, 41)
            .Run(observations, values => values[0]);
        var multi = BootstrapMonteCarloEngine<Half>.FromSeed(10_000, 41)
            .RunMulti(observations, 1, (values, output) => output[0] = values[0]);

        var expected = scalar.Statistics.Sum(value => (double)value) / scalar.SimulationCount;
        ((double)scalar.Mean).Should().BeApproximately(expected, 0.001);
        ((double)multi.Means[0]).Should().BeApproximately(expected, 0.001);
    }

    [Fact]
    public void DecimalMeanPreservesCancellationResidualInScalarAndMultiRuns()
    {
        decimal[] observations = [-1e28m, 1m, 1e28m];
        var scalar = new BootstrapMonteCarloEngine<decimal>(3, new CyclingBootstrapSource())
            .Run(observations, static values => values[0]);
        var multi = new BootstrapMonteCarloEngine<decimal>(3, new CyclingBootstrapSource())
            .RunMulti(observations, 1, static (values, output) => output[0] = values[0]);

        scalar.Statistics.Should().Equal(observations);
        scalar.Mean.Should().Be(1m / 3m);
        multi.Means[0].Should().Be(1m / 3m);
    }

    private sealed class CyclingBootstrapSource : Boutquin.Numerics.Random.IRandomSource
    {
        private int _draw;

        public ulong NextULong()
        {
            var index = (_draw++ / 3) % 3;
            return (ulong)(0x10000000U + index * 0x50000000U) << 32;
        }
    }

    private static decimal Mean(ReadOnlySpan<decimal> values)
    {
        var sum = 0m;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
        }

        return sum / values.Length;
    }
}
