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

public sealed class PercentileTests
{
    [Fact]
    public void Compute_EmptyArray_ReturnsZero()
    {
        Percentile.Compute<decimal>([], 0.5).Should().Be(0m);
    }

    [Fact]
    public void Compute_SingleElement_ReturnsElement()
    {
        Percentile.Compute([42m], 0.95).Should().Be(42m);
    }

    [Fact]
    public void Compute_EndPoints()
    {
        decimal[] sorted = [1m, 2m, 3m, 4m, 5m];
        Percentile.Compute(sorted, 0.0).Should().Be(1m);
        Percentile.Compute(sorted, 1.0).Should().Be(5m);
    }

    [Fact]
    public void Compute_FiniteExtremeLowerEndpoint_ReturnsExactSample()
    {
        double[] sorted = [-double.MaxValue, double.MaxValue];

        Percentile.Compute(sorted, 0.0).Should().Be(-double.MaxValue);
        Percentile.Compute(sorted, 1.0).Should().Be(double.MaxValue);
    }

    [Fact]
    public void Compute_FiniteOppositeSignExtremes_InterpolatesWithoutOverflow()
    {
        double[] sorted = [-double.MaxValue, double.MaxValue];

        Percentile.Compute(sorted, 0.5).Should().Be(0.0);
    }

    [Fact]
    public void Compute_FiniteExtremeExactInteriorRank_ReturnsExactSample()
    {
        double[] sorted = [-double.MaxValue, -double.MaxValue, double.MaxValue];

        Percentile.Compute(sorted, 0.5).Should().Be(-double.MaxValue);
    }

    [Fact]
    public void Compute_ExactRanks_PreserveNegativeZeroSample()
    {
        double[] endpoint = [-0.0, 1.0];
        double[] interior = [-1.0, -0.0, 1.0];
        double[] upperEndpoint = [-1.0, -0.0];

        BitConverter.DoubleToInt64Bits(Percentile.Compute(endpoint, 0.0))
            .Should().Be(BitConverter.DoubleToInt64Bits(-0.0));
        BitConverter.DoubleToInt64Bits(Percentile.Compute(interior, 0.5))
            .Should().Be(BitConverter.DoubleToInt64Bits(-0.0));
        BitConverter.DoubleToInt64Bits(Percentile.Compute(upperEndpoint, 1.0))
            .Should().Be(BitConverter.DoubleToInt64Bits(-0.0));
    }

    [Theory]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Compute_RejectsNonFiniteSingleElement(double value)
    {
        double[] sorted = [value];

        FluentActions.Invoking(() => Percentile.Compute(sorted, 0.5))
            .Should().Throw<ArgumentException>().WithParameterName("sorted");
    }

    [Fact]
    public void Compute_RejectsNonFiniteSampleOutsideSelectedRank()
    {
        double[] sorted = [-1.0, double.PositiveInfinity];

        FluentActions.Invoking(() => Percentile.Compute(sorted, 0.0))
            .Should().Throw<ArgumentException>().WithParameterName("sorted");
    }

    [Fact]
    public void Compute_Median_MatchesMiddle()
    {
        decimal[] sorted = [1m, 2m, 3m, 4m, 5m];
        Percentile.Compute(sorted, 0.5).Should().Be(3m);
    }

    [Fact]
    public void Compute_LinearInterpolation_MatchesNumPyConvention()
    {
        // For sorted [1, 2, 3, 4], p=0.25 → index 0.75 → 1 + 0.75*(2-1) = 1.75
        decimal[] sorted = [1m, 2m, 3m, 4m];
        Percentile.Compute(sorted, 0.25).Should().Be(1.75m);
    }

    [Fact]
    public void Compute_ClampsFiniteOutOfRangeProbability()
    {
        decimal[] sorted = [1m, 2m, 3m];
        Percentile.Compute(sorted, -0.1).Should().Be(1m);
        Percentile.Compute(sorted, 1.5).Should().Be(3m);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Compute_RejectsNonFiniteProbabilityBeforeLengthShortcuts(double p)
    {
        FluentActions.Invoking(() => Percentile.Compute<double>([], p))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(p));
        FluentActions.Invoking(() => Percentile.Compute([42.0], p))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(p));
    }

    [Fact]
    public void Compute_RejectsUnsortedInput()
    {
        double[] unsorted = [3.0, 1.0, 2.0];

        FluentActions.Invoking(() => Percentile.Compute(unsorted, 0.5))
            .Should().Throw<ArgumentException>().WithParameterName("sorted");
    }
}
