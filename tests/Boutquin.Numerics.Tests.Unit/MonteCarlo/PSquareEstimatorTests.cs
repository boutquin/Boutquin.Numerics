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

public sealed class PSquareEstimatorTests
{
    [Fact]
    public void Ctor_RejectsOutOfRangePercentile()
    {
        FluentActions.Invoking(() => new PSquareEstimator(0.0))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new PSquareEstimator(1.0))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new PSquareEstimator(double.NaN))
            .Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Add_RejectsNonFiniteObservation()
    {
        var e = new PSquareEstimator(0.5);
        FluentActions.Invoking(() => e.Add(double.NaN))
            .Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => e.Add(double.PositiveInfinity))
            .Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Estimate_IsZeroBeforeAnyAdd()
    {
        new PSquareEstimator(0.5).Estimate.Should().Be(0.0);
    }

    [Fact]
    public void Estimate_ConvergesToTrueMedianForUniform()
    {
        var rng = new System.Random(2026);
        var e = new PSquareEstimator(0.5);
        for (var i = 0; i < 100_000; i++)
        {
            e.Add(rng.NextDouble());
        }

        // True median of U(0,1) is 0.5.
        Math.Abs(e.Estimate - 0.5).Should().BeLessThan(0.01);
        e.Count.Should().Be(100_000);
    }

    [Fact]
    public void Estimate_ConvergesTo95thPercentileForUniform()
    {
        var rng = new System.Random(42);
        var e = new PSquareEstimator(0.95);
        for (var i = 0; i < 100_000; i++)
        {
            e.Add(rng.NextDouble());
        }

        Math.Abs(e.Estimate - 0.95).Should().BeLessThan(0.01);
    }

    [Fact]
    public void Estimate_WorksWithFewerThanFiveObservations()
    {
        var e = new PSquareEstimator(0.5);
        e.Add(1.0);
        e.Add(3.0);
        e.Add(5.0);
        // Median of {1, 3, 5} via linear interpolation is 3.0.
        e.Estimate.Should().Be(3.0);
        e.Count.Should().Be(3);
    }

    [Fact]
    public void Estimate_PercentilePropertyIsPreserved()
    {
        new PSquareEstimator(0.99).Percentile.Should().Be(0.99);
    }
}
