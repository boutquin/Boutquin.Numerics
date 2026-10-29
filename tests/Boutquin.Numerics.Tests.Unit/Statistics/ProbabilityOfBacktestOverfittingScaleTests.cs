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

public sealed class ProbabilityOfBacktestOverfittingScaleTests
{
    [Fact]
    public void Compute_TinyDoubleReturns_PreservesScaleInvariantFoldRanks()
    {
        var original = new double[4, 16];
        var scaled = new double[4, 16];
        for (var strategy = 0; strategy < 4; strategy++)
        {
            for (var observation = 0; observation < 16; observation++)
            {
                original[strategy, observation] = Math.Sin((strategy + 1) * (observation + 1) * 1.234) + observation * 0.1;
                scaled[strategy, observation] = original[strategy, observation] * 1e-200;
            }
        }

        var expected = ProbabilityOfBacktestOverfitting<double>.Compute(original, 4);
        var actual = ProbabilityOfBacktestOverfitting<double>.Compute(scaled, 4);

        expected.Pbo.Should().Be(5.0 / 6.0);
        actual.Pbo.Should().Be(expected.Pbo);
        actual.LogitValues.Should().Equal(expected.LogitValues);
        actual.LogitMedian.Should().Be(expected.LogitMedian);
    }

    [Fact]
    public void Compute_TinyDecimalReturns_PreservesScaleInvariantFoldRanks()
    {
        var original = new decimal[4, 16];
        var scaled = new decimal[4, 16];
        for (var strategy = 0; strategy < 4; strategy++)
        {
            for (var observation = 0; observation < 16; observation++)
            {
                original[strategy, observation] = (decimal)(Math.Sin((strategy + 1) * (observation + 1) * 1.234) + observation * 0.1);
                scaled[strategy, observation] = original[strategy, observation] * 1e-15m;
            }
        }

        var expected = ProbabilityOfBacktestOverfitting<decimal>.Compute(original, 4);
        var actual = ProbabilityOfBacktestOverfitting<decimal>.Compute(scaled, 4);

        expected.Pbo.Should().Be(5m / 6m);
        actual.Pbo.Should().Be(expected.Pbo);
        actual.LogitValues.Should().Equal(expected.LogitValues);
        actual.LogitMedian.Should().Be(expected.LogitMedian);
    }
}
