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

public sealed class PearsonCorrelationTests
{
    [Fact]
    public void Compute_PerfectPositiveCorrelation_EqualsOne()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [2m, 4m, 6m, 8m, 10m];
        PearsonCorrelation.Compute(x, y).Should().Be(1m);
    }

    [Fact]
    public void Compute_PerfectNegativeCorrelation_EqualsMinusOne()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [10m, 8m, 6m, 4m, 2m];
        PearsonCorrelation.Compute(x, y).Should().Be(-1m);
    }

    [Fact]
    public void Compute_ConstantSeries_ReturnsZero()
    {
        decimal[] x = [1m, 1m, 1m, 1m];
        decimal[] y = [2m, 5m, 1m, 3m];
        PearsonCorrelation.Compute(x, y).Should().Be(0m);
    }

    [Fact]
    public void Compute_TooFewObservations_ReturnsZero()
    {
        decimal[] x = [1m, 2m];
        decimal[] y = [3m, 4m];
        PearsonCorrelation.Compute(x, y).Should().Be(0m);
    }

    [Fact]
    public void Compute_Slice_RespectsBounds()
    {
        decimal[] x = [100m, 1m, 2m, 3m, 4m, 5m, 100m];
        decimal[] y = [100m, 2m, 4m, 6m, 8m, 10m, 100m];
        PearsonCorrelation.Compute(x, y, start: 1, end: 6).Should().Be(1m);
    }

    [Fact]
    public void Rolling_ProducesCorrectWindowCount()
    {
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        decimal[] y = [2m, 4m, 6m, 8m, 10m];
        var result = PearsonCorrelation.Rolling(x, y, windowSize: 3);
        result.Should().HaveCount(3);
        foreach (var r in result)
        {
            r.Should().Be(1m);
        }
    }

    [Fact]
    public void Rolling_RejectsMismatchedLengths()
    {
        decimal[] x = [1m, 2m, 3m];
        decimal[] y = [1m, 2m];
        Action act = () => PearsonCorrelation.Rolling(x, y, 2);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Rolling_RejectsOutOfRangeWindow()
    {
        decimal[] x = [1m, 2m, 3m];
        decimal[] y = [1m, 2m, 3m];
        Action tooSmall = () => PearsonCorrelation.Rolling(x, y, 1);
        Action tooLarge = () => PearsonCorrelation.Rolling(x, y, 4);
        tooSmall.Should().Throw<ArgumentException>();
        tooLarge.Should().Throw<ArgumentException>();
    }
}
