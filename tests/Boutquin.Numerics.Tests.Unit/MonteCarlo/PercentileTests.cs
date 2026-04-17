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
        Percentile.Compute([], 0.5m).Should().Be(0m);
    }

    [Fact]
    public void Compute_SingleElement_ReturnsElement()
    {
        Percentile.Compute([42m], 0.95m).Should().Be(42m);
    }

    [Fact]
    public void Compute_EndPoints()
    {
        decimal[] sorted = [1m, 2m, 3m, 4m, 5m];
        Percentile.Compute(sorted, 0m).Should().Be(1m);
        Percentile.Compute(sorted, 1m).Should().Be(5m);
    }

    [Fact]
    public void Compute_Median_MatchesMiddle()
    {
        decimal[] sorted = [1m, 2m, 3m, 4m, 5m];
        Percentile.Compute(sorted, 0.5m).Should().Be(3m);
    }

    [Fact]
    public void Compute_LinearInterpolation_MatchesNumPyConvention()
    {
        // For sorted [1, 2, 3, 4], p=0.25 → index 0.75 → 1 + 0.75*(2-1) = 1.75
        decimal[] sorted = [1m, 2m, 3m, 4m];
        Percentile.Compute(sorted, 0.25m).Should().Be(1.75m);
    }

    [Fact]
    public void Compute_ClampsOutOfRangeProbability()
    {
        decimal[] sorted = [1m, 2m, 3m];
        Percentile.Compute(sorted, -0.1m).Should().Be(1m);
        Percentile.Compute(sorted, 1.5m).Should().Be(3m);
    }
}
