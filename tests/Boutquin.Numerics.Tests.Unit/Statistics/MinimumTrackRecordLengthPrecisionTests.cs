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

public sealed class MinimumTrackRecordLengthPrecisionTests
{
    [Fact]
    public void Compute_HalfSmallSignificance_ReturnsRepresentableTrackLength()
    {
        Func<Half> compute = () => MinimumTrackRecordLength<Half>.Compute((Half)2, 0.0001, 252);

        var actual = compute.Should().NotThrow().Subject;

        Half.IsFinite(actual).Should().BeTrue();
        ((double)actual).Should().BeApproximately(879.2738098122429, 2.0);
    }

    [Fact]
    public void Compute_HalfHighObservationFrequency_ReturnsFiniteTrackLength()
    {
        var actual = MinimumTrackRecordLength<Half>.Compute((Half)2, 0.05, 70_000);

        Half.IsFinite(actual).Should().BeTrue();
        ((double)actual).Should().BeApproximately(47349.36321839675, 200);
    }

    [Fact]
    public void Compute_DoubleTinySignificance_DoesNotRoundProbabilityToEndpoint()
    {
        Func<double> compute = () => MinimumTrackRecordLength<double>.Compute(2, 1e-20, 252);

        var actual = compute.Should().NotThrow().Subject;

        // Gaussian MinTRL simplifies to 1 + z² * (periods / annualSharpe² + 1/2).
        // Independent standard-normal quantile at 1e-20 is -9.262340089798409.
        var expected = 1 + 9.262340089798409 * 9.262340089798409 * (252.0 / 4 + 0.5);
        actual.Should().BeApproximately(expected, expected * 1e-12);
    }
}
