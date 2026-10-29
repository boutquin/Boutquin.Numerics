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

public sealed class ReleaseQualificationPSquareTests
{
    [Theory]
    [InlineData(.1, 71, 8)]
    [InlineData(.05, 81, 5)]
    [InlineData(.9, 21, 18)]
    [InlineData(.3, 21, 6)]
    [InlineData(.1, 1000, 100)]
    [InlineData(.3, 1000, 300)]
    public void NonDyadicProbability_MarkerDecisionsMatchExactPaperRecurrence(double p, int count, int estimate)
    {
        // Exact Fraction Box 1 replay with the exact binary-double p; retained
        // generator: tests/Verification/generate_release_p2_oracles.py.
        var wide = new PSquareEstimator<double>(p);
        var precise = new PSquareEstimator<decimal>(p);
        for (var value = 1; value <= count; value++)
        {
            wide.Add(value);
            precise.Add(value);
        }
        wide.Estimate.Should().BeApproximately(estimate, 1e-10);
        precise.Estimate.Should().BeApproximately(estimate, 1e-10m);
        wide.Count.Should().Be(count);
        precise.Count.Should().Be(count);
    }
}
