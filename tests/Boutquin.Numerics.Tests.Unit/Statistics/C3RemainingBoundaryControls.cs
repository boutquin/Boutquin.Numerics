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

[Trait("Tier", "Unit")]
public sealed class C3RemainingBoundaryControls
{
    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void Kendall_ValidPerfectTiedPanel_ClampsRoundoff(double direction)
    {
        // Ten pairs: one mutual tie and nine concordant (or discordant).
        // Exact tau is +/-1. Normalized counts9/10 lead to a binary rounded
        // raw quotient just outside the mathematical interval.
        double[] x = [0, 0, 1, 2, 3];
        var y = x.Select(value => direction * value).ToArray();
        RankCorrelation<double>.KendallTauB(x, y).Should().Be(direction);
    }

    [Fact]
    public void Spearman_TwoObservations_UsesSelectedZeroConvention()
    {
        RankCorrelation<double>.Spearman([1, 2], [3, 4]).Should().Be(0);
    }

    [Fact]
    public void Welford_SingletonMerge_MatchesExactCenteredSums()
    {
        var left = new WelfordMoments<double>();
        left.Add(1);
        left.Add(2);
        var singleton = new WelfordMoments<double>();
        singleton.Add(4);
        singleton.Variance.Should().Be(0);
        left.Merge(singleton);
        // Direct sums: total7, sum of squares21, M2=21-49/3=14/3.
        left.Count.Should().Be(3);
        left.Mean.Should().BeApproximately(7.0 / 3, 1e-15);
        left.Variance.Should().BeApproximately(7.0 / 3, 2e-15);
        left.PopulationVariance.Should().BeApproximately(14.0 / 9, 2e-15);
    }
}
