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

public sealed class C3IndependentMomentsAndRanksTests
{
    [Fact]
    public void Welford_PrefixMomentsMatchDirectRationalSums()
    {
        // Direct sums for {1}, {1,2}, {1,2,4}: M2=0,1/2,14/3.
        double[] data = [1, 2, 4];
        double[] means = [1, 1.5, 7.0 / 3];
        double[] sample = [0, 0.5, 7.0 / 3];
        double[] population = [0, 0.25, 14.0 / 9];
        var accumulator = new WelfordMoments<double>();
        for (var i = 0; i < data.Length; i++)
        {
            accumulator.Add(data[i]);
            accumulator.Count.Should().Be(i + 1);
            accumulator.Mean.Should().BeApproximately(means[i], 1e-14);
            accumulator.Variance.Should().BeApproximately(sample[i], 1e-14);
            accumulator.PopulationVariance.Should().BeApproximately(population[i], 1e-14);
        }
    }

    [Fact]
    public void Welford_MergeRetainsWithinAndBetweenPartitionVariation()
    {
        // Direct centered sum for {1,2,4,8} is115/4, mean15/4.
        // The within-block sums are1/2 and8; between-block correction81/4.
        var left = new WelfordMoments<double>();
        left.Add(1);
        left.Add(2);
        var right = new WelfordMoments<double>();
        right.Add(4);
        right.Add(8);
        var combined = WelfordMoments<double>.Combine(left, right);
        combined.Count.Should().Be(4);
        combined.Mean.Should().Be(3.75);
        combined.Variance.Should().BeApproximately(115.0 / 12, 1e-14);
        combined.PopulationVariance.Should().Be(115.0 / 16);
        left.Mean.Should().Be(1.5);
        right.Mean.Should().Be(6);
    }

    [Fact]
    public void Welford_PairedCorrectedProductMatchesDirectCenteredSums()
    {
        // x={1,2,4},y={3,-1,2}: Sxx=14/3,Syy=26/3,Sxy=-1/3.
        var accumulator = new WelfordMoments<double>.Pearson();
        accumulator.Add(1, 3);
        accumulator.Add(2, -1);
        accumulator.Add(4, 2);
        accumulator.Covariance.Should().BeApproximately(-1.0 / 6, 1e-14);
        accumulator.Correlation.Should().BeApproximately(-1 / Math.Sqrt(364), 1e-14);
    }

    [Fact]
    public void RankCorrelation_OverlappingTiesHaveIndependentExactCounts()
    {
        // Six pairs: C=3,D=2,Tx=Ty=1, so tau=(3-2)/sqrt(5*5)=1/5.
        // Average ranks x=(1.5,1.5,3,4),y=(2.5,2.5,1,4).
        // Centered dot product3/2 and each squared norm9/2 give rho=1/3.
        double[] x = [1, 1, 2, 3];
        double[] y = [2, 2, 1, 3];
        RankCorrelation<double>.KendallTauB(x, y).Should().BeApproximately(0.2, 1e-14);
        RankCorrelation<double>.Spearman(x, y).Should().BeApproximately(1.0 / 3, 1e-14);
    }
}
