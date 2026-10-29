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

/// <summary>Independent exact-rational component pins for the declared library variant.</summary>
public sealed class GtInspiredVariantContractTests
{
    [Fact]
    public void PublicName_ExplicitlyIdentifiesVariant()
    {
        typeof(GtInspiredScore<double>).Name.Should().Be("GtInspiredScore`1");
        typeof(GtInspiredScoreResult<double>).Name.Should().Be("GtInspiredScoreResult`1");
        typeof(GtInspiredScore<double>).Assembly.GetType("Boutquin.Numerics.Statistics.GeneralizationScore`1")
            .Should().BeNull("the RC removes the ambiguous former API name");
    }

    [Fact]
    public void Components_SingleLoss_UseSampleSpreadCumulativeReturnAndAllObservationRms()
    {
        // Exact arithmetic: mean=11/4, sample variance=83/12;
        // cumulative returns=[3,7,12,11], squared correlation=29/35;
        // downside mean square=1/4. None comes from the production implementation.
        var result = GtInspiredScore<double>.Compute([3, 4, 5, -1], 0, .125, 1);
        result.Mu.Should().BeApproximately(11.0 / 4, 1e-13);
        result.Z.Should().BeApproximately(11.0 / 4 * Math.Sqrt(48.0 / 83), 1e-13);
        result.RSquared.Should().BeApproximately(29.0 / 35, 1e-13);
        result.DownsideDeviation.Should().BeApproximately(.5, 1e-13);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void DownsideRms_RepeatedLossAllNonnegativeAndMixedSeries(int index)
    {
        double[][] series = [[3, 4, -1, -1], [1, 2, 3, 4], [-2, 3, -1, 5]];
        double[] squares = [.5, 0, 1.25];
        var result = GtInspiredScore<double>.Compute(series[index], 0, .125, 1);
        result.DownsideDeviation.Should().BeApproximately(Math.Sqrt(squares[index]), 1e-13);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Epsilon_IsAddedToDownsideForPositiveAndMixedSeries(int index)
    {
        double[][] series = [[1, 2, 3, 4], [3, 4, 5, -1]];
        double[] means = [2.5, 2.75];
        double[] variances = [5.0 / 3, 83.0 / 12];
        double[] rSquared = [45.0 / 46, 29.0 / 35];
        double[] downside = [0, .5];
        var expected = -means[index] * Math.Log(means[index] / Math.Sqrt(variances[index] / 4))
            * rSquared[index] / (downside[index] + .125);
        var result = GtInspiredScore<double>.Compute(series[index], 0, .125, 1);
        result.Score.Should().BeApproximately(expected, 1e-12);
    }
}
