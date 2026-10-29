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

using System.Reflection;
using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3PwAuthorListingTests
{
    [Fact]
    public void AuthorListing_paired_window_ExactRationalCovarianceOracle()
    {
        // Common-row Fraction oracle: m=3,M=6,G=702422/4959,g=444272/1653.
        double[] series = [8, 3, 14, 12, 18, 20, 16, 11, 5, 0, 0, 2, -4, -4, 0, 5, 15, 12, 12, 12, 9, 5, -7, -7, -13, 3, 7, 4, 12, 1, -1, -6, -16, -16, -6, -4, -5, 4, -6, -4, 10, 8, 11, 14, 15, 4, 0, 2, 0, 11, 24, 22, 12, 15, 7, 11, 7, -7, -12, -16, -1, 10, 20, 13];
        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(2.6098282616481647, 1e-11);
    }

    [Fact]
    public void AuthorListing_fallback_ExactRationalCovarianceOracle()
    {
        // Common-row Fraction oracle: m=13,M=13,G=17364/5525,g=3514/5525.
        double[] series = [0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 1, 2, 3, 2, 1];
        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(11.604470744866159, 1e-11);
    }

    [Fact]
    public void AuthorListing_cap_ExactRationalCovarianceOracle()
    {
        // Common-row Fraction oracle: m=12,M=13,G=-2853/425,g=-1132/5525.
        double[] series = [3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0, 3, 4, -3, 2, 4, 3, -3, 0];
        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(22, 1e-11);
    }

    [Fact]
    public void AuthorListing_below_one_ExactRationalCovarianceOracle()
    {
        // Common-row Fraction oracle: m=1,M=2,G=-76/145,g=240/29.
        double[] series = [-1, 3, 2, -4, 4, 0, 0, -1, -1, -4, -5, -4, 1, -3, 0, -4, -1, 0, 4, -4, -2, 1, -2, 1, 0, 4, 5, 1, -3, 2, -4, 5];
        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(0.50443462597769073, 1e-11);
    }

    [Fact]
    public void AuthorListing_ZeroTruncatedVariance_UsesMZeroReturn()
    {
        var series = new double[64];
        series[0] = 1;
        series[1] = -1;
        PolitisWhiteBlockLength.EstimateDouble(series).Should().Be(1);
    }

    [Fact]
    public void AuthorListing_PairedCovariance_ExactMeansAndSampleDenominator()
    {
        var method = typeof(PolitisWhiteBlockLength).GetMethod("PairedCovariance",
            BindingFlags.NonPublic | BindingFlags.Static)!;
        var covariance = method.CreateDelegate<Covariance>();
        var actual = covariance([0, 1, 4, 2, 8, 3], 2, 1, out var left, out var right);
        // Four paired rows: [4,2,8,3] and [1,4,2,8], means17/4 and15/4.
        actual.Should().BeApproximately(-47.0 / 12, 1e-14);
        left.Should().BeApproximately(83.0 / 12, 1e-14);
        right.Should().BeApproximately(115.0 / 12, 1e-14);
    }

    private delegate double Covariance(ReadOnlySpan<double> data, int start, int lag,
        out double leftVariance, out double rightVariance);
}
