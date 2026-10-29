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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class Arena20261003C3RegressionTests
{
    [Fact]
    public void R4HalfLinear_LongStream_PreservesRepresentableIncrement()
    {
        var estimator = new PSquareEstimator<Half>(0.5);
        for (var i = 0; i < 300000; i++)
        {
            estimator.Add((Half)0);
        }

        estimator.Add((Half)60000);
        const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
        var q = (Half[])typeof(PSquareEstimator<Half>).GetField("_q", flags)!.GetValue(estimator)!;
        var n = (long[])typeof(PSquareEstimator<Half>).GetField("_n", flags)!.GetValue(estimator)!;
        var expected = (Half)((double)q[3] + ((double)q[4] - (double)q[3]) / (n[4] - n[3]));
        var actual = (Half)typeof(PSquareEstimator<Half>).GetMethod("Linear", flags)!.Invoke(estimator, [3, 1])!;
        actual.Should().Be(expected);
    }

    [Fact]
    public void R4Pearson_SubnormalAxis_PreservesPerfectCorrelation()
    {
        var moments = new WelfordMoments<double>.Pearson();
        moments.Add(0, 0);
        moments.Add(double.Epsilon, 1e300);
        moments.Correlation.Should().BeApproximately(1, 1e-14);
        moments.Add(double.Epsilon, 1e300);
        moments.Correlation.Should().BeApproximately(1, 1e-14);
    }

    [Fact]
    public void R4Covariance_SubnormalAxis_PreservesRepresentableProduct()
    {
        var moments = new WelfordMoments<double>.Pearson();
        moments.Add(0, 0);
        moments.Add(double.Epsilon, 1e300);
        var expected = (double.Epsilon * 1e300) / 2;
        moments.Covariance.Should().BeApproximately(expected, expected * 1e-14);
    }

    [Fact]
    public void R4Covariance_DecimalTinyAxis_PreservesRepresentableProduct()
    {
        var moments = new WelfordMoments<decimal>.Pearson();
        moments.Add(0, 0);
        moments.Add(0.0000000000000000000000000001m, 10000000000000000000000000000m);
        moments.Covariance.Should().Be(0.5m);
    }

    [Fact]
    public void R4Merge_TranslatedPartitions_PreservesResidualMean()
    {
        const double translation = 9007199254740992;
        var left = new WelfordMoments<double>();
        var right = new WelfordMoments<double>();
        foreach (var offset in new[] { 0, 2 })
        {
            left.Add(translation + offset);
        }

        foreach (var offset in new[] { 4, 8, 14 })
        {
            right.Add(translation + offset);
        }

        left.Merge(right);
        left.Variance.Should().BeApproximately(30.8, 1e-12);
        left.Add(translation + 16);
        left.Variance.Should().BeApproximately(42.66666666666667, 1e-12);
    }

    [Fact]
    public void R4Percentile_HalfTinyProbability_PreservesAmplifiedFraction()
    {
        Percentile.Compute<Half>([(Half)0, (Half)60000], 1e-9).Should().Be((Half)0.00006);
    }

    [Fact]
    public void R4PSquareStartup_HalfTinyProbability_PreservesAmplifiedFraction()
    {
        var estimator = new PSquareEstimator<Half>(1e-9);
        estimator.Add((Half)0);
        estimator.Add((Half)60000);
        estimator.Estimate.Should().Be((Half)0.00006);
    }

    [Fact]
    public void R4Skewness_HalfOneOutlier_PreservesFiniteMoment()
    {
        var values = new Half[65519];
        values[^1] = (Half)1;
        SampleSkewness<Half>.Compute(values).Should().Be((Half)Math.Sqrt(values.Length));
    }

    [Fact]
    public void R4Kurtosis_HalfOneOutlier_PreservesFiniteMoment()
    {
        var values = new Half[65519];
        values[^1] = (Half)1;
        SampleExcessKurtosis<Half>.Compute(values).Should().Be((Half)values.Length);
    }

    [Theory]
    [InlineData(1e200)]
    [InlineData(1e308)]
    [InlineData(1e-200)]
    [InlineData(1e-300)]
    public void WelfordStdDev_DoublePair_PreservesRepresentableRoot(double magnitude)
    {
        var moments = new WelfordMoments<double>();
        moments.Add(-magnitude);
        moments.Add(magnitude);

        // The sample mean is zero and the N-1 divisor is one: s = sqrt(2) * a.
        (moments.StdDev / magnitude).Should().BeApproximately(Math.Sqrt(2), 1e-14);
    }

    [Theory]
    [InlineData(20)]
    [InlineData(-20)]
    public void WelfordStdDev_DecimalPair_PreservesRepresentableRoot(int exponent)
    {
        var magnitude = exponent > 0 ? 100000000000000000000m : 0.00000000000000000001m;
        var moments = new WelfordMoments<decimal>();
        moments.Add(-magnitude);
        moments.Add(magnitude);

        var read = () => moments.StdDev;
        read.Should().NotThrow();
        // At 1e-20 the final decimal has only nine significant digits.
        (read() / magnitude).Should().BeApproximately(1.4142135623730950488016887242m, 0.00000001m);
    }

    [Theory]
    [InlineData(1e200)]
    [InlineData(1e-200)]
    public void WelfordStdDev_MergedPartitions_PreservesRepresentableRoot(double magnitude)
    {
        var left = new WelfordMoments<double>();
        var right = new WelfordMoments<double>();
        left.Add(-magnitude);
        right.Add(magnitude);
        var pair = WelfordMoments<double>.Combine(left, right);
        (pair.StdDev / magnitude).Should().BeApproximately(Math.Sqrt(2), 1e-14);

        // Merge two internally varying partitions with equal means: the cross term is zero.
        pair.Merge(pair);
        (pair.StdDev / magnitude).Should().BeApproximately(Math.Sqrt(4.0 / 3), 1e-14);
    }

    [Fact]
    public void ExcessKurtosis_HalfAtAcceptedCountEndpoint_MatchesExactBinaryMoment()
    {
        var values = Enumerable.Range(0, 65519).Select(i => (Half)(i % 2)).ToArray();

        // Direct binary-population moments reduce the adjusted excess to this rational.
        var expected = -131038.0 / 65517;
        ((double)SampleExcessKurtosis<Half>.Compute(values)).Should().BeApproximately(expected, 0.01);
    }

    [Theory]
    [InlineData(3, 4, 1)]
    [InlineData(4, 3, -1)]
    [InlineData(3, 3, 0)]
    public void KendallTauB_TwoObservations_MatchesSinglePair(int firstY, int secondY, int expected)
    {
        RankCorrelation<double>.KendallTauB([1, 2], [firstY, secondY]).Should().Be(expected);
        RankCorrelation<decimal>.KendallTauB([1, 2], [firstY, secondY]).Should().Be(expected);
        RankCorrelation<float>.KendallTauB([1, 2], [firstY, secondY]).Should().Be(expected);
        ((double)RankCorrelation<Half>.KendallTauB([(Half)1, (Half)2], [(Half)firstY, (Half)secondY])).Should().Be(expected);
        RankCorrelation<double>.Spearman([1, 2], [firstY, secondY]).Should().Be(0);
    }
}
