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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Round-1 code-arena fixes (batch B08-moments): WelfordMoments{T}.Add/Merge and the
/// nested Pearson accumulator must not corrupt the running mean/correlation to Inf/NaN,
/// nor throw for decimal, when individual observations are finite but an intermediate
/// anchor-relative subtraction would overflow the type's representable range.
/// </summary>
public sealed class WelfordOverflowRegressionTests
{
    [Theory]
    [InlineData(3, 200000000000000d)]
    [InlineData(6, 300000000000000d)]
    [InlineData(4, 200000000000000d)]
    [InlineData(1000, 100000000000000d)]
    public void Covariance_RepresentableScaledResult_DividesBeforeRestoringScale(int count, double magnitude)
    {
        var paired = new WelfordMoments<decimal>.Pearson();
        var single = new WelfordMoments<decimal>();
        for (var i = 0; i < count; i++)
        {
            var value = count == 3 && i == 2 || count == 6 && i >= 2
                ? 0m : (i % 2 == 0 ? -1m : 1m) * (decimal)magnitude;
            paired.Add(value, value);
            single.Add(value);
        }

        var read = () => paired.Covariance;
        read.Should().NotThrow();
        paired.Covariance.Should().BeApproximately(single.Variance, single.Variance * 0.000000000000000000000001m);
    }

    [Fact]
    public void Covariance_RepresentableIeeeResult_RemainsFinite()
    {
        var paired = new WelfordMoments<double>.Pearson();
        foreach (var value in new[] { -1e154, 0, 1e154 })
        {
            paired.Add(value, value);
        }

        paired.Covariance.Should().BeApproximately(1e308, 1e294);
        var half = new WelfordMoments<Half>.Pearson();
        foreach (var value in new[] { 300, -300, 0, 0, 0, 0 })
        {
            half.Add((Half)value, (Half)value);
        }

        ((double)half.Covariance).Should().BeApproximately(36000, 64);
    }

    [Fact]
    public void Merge_AsymmetricExtremeDecimalPartitions_PreservesRepresentableMean()
    {
        var left = new WelfordMoments<decimal>();
        left.Add(decimal.MinValue);
        var right = new WelfordMoments<decimal>();
        for (var i = 0; i < 3; i++)
        {
            right.Add(decimal.MaxValue);
        }

        var merge = () => left.Merge(right);
        merge.Should().NotThrow();
        left.Count.Should().Be(4);
        left.Mean.Should().BeApproximately(decimal.MaxValue / 2m, 2m);
    }

    // F-L1-02: WelfordMoments<T>.Add overflows the anchor residual for a finite,
    // opposite-sign, near-MaxValue pair even though the true mean (0) is representable.
    [Fact]
    public void Add_OppositeSignExtremeDoublePair_MeanStaysFinite()
    {
        var moments = new WelfordMoments<double>();
        moments.Add(1e308);
        moments.Add(-1e308);

        moments.Mean.Should().Be(0.0);
        double.IsNaN(moments.Variance).Should().BeFalse();
    }

    [Fact]
    public void Add_OppositeSignExtremeDecimalPair_DoesNotThrow()
    {
        var moments = new WelfordMoments<decimal>();
        moments.Add(decimal.MinValue);

        var act = () => moments.Add(decimal.MaxValue);

        act.Should().NotThrow();
        moments.Mean.Should().Be(0m);
    }

    // Pins the finding's third probe case exactly: an extreme opposite-sign pair
    // followed by moderate-magnitude observations must not leave the mean permanently
    // NaN-poisoned; it must recover the true (representable) mean of the whole sample.
    [Fact]
    public void Add_ExtremePairThenModerateValues_RecoversFiniteMean()
    {
        var moments = new WelfordMoments<double>();
        moments.Add(-1e308);
        moments.Add(1e308);
        for (var i = 0; i < 5; i++)
        {
            moments.Add(1e307);
        }

        double.IsNaN(moments.Mean).Should().BeFalse();
        moments.Mean.Should().BeApproximately(7.142857142857143e306, 1e293);
    }

    // Blast radius of F-L1-02: the same raw subtraction (delta = otherMean - mean)
    // overflows in Merge for two partitions whose means sit at opposite extremes.
    [Fact]
    public void Merge_OppositeSignExtremePartitionMeans_CombinedMeanStaysFinite()
    {
        var left = new WelfordMoments<double>();
        left.Add(1e308);
        left.Add(1e308);

        var right = new WelfordMoments<double>();
        right.Add(-1e308);
        right.Add(-1e308);

        left.Merge(right);

        left.Mean.Should().Be(0.0);
        left.Count.Should().Be(4);
    }

    // F-L1-03 / F-L5-02: the online Pearson accumulator must stay finite and agree with
    // the batch PearsonCorrelation sibling (which normalizes by scale) at magnitudes
    // where the raw co-moment accumulation would otherwise overflow to NaN/Infinity.
    [Fact]
    public void PearsonAdd_ExtremeScaleDoublePairs_MatchesBatchCorrelation()
    {
        double[] baseX = [1, 2, 3, 4, 5, 6];
        double[] baseY = [2, 1, 5, 3, 8, 4];
        const double scale = 1e200;
        var xs = new double[baseX.Length];
        var ys = new double[baseY.Length];
        for (var i = 0; i < baseX.Length; i++)
        {
            xs[i] = baseX[i] * scale;
            ys[i] = baseY[i] * scale;
        }

        var expected = PearsonCorrelation<double>.Compute(xs, ys);

        var online = new WelfordMoments<double>.Pearson();
        for (var i = 0; i < xs.Length; i++)
        {
            online.Add(xs[i], ys[i]);
        }

        double.IsNaN(online.Correlation).Should().BeFalse();
        online.Correlation.Should().BeApproximately(expected, 1e-9);
    }

    // F-L23-01 / F-L5-02: the streaming Pearson accumulator must not throw
    // OverflowException for decimal inputs of legitimately representable magnitude and
    // spread that its univariate sibling (WelfordMoments<decimal>.Add) already handles,
    // and its result must agree with the batch PearsonCorrelation<decimal> sibling.
    [Fact]
    public void PearsonAdd_LargeSpreadDecimalPairs_DoesNotThrowAndMatchesBatch()
    {
        var rng = new System.Random(42);
        var xs = new decimal[200];
        var ys = new decimal[200];
        for (var i = 0; i < xs.Length; i++)
        {
            var v = 1_000_000_000_000_00m + (decimal)(rng.NextDouble() * 2e14 - 1e14);
            xs[i] = v;
            ys[i] = v * 1.0000001m;
        }

        var expected = PearsonCorrelation<decimal>.Compute(xs, ys);

        var online = new WelfordMoments<decimal>.Pearson();
        var act = () =>
        {
            foreach (var (x, y) in xs.Zip(ys))
            {
                online.Add(x, y);
            }
        };

        act.Should().NotThrow();
        Math.Abs(online.Correlation - expected).Should().BeLessThan(0.0001m);
    }
}
