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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class ArenaSuccessorC3Tests
{
    [Theory]
    [InlineData(4096)]
    [InlineData(10000)]
    public void HalfMean_AsymmetricStream_MatchesBernoulliMoments(int count)
    {
        var moments = new WelfordMoments<Half>();
        for (var i = 0; i < count; i++)
        {
            moments.Add((Half)(i % 4 == 0 ? 1 : 0));
        }

        // Exact integer counts: n/4 ones, 3n/4 zeros; central sum is 3n/16.
        ((double)moments.Mean).Should().BeApproximately(0.25, 0.001);
        ((double)moments.Variance).Should().BeApproximately(3.0 * count / (16 * (count - 1)), 0.001);
    }

    [Theory]
    [InlineData(4096)]
    [InlineData(10000)]
    public void HalfPaired_AsymmetricStream_MatchesBernoulliCovariance(int count)
    {
        var moments = new WelfordMoments<Half>.Pearson();
        for (var i = 0; i < count; i++)
        {
            var value = (Half)(i % 4 == 0 ? 1 : 0);
            moments.Add(value, value);
        }

        ((double)moments.Covariance).Should().BeApproximately(3.0 * count / (16 * (count - 1)), 0.001);
    }

    [Fact]
    public void HalfSkewness_AsymmetricStream_MatchesBernoulliMoment()
    {
        const int count = 4096;
        var values = Enumerable.Range(0, count).Select(i => (Half)(i % 4 == 0 ? 1 : 0)).ToArray();
        // Bernoulli p=1/4: population skewness 2/sqrt(3), adjusted by sqrt(n(n-1))/(n-2).
        var expected = 2 / Math.Sqrt(3) * Math.Sqrt((double)count * (count - 1)) / (count - 2);
        ((double)SampleSkewness<Half>.Compute(values)).Should().BeApproximately(expected, 0.015);
    }

    [Fact]
    public void PSquare_DecimalExtremes_PreservesFiniteMarkerBounds()
    {
        var estimator = new PSquareEstimator<decimal>(0.01);
        Action add = () =>
        {
            for (var i = 0; i < 2000; i++)
            {
                estimator.Add(i % 2 == 0 ? decimal.MinValue : decimal.MaxValue);
            }
        };

        add.Should().NotThrow("finite endpoint observations must retain finite marker heights");
        estimator.Count.Should().Be(2000);
        estimator.Estimate.Should().BeInRange(decimal.MinValue, decimal.MaxValue);
        // Positive-scale equivariance, compared with a well-scaled independent run.
        var normalized = new PSquareEstimator<decimal>(0.01);
        for (var i = 0; i < 2000; i++)
        {
            normalized.Add(i % 2 == 0 ? -1m : 1m);
        }

        (estimator.Estimate / decimal.MaxValue).Should().BeApproximately(normalized.Estimate, 0.000000000000000000000001m);
    }

    [Theory]
    [InlineData(71)]
    [InlineData(523)]
    public void Spearman_Ties_MatchesIndependentPairRanksAndLaws(int seed)
    {
        var rng = new System.Random(seed);
        for (var trial = 0; trial < 24; trial++)
        {
            var count = rng.Next(5, 14);
            var x = Enumerable.Range(0, count).Select(_ => (decimal)rng.Next(4)).ToArray();
            var y = Enumerable.Range(0, count).Select(_ => (decimal)rng.Next(4)).ToArray();
            var ranksX = PairRanks(x);
            var ranksY = PairRanks(y);
            var expected = BatchCorrelation(ranksX, ranksY);
            ((double)RankCorrelation<decimal>.Spearman(x, y)).Should().BeApproximately(expected, 1e-12);
            Array.Reverse(x);
            Array.Reverse(y);
            ((double)RankCorrelation<decimal>.Spearman(x, y)).Should().BeApproximately(expected, 1e-12);
            x = x.Select(v => 3m * v + 17m).ToArray();
            y = y.Select(v => 7m * v - 11m).ToArray();
            ((double)RankCorrelation<decimal>.Spearman(x, y)).Should().BeApproximately(expected, 1e-12);
        }

        RankCorrelation<decimal>.Spearman([1m, 2m, 2m, 3m, 4m], [1m, 2m, 3m, 3m, 5m])
            .Should().BeApproximately(35m / 38m, 0.000000000000000000000001m);
    }

    [Theory]
    [InlineData(91)]
    [InlineData(659)]
    public void Kendall_Ties_MatchesIndependentUnorderedPairOracle(int seed)
    {
        var rng = new System.Random(seed);
        for (var trial = 0; trial < 24; trial++)
        {
            var count = rng.Next(5, 14);
            var x = Enumerable.Range(0, count).Select(_ => (decimal)rng.Next(4)).ToArray();
            // Includes unilateral ties as well as joint ties.
            var y = Enumerable.Range(0, count).Select(i => trial % 2 == 0 ? (decimal)i : rng.Next(4)).ToArray();
            var comparableX = 0;
            var comparableY = 0;
            var concordance = 0;
            for (var i = 0; i < count; i++)
            {
                for (var j = i + 1; j < count; j++)
                {
                    var dx = decimal.Sign(x[i] - x[j]);
                    var dy = decimal.Sign(y[i] - y[j]);
                    comparableX += dx != 0 ? 1 : 0;
                    comparableY += dy != 0 ? 1 : 0;
                    concordance += dx * dy;
                }
            }

            var expected = comparableX * comparableY == 0
                ? 0.0 : concordance / Math.Sqrt(comparableX * comparableY);
            ((double)RankCorrelation<decimal>.KendallTauB(x, y)).Should().BeApproximately(expected, 1e-12);
        }

        RankCorrelation<decimal>.KendallTauB([1m, 2m, 2m, 3m, 4m], [1m, 2m, 3m, 3m, 5m])
            .Should().BeApproximately(8m / 9m, 0.000000000000000000000001m);
    }

    [Theory]
    [InlineData(101)]
    [InlineData(887)]
    public void WelfordMerge_UnequalPartitionsAndTrees_MatchBatchMoments(int seed)
    {
        var rng = new System.Random(seed);
        for (var trial = 0; trial < 12; trial++)
        {
            var values = Enumerable.Range(0, rng.Next(3, 15)).Select(_ => (decimal)rng.Next(-25, 26)).ToArray();
            var mean = values.Sum() / values.Length;
            var variance = values.Sum(v => (v - mean) * (v - mean)) / (values.Length - 1);
            for (var split = 1; split < values.Length; split++)
            {
                var left = Accumulate(values[..split]);
                var right = Accumulate(values[split..]);
                AssertMoments(WelfordMoments<decimal>.Combine(left, right), mean, variance, values.Length);
                AssertMoments(WelfordMoments<decimal>.Combine(right, left), mean, variance, values.Length);
                left.Count.Should().Be(split);
                right.Count.Should().Be(values.Length - split);
            }

            var forward = new WelfordMoments<decimal>();
            var backward = new WelfordMoments<decimal>();
            foreach (var value in values)
            {
                forward.Merge(Accumulate([value]));
                backward = WelfordMoments<decimal>.Combine(Accumulate([value]), backward);
            }

            AssertMoments(forward, mean, variance, values.Length);
            AssertMoments(backward, mean, variance, values.Length);
        }
    }

    private static double[] PairRanks(decimal[] values)
        => values.Select(v => 1.0 + values.Count(w => w < v) + (values.Count(w => w == v) - 1) / 2.0).ToArray();

    private static double BatchCorrelation(double[] x, double[] y)
    {
        var mx = x.Average();
        var my = y.Average();
        var xx = x.Sum(v => (v - mx) * (v - mx));
        var yy = y.Sum(v => (v - my) * (v - my));
        var xy = x.Zip(y).Sum(v => (v.First - mx) * (v.Second - my));
        return xx * yy == 0 ? 0 : xy / Math.Sqrt(xx * yy);
    }

    private static WelfordMoments<decimal> Accumulate(decimal[] values)
    {
        var moments = new WelfordMoments<decimal>();
        foreach (var value in values)
        {
            moments.Add(value);
        }

        return moments;
    }

    private static void AssertMoments(WelfordMoments<decimal> actual, decimal mean, decimal variance, int count)
    {
        actual.Count.Should().Be(count);
        actual.Mean.Should().BeApproximately(mean, 0.000000000000000000001m);
        actual.Variance.Should().BeApproximately(variance, 0.000000000000000000001m);
    }
}
