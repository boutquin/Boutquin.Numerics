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

public sealed class NewBootstrapTests
{
    [Fact]
    public void StationaryBootstrap_ProducesSameLengthAsInput()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m, 10m];
        var resampler = StationaryBootstrapResampler.FromSeed(meanBlockLength: 3.0, seed: 1);
        resampler.Resample(source).Length.Should().Be(source.Length);
    }

    [Fact]
    public void StationaryBootstrap_DeterministicWithSeed()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m];
        var a = StationaryBootstrapResampler.FromSeed(2.5, seed: 42).Resample(source);
        var b = StationaryBootstrapResampler.FromSeed(2.5, seed: 42).Resample(source);
        a.Should().Equal(b);
    }

    [Fact]
    public void StationaryBootstrap_RestartsOnlyBelowReciprocalMeanBlockLength()
    {
        decimal[] source = [0m, 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m];
        var random = new ScriptedStationarySource();
        var actual = new StationaryBootstrapResampler(3.0, random).Resample(source);

        actual.Take(5).Should().Equal(0m, 5m, 6m, 9m, 0m);
    }

    private sealed class ScriptedStationarySource : Boutquin.Numerics.Random.IRandomSource
    {
        private readonly int[] _indices = [0, 5, 9, 3];
        private readonly double[] _uniforms = [0.2, 0.4, 0.33, 0.34, 0.1];
        private int _indexPosition;
        private int _uniformPosition;

        public ulong NextULong() => throw new NotSupportedException();

        public int NextInt(int upperExclusive) => _indices[_indexPosition++ % _indices.Length];

        public double NextDouble() => _uniforms[_uniformPosition++ % _uniforms.Length];
    }

    [Fact]
    public void MovingBlockBootstrap_NeverWraps()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m];
        var rs = new MovingBlockBootstrapResampler<decimal>(blockSize: 3, new MaximumRandomSource());
        var result = rs.Resample(source);
        result.Length.Should().Be(8);

        for (var blockStart = 0; blockStart < result.Length; blockStart += 3)
        {
            var blockLength = Math.Min(3, result.Length - blockStart);
            for (var offset = 1; offset < blockLength; offset++)
            {
                result[blockStart + offset].Should().Be(
                    result[blockStart] + offset,
                    "every emitted moving block is a contiguous source slice and cannot wrap from the final element to the first");
            }
        }
    }

    private sealed class MaximumRandomSource : Boutquin.Numerics.Random.IRandomSource
    {
        public ulong NextULong() => ulong.MaxValue;
    }

    [Fact]
    public void WildBootstrap_RademacherPreservesAbsoluteValues()
    {
        decimal[] residuals = [0.01m, -0.02m, 0.005m, 0.04m, -0.01m];
        var rs = WildBootstrapResampler.FromSeed(WildBootstrapWeights.Rademacher, seed: 11);
        var result = rs.Resample(residuals);

        for (var i = 0; i < residuals.Length; i++)
        {
            Math.Abs(result[i]).Should().Be(Math.Abs(residuals[i]));
        }
    }

    [Fact]
    public void Subsampler_ReturnsSortedValues_FromNonMonotoneStatistic()
    {
        // A non-monotone source so the per-window statistic is genuinely out of
        // order before Subsampler.Run sorts it — the old test used a strictly
        // increasing source with a window-mean statistic, so the window values
        // arrived already ascending and Array.Sort was never exercised.
        decimal[] source = [5m, 1m, 9m, 2m, 8m, 3m, 7m, 4m];

        // Window mean for b = 3 over this source:
        //   [5,1,9]->5, [1,9,2]->4, [9,2,8]->19/3, [2,8,3]->13/3,
        //   [8,3,7]->6, [3,7,4]->14/3  -> raw order is NOT sorted.
        var values = Subsampler.Run(source, 3, span =>
        {
            decimal sum = 0m;
            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return sum / span.Length;
        });

        values.Length.Should().Be(6); // 8 − 3 + 1 = 6

        // Output must be ascending — proves Array.Sort actually reordered the
        // non-monotone window statistics.
        for (var i = 1; i < values.Length; i++)
        {
            values[i].Should().BeGreaterThanOrEqualTo(values[i - 1]);
        }

        // And the sorted set must be a permutation of the known window means.
        var expected = new[] { 5m, 4m, 19m / 3m, 13m / 3m, 6m, 14m / 3m }.OrderBy(x => x).ToArray();
        for (var i = 0; i < expected.Length; i++)
        {
            Math.Abs(values[i] - expected[i]).Should().BeLessThan(1e-9m);
        }
    }

    [Fact]
    public void SuggestSubsampleLength_SeriesLengthBelowTwo_ThrowsArgumentOutOfRangeException()
    {
        // M-3 guard: Run requires 1 <= subsampleLength < series.Length (Subsampler.cs
        // domain), so no Run-valid length exists for seriesLength < 2. Pre-fix,
        // SuggestSubsampleLength(1) returns 1, which Run then rejects (1 >= series.Length
        // of 1) — contradicting the documented "always a valid argument to Run" contract.
        var act = () => Subsampler.SuggestSubsampleLength(1);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("seriesLength");
    }

    [Fact]
    public void SuggestSubsampleLength_RoundTripsIntoRun_ForEverySeriesLengthInRange()
    {
        // AC round-trip property: for every seriesLength in [2, 64], Run must accept
        // SuggestSubsampleLength(seriesLength) without throwing — deterministic
        // iteration, no randomness.
        for (var seriesLength = 2; seriesLength <= 64; seriesLength++)
        {
            var series = new double[seriesLength];
            for (var i = 0; i < seriesLength; i++)
            {
                series[i] = i;
            }

            var suggested = Subsampler.SuggestSubsampleLength(seriesLength);

            var act = () => Subsampler.Run<double>(series, suggested, static span => span[0]);

            act.Should().NotThrow($"seriesLength={seriesLength}, suggested={suggested}");
        }
    }

    [Fact]
    public void PolitisWhite_ReturnsPositiveBlockLength()
    {
        // White noise — no autocorrelation, optimal block ~1.
        var rng = new System.Random(0);
        var series = new double[500];
        for (var i = 0; i < series.Length; i++)
        {
            series[i] = rng.NextDouble() * 2 - 1;
        }

        var b = PolitisWhiteBlockLength.EstimateDouble(series);
        b.Should().BeGreaterThan(0.0);
        b.Should().BeLessThan(series.Length / 2.0 + 1.0);
    }

    [Fact]
    public void FastDoubleBootstrap_ReturnsValidPValue()
    {
        decimal[] obs = [0.005m, 0.01m, -0.005m, 0.015m, 0.002m, -0.01m, 0.008m];
        var fdb = FastDoubleBootstrap.FromSeed(outerCount: 200, seed: 99);
        var p = fdb.PValue(obs, 0.5m, vals =>
        {
            decimal sum = 0m;
            for (var i = 0; i < vals.Length; i++)
            {
                sum += vals[i];
            }

            return sum / vals.Length;
        });

        p.Should().BeGreaterThanOrEqualTo(0m).And.BeLessThanOrEqualTo(1m);
    }

    [Fact]
    public void FastDoubleBootstrap_DefaultTailIsRightTail()
    {
        // A signed-mean statistic on a mean-zero sample puts the observed in
        // the upper tail; right-tail p-value should sit well below 0.5 on a
        // large-magnitude observed deviation.
        decimal[] obs = [0.5m, -0.3m, 0.2m, -0.1m, 0.4m, -0.2m, 0.6m, -0.1m];
        var fdb = FastDoubleBootstrap.FromSeed(outerCount: 500, seed: 101);

        static decimal Stat(ReadOnlySpan<decimal> v)
        {
            decimal sum = 0m;
            for (var i = 0; i < v.Length; i++)
            {
                sum += v[i];
            }

            return sum / v.Length;
        }

        var pDefault = fdb.PValue(obs, observed: 0.5m, Stat);
        var pRight = FastDoubleBootstrap.FromSeed(outerCount: 500, seed: 101)
            .PValue(obs, observed: 0.5m, Stat, PValueTail.RightTail);

        pDefault.Should().Be(pRight,
            "the default tail must be RightTail — backward-compatible with the previous shipped signature");
    }

    [Theory]
    [InlineData(PValueTail.RightTail)]
    [InlineData(PValueTail.LeftTail)]
    [InlineData(PValueTail.TwoSided)]
    public void FastDoubleBootstrap_AllTails_ReturnValidPValue(PValueTail tail)
    {
        decimal[] obs = [0.005m, 0.01m, -0.005m, 0.015m, 0.002m, -0.01m, 0.008m];
        var fdb = FastDoubleBootstrap.FromSeed(outerCount: 500, seed: 199);
        var p = fdb.PValue(obs, 0.0m, static v =>
        {
            decimal sum = 0m;
            for (var i = 0; i < v.Length; i++)
            {
                sum += v[i];
            }

            return sum / v.Length;
        }, tail);

        p.Should().BeGreaterThanOrEqualTo(0m).And.BeLessThanOrEqualTo(1m);
    }

    [Fact]
    public void FastDoubleBootstrap_TwoSided_IsDeterministicGoldenValue_UnderFixedSeed()
    {
        // With a fixed seed the outer/inner resample draws are reproducible, so
        // every FDB p-value is a deterministic number. Pin the actual values so any
        // change to the resampling, ranking, or tail logic is caught. The two-sided
        // value is the equal-tail fold min(2*min(right, left), 1) of the two
        // one-sided FDB p-values, each of which takes its own quantile of the inner
        // distribution (right: the (1 - p*)-quantile; left: the p*-quantile).
        decimal[] obs = [0.02m, -0.01m, 0.03m, -0.015m, 0.005m, -0.02m, 0.025m, -0.005m];
        const int seed = 2026;
        const int outerCount = 600;

        static decimal Stat(ReadOnlySpan<decimal> v)
        {
            decimal sum = 0m;
            for (var i = 0; i < v.Length; i++)
            {
                sum += v[i];
            }

            return sum / v.Length;
        }

        var right = FastDoubleBootstrap.FromSeed(outerCount, seed).PValue(obs, 0.01m, Stat, PValueTail.RightTail);
        var left = FastDoubleBootstrap.FromSeed(outerCount, seed).PValue(obs, 0.01m, Stat, PValueTail.LeftTail);
        var two = FastDoubleBootstrap.FromSeed(outerCount, seed).PValue(obs, 0.01m, Stat, PValueTail.TwoSided);

        // Exact rational sample means over the PCG index tape give strict first-level
        // counts right93/left492 and final counts right34/left538 out of600.
        right.Should().Be(34m / 600m);
        left.Should().Be(538m / 600m);
        two.Should().Be(Math.Min(2m * Math.Min(right, left), 1m),
            "the two-sided FDB p-value is the equal-tail fold of the one-sided FDB p-values");
    }

    [Fact]
    public void HarrellDavisPercentile_MedianApproachesSampleMedian()
    {
        decimal[] sorted = [1m, 2m, 3m, 4m, 5m];
        var hd50 = HarrellDavisPercentile.Compute(sorted, 0.50);
        // For n=5, HD median = weighted average of the order stats — should be close to 3.
        Math.Abs(hd50 - 3m).Should().BeLessThan(0.5m);
    }

    [Fact]
    public void HarrellDavisPercentile_NaNPercentile_ThrowsArgumentOutOfRangeException()
    {
        // M-3 guard: a NaN percentile must be rejected with the same exception type
        // PSquareEstimator's sibling percentile-domain guard uses (ArgumentOutOfRangeException).
        // Pre-guard, NaN fails both the p<=0 and p>=1 boundary checks (NaN comparisons are
        // always false), falls through to the open-interval Beta-weight computation, and
        // the weightSum<=0 fallback check ALSO silently passes on NaN — so Compute()
        // returns a NaN result instead of throwing.
        double[] sorted = [1.0, 2.0, 3.0, 4.0, 5.0];

        var act = () => HarrellDavisPercentile.Compute<double>(sorted, double.NaN);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("p");
    }

    [Fact]
    public void HaltonSequence_ProducesUnitIntervalCoordinates()
    {
        var halton = new HaltonSequence(dimension: 3);
        for (var i = 0; i < 100; i++)
        {
            var pt = halton.Next();
            pt.Length.Should().Be(3);
            foreach (var c in pt)
            {
                c.Should().BeGreaterThanOrEqualTo(0.0).And.BeLessThan(1.0);
            }
        }
    }

    [Fact]
    public void SobolSequence_ProducesUnitIntervalCoordinates()
    {
        var sobol = new SobolSequence(dimension: 4);
        for (var i = 0; i < 100; i++)
        {
            var pt = sobol.Next();
            pt.Length.Should().Be(4);
            foreach (var c in pt)
            {
                c.Should().BeGreaterThanOrEqualTo(0.0).And.BeLessThan(1.0);
            }
        }
    }

    [Fact]
    public void SobolSequence_HalfCoordinatesStayBelowOne()
    {
        var sobol = new SobolSequence<Half>(dimension: 32);
        for (var i = 0; i < 4096; i++)
        {
            foreach (var coordinate in sobol.Next())
            {
                Assert.True(coordinate >= Half.Zero && coordinate < Half.One,
                    $"Sobol Half point {i} produced {coordinate} outside [0, 1).");
            }
        }
    }

    [Fact]
    public void SobolSequence_DiffersFromHaltonInHigherDim()
    {
        var halton = new HaltonSequence(8);
        var sobol = new SobolSequence(8, skip: 1);
        var differences = 0;
        for (var i = 0; i < 50; i++)
        {
            var h = halton.Next();
            var s = sobol.Next();
            for (var d = 0; d < 8; d++)
            {
                if (Math.Abs(h[d] - s[d]) > 1e-3)
                {
                    differences++;
                    break;
                }
            }
        }

        differences.Should().BeGreaterThan(45, "different QMC sequences should produce mostly different points");
    }
}
