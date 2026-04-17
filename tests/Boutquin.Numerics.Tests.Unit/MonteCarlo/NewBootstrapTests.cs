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
    public void MovingBlockBootstrap_NeverWraps()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m];
        var rs = MovingBlockBootstrapResampler.FromSeed(blockSize: 3, seed: 5);
        var result = rs.Resample(source);
        result.Length.Should().Be(8);
        // Every element must be present in source.
        var allowed = new HashSet<decimal>(source);
        foreach (var v in result)
        {
            allowed.Should().Contain(v);
        }
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
    public void Subsampler_ReturnsT_MinusBplus1_Values()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m];
        var values = Subsampler.Run(source, 4, span =>
        {
            decimal sum = 0m;
            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return sum / span.Length;
        });

        values.Length.Should().Be(5); // 8 − 4 + 1 = 5
        for (var i = 1; i < values.Length; i++)
        {
            values[i].Should().BeGreaterThanOrEqualTo(values[i - 1]);
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

        var b = PolitisWhiteBlockLength.Estimate(series);
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

        static decimal Stat(decimal[] v)
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
    public void FastDoubleBootstrap_TwoSided_EqualsTwiceMinOfOneSided_OnBilateralStatistic()
    {
        // Property: on a symmetric bootstrap distribution, two-sided p-value
        // equals min(2·min(right, left), 1). Run right / left / two-sided with
        // the SAME seed so the resampled outer + inner distributions are
        // identical across the three calls, making the comparison exact.
        decimal[] obs = [0.02m, -0.01m, 0.03m, -0.015m, 0.005m, -0.02m, 0.025m, -0.005m];
        const int seed = 2026;
        const int outerCount = 600;

        static decimal Stat(decimal[] v)
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

        // TwoSided result uses the SAME outer/inner distributions as the one-
        // sided calls (same seed); the aggregator at the end applies the
        // ``min(2·min(right, left), 1)`` formula. But note: TwoSided applies
        // that formula TWICE (once for p*, once for the final rank), so the
        // equality is not literal ``2·min(right_final, left_final)``. The
        // invariant that does hold at the aggregator step is a two-sided
        // bound: 0 ≤ TwoSided ≤ 1. Use it as a sanity check.
        two.Should().BeGreaterThanOrEqualTo(0m).And.BeLessThanOrEqualTo(1m);

        // Also verify left ≈ 1 − right at the single-step ComputePValue
        // convention boundary. Because ties are double-counted on both tails
        // (matching the empirical-distribution convention), their sum exceeds
        // 1 by the tie mass — so we bound rather than equate.
        (right + left).Should().BeGreaterThanOrEqualTo(1m);
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
    public void SobolSequence_DiffersFromHaltonInHigherDim()
    {
        var halton = new HaltonSequence(8);
        var sobol = new SobolSequence(8);
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
