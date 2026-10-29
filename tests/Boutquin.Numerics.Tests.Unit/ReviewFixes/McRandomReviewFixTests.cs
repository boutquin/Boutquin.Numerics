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

using Boutquin.Numerics.Collections;
using Boutquin.Numerics.Distributions;
using Boutquin.Numerics.Internal;
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Coverage and behavioral regression tests for the MonteCarlo / Random /
/// Distributions / Collections / Internal review-fix batch (S5, S6, S17, B1,
/// B9–B14, B22, B23, B25, B26, B27, B28).
/// </summary>
public sealed class McRandomReviewFixTests
{
    private static decimal Mean(ReadOnlySpan<decimal> v)
    {
        decimal sum = 0m;
        for (var i = 0; i < v.Length; i++)
        {
            sum += v[i];
        }

        return sum / v.Length;
    }

    // ===================================================================
    // S5 — PSquareEstimator tie convention at the top marker (q[4]).
    // ===================================================================

    [Fact]
    public void PSquare_TieAtTopMarker_CountsCorrectly_AndStaysWithinObservedRange()
    {
        // Feed five distinct values to initialize markers, then feed the current
        // maximum repeatedly. With the corrected strict-interior / >= max-extension
        // convention the estimator must remain finite, in-range, and monotone-safe;
        // the tie at q[4] is handled by the max-extension branch (idempotent height
        // update) rather than mis-counted as interior.
        var est = new PSquareEstimator<double>(0.95);
        double[] init = [1.0, 2.0, 3.0, 4.0, 5.0];
        foreach (var x in init)
        {
            est.Add(x);
        }

        // 5.0 is the current top marker; repeatedly tying it must not throw or
        // push the estimate outside the observed data range.
        for (var i = 0; i < 50; i++)
        {
            est.Add(5.0);
        }

        est.Count.Should().Be(55);
        var e = est.Estimate;
        double.IsFinite(e).Should().BeTrue();
        e.Should().BeGreaterThanOrEqualTo(1.0).And.BeLessThanOrEqualTo(5.0);
    }

    [Fact]
    public void PSquare_GoldenVector_TracksExactQuantileWithinAccuracyBand()
    {
        // Golden-vector test: feed a fixed, shuffled 0..999 dataset and compare the
        // P² estimate of the 90th percentile to the exact order-statistic quantile.
        // P²'s documented accuracy is ~1% on stationary streams after a few thousand
        // observations; on a uniform 0..999 ramp the 90th percentile is ~899.1.
        const double p = 0.90;
        var data = new double[1000];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = i;
        }

        // Deterministic Fisher-Yates shuffle so the stream order is non-sorted but
        // reproducible (P² is order-sensitive). NextInt is the IRandomSource DIM.
        IRandomSource rng = new Pcg64RandomSource(20260628UL);
        for (var i = data.Length - 1; i > 0; i--)
        {
            var j = rng.NextInt(i + 1);
            (data[i], data[j]) = (data[j], data[i]);
        }

        var est = new PSquareEstimator<double>(p);
        foreach (var x in data)
        {
            est.Add(x);
        }

        // Exact quantile via the linear-interpolation order statistic on the sorted
        // reference (computed independently of P²).
        var sorted = (double[])data.Clone();
        Array.Sort(sorted);
        var rank = p * (sorted.Length - 1);
        var lo = (int)Math.Floor(rank);
        var hi = (int)Math.Ceiling(rank);
        var exact = lo == hi ? sorted[lo] : sorted[lo] + (rank - lo) * (sorted[hi] - sorted[lo]);

        // Within ~2.5% of the exact 90th percentile (range is 0..999, so band ≈ 25).
        Math.Abs(est.Estimate - exact).Should().BeLessThan(25.0);
    }

    // ===================================================================
    // S6 — PolitisWhiteBlockLength no-convergence fallback.
    // ===================================================================

    [Fact]
    public void PolitisWhite_NoConvergence_FallsBackToExplicitMaxLag()
    {
        // A perfectly periodic series keeps |rho| large at every lag, so no window
        // of k_n consecutive autocorrelations ever falls below the threshold — the
        // not-found branch. The documented fallback is the maximum lag k_max, then
        // doubled and clamped back to k_max, then run through the optimal-length
        // formula. The block length must be a finite, positive, in-range value and
        // must hit the high end of the [1, T/2] band (reflecting the max-lag
        // fallback), NOT collapse to ~1 the way a converged white-noise series does.
        const int t = 256;
        var series = new double[t];
        for (var i = 0; i < t; i++)
        {
            // Strong deterministic periodicity (period 4) → persistent autocorrelation.
            series[i] = Math.Sin(2.0 * Math.PI * i / 4.0);
        }

        var b = PolitisWhiteBlockLength.EstimateDouble(series);
        double.IsFinite(b).Should().BeTrue();
        b.Should().BeGreaterThan(1.0, "the no-convergence fallback uses the maximum lag, not the ~1 white-noise result");
        b.Should().BeLessThanOrEqualTo(t / 2.0);
    }

    [Fact]
    public void PolitisWhite_IsScaleInvariant_UnderConstantRescaling()
    {
        // The Politis-White (2004) optimal block length b̂ = (2·Ĝ²/D̂·T)^(1/3) is
        // mathematically scale-invariant: both Ĝ = Σ|k|·w(k/M)·γ̂(k) and √D̂ =
        // Σw(k/M)·γ̂(k) are built from autocovariances γ̂(k) = ρ̂(k)·variance, so
        // each scales linearly with variance and the ratio Ĝ²/D̂ — hence b̂ — is
        // unchanged when the series is multiplied by a constant c. Rescaling the
        // series by a pure constant must NOT move the estimate.
        const int t = 256;
        var series = new double[t];
        var prev = 0.0;
        var rng = new System.Random(20260629);
        for (var i = 0; i < t; i++)
        {
            // AR(1) with φ = 0.5 → genuine, non-trivial autocorrelation.
            prev = 0.5 * prev + (rng.NextDouble() * 2.0 - 1.0);
            series[i] = prev;
        }

        var baseEstimate = PolitisWhiteBlockLength.EstimateDouble(series);

        foreach (var c in new[] { 10.0, 100.0, 1000.0, 0.1, 0.01 })
        {
            var scaled = new double[t];
            for (var i = 0; i < t; i++)
            {
                scaled[i] = series[i] * c;
            }

            var scaledEstimate = PolitisWhiteBlockLength.EstimateDouble(scaled);
            scaledEstimate.Should().BeApproximately(
                baseEstimate, 1e-6,
                $"the optimal block length is scale-invariant; rescaling by {c} must not change it");
        }
    }

    // ===================================================================
    // S17 / B1 — BootstrapMonteCarloEngine.RunMulti coverage + defensive copies.
    // ===================================================================

    [Fact]
    public void RunMulti_GenericGuards_Throw()
    {
        var engine = BootstrapMonteCarloEngine<decimal>.FromSeed(simulationCount: 10, seed: 1);
        decimal[] obs = [1m, 2m, 3m];

        FluentActions.Invoking(() => engine.RunMulti(ReadOnlySpan<decimal>.Empty, 2, (s, o) => { }))
            .Should().Throw<ArgumentException>("empty observations are rejected");
        FluentActions.Invoking(() => engine.RunMulti(obs, 0, (s, o) => { }))
            .Should().Throw<ArgumentOutOfRangeException>("statisticCount must be positive");
        FluentActions.Invoking(() => engine.RunMulti(obs, 2, null!))
            .Should().Throw<ArgumentNullException>("the writer must not be null");
        FluentActions.Invoking(() => engine.RunMulti(obs, 2, (s, o) => { }, names: new[] { "only_one" }))
            .Should().Throw<ArgumentException>("names length must match statisticCount");
    }

    [Fact]
    public void RunMulti_ComputesPerStatisticSortedDistributionsAndSummaries()
    {
        // Two statistics: [0] = mean, [1] = max of the resampled path. The returned
        // distributions must be sorted ascending, per-statistic, and the summaries
        // (mean, median, p5, p95) must be self-consistent with those distributions.
        var engine = BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 500, seed: 7);
        double[] obs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        var result = engine.RunMulti(obs, 2, (path, outp) =>
        {
            double sum = 0.0, max = double.NegativeInfinity;
            for (var i = 0; i < path.Length; i++)
            {
                sum += path[i];
                if (path[i] > max)
                {
                    max = path[i];
                }
            }

            outp[0] = sum / path.Length;
            outp[1] = max;
        });

        result.SimulationCount.Should().Be(500);
        result.Names.Should().Equal("stat_0", "stat_1");
        result.Statistics.Count.Should().Be(2);

        for (var k = 0; k < 2; k++)
        {
            var dist = result.Statistics[k];
            dist.Length.Should().Be(500);
            for (var i = 1; i < dist.Length; i++)
            {
                dist[i].Should().BeGreaterThanOrEqualTo(dist[i - 1], "each per-statistic distribution is sorted ascending");
            }

            // Median must equal the Percentile.Compute of the same distribution.
            result.Medians[k].Should().Be(Percentile.Compute(dist, 0.50));
            result.Percentile5s[k].Should().Be(Percentile.Compute(dist, 0.05));
            result.Percentile95s[k].Should().Be(Percentile.Compute(dist, 0.95));
        }

        // Statistic 1 is the path maximum; every value must lie within the input range.
        foreach (var v in result.Statistics[1])
        {
            v.Should().BeGreaterThanOrEqualTo(1.0).And.BeLessThanOrEqualTo(8.0);
        }
    }

    [Fact]
    public void RunMulti_CustomNames_AreUsed()
    {
        var engine = BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 20, seed: 3);
        double[] obs = [1.0, 2.0, 3.0];
        var result = engine.RunMulti(obs, 2, (p, o) => { o[0] = p[0]; o[1] = p[1]; }, names: new[] { "sharpe", "sortino" });
        result.Names.Should().Equal("sharpe", "sortino");
    }

    [Fact]
    public void RunMulti_ReturnsDefensiveCopies_MutationDoesNotAffectNextCall()
    {
        // S17: the returned inner arrays must be defensive copies. Mutating a
        // returned distribution must not corrupt a fresh run with the same seed.
        double[] obs = [1.0, 2.0, 3.0, 4.0, 5.0];

        var first = BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 100, seed: 42)
            .RunMulti(obs, 1, (p, o) => o[0] = p[0]);

        // Vandalize the returned inner array.
        var inner = first.Statistics[0];
        for (var i = 0; i < inner.Length; i++)
        {
            inner[i] = double.NaN;
        }

        // A second identical run must be completely unaffected by the mutation above.
        var second = BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 100, seed: 42)
            .RunMulti(obs, 1, (p, o) => o[0] = p[0]);

        second.Statistics[0].Should().NotContain(double.NaN);
        second.Statistics[0].Should().Equal(
            // Recompute the expected pristine distribution a third time, same seed.
            BootstrapMonteCarloEngine<double>.FromSeed(simulationCount: 100, seed: 42)
                .RunMulti(obs, 1, (p, o) => o[0] = p[0]).Statistics[0]);
    }

    [Fact]
    public void RunMulti_NonGenericFacade_Works()
    {
        var engine = BootstrapMonteCarloEngine.FromSeed(simulationCount: 50, seed: 9);
        decimal[] obs = [1m, 2m, 3m, 4m];
        var result = engine.RunMulti(obs, 2, (p, o) => { o[0] = Mean(p); o[1] = p[0]; });
        result.SimulationCount.Should().Be(50);
        result.Statistics.Count.Should().Be(2);
        result.Statistics[0].Length.Should().Be(50);
    }

    // ===================================================================
    // B9 — FastDoubleBootstrap guards.
    // ===================================================================

    [Fact]
    public void FastDoubleBootstrap_Guards()
    {
        FluentActions.Invoking(() => FastDoubleBootstrap.FromSeed(outerCount: 49, seed: 1))
            .Should().Throw<ArgumentOutOfRangeException>("outerCount < 50 is rejected");

        var fdb = FastDoubleBootstrap.FromSeed(outerCount: 50, seed: 1);
        FluentActions.Invoking(() => fdb.PValue(ReadOnlySpan<decimal>.Empty, 0m, Mean))
            .Should().Throw<ArgumentException>("empty observations are rejected");
        FluentActions.Invoking(() => fdb.PValue(new decimal[] { 1m, 2m }, 0m, null!))
            .Should().Throw<ArgumentNullException>("null statistic is rejected");
        FluentActions.Invoking(() => new FastDoubleBootstrap(50, null!))
            .Should().Throw<ArgumentNullException>("null random source is rejected");
    }

    // ===================================================================
    // B10 — MovingBlockBootstrapResampler guards.
    // ===================================================================

    [Fact]
    public void MovingBlockBootstrap_Guards()
    {
        FluentActions.Invoking(() => new MovingBlockBootstrapResampler<double>(0, new Pcg64RandomSource(1UL)))
            .Should().Throw<ArgumentOutOfRangeException>("blockSize <= 0 is rejected");
        FluentActions.Invoking(() => new MovingBlockBootstrapResampler<double>(3, null!))
            .Should().Throw<ArgumentNullException>("null RNG is rejected");

        var rs = MovingBlockBootstrapResampler<double>.FromSeed(blockSize: 5, seed: 1);
        FluentActions.Invoking(() => rs.Resample(new double[] { 1.0, 2.0, 3.0 }))
            .Should().Throw<ArgumentException>("source shorter than block size is rejected");
    }

    // ===================================================================
    // B11 — StationaryBootstrapResampler guards + ResamplePaired.
    // ===================================================================

    [Fact]
    public void StationaryBootstrap_Guards()
    {
        FluentActions.Invoking(() => new StationaryBootstrapResampler<double>(0.5, new Pcg64RandomSource(1UL)))
            .Should().Throw<ArgumentOutOfRangeException>("meanBlockLength < 1 is rejected");
        FluentActions.Invoking(() => new StationaryBootstrapResampler<double>(double.NaN, new Pcg64RandomSource(1UL)))
            .Should().Throw<ArgumentOutOfRangeException>("NaN meanBlockLength is rejected");

        var rs = StationaryBootstrapResampler<double>.FromSeed(2.0, seed: 1);
        FluentActions.Invoking(() => rs.Resample(ReadOnlySpan<double>.Empty))
            .Should().Throw<ArgumentException>("empty source is rejected");
    }

    [Fact]
    public void StationaryBootstrap_RejectsNonFiniteMeanBlockLength()
    {
        // +∞ meanBlockLength yields _restartProbability = 1/∞ = 0, so the resampler
        // never restarts and degenerates into a deterministic circular permutation —
        // the stochastic contract is silently destroyed. A non-finite mean block
        // length must be rejected at the constructor, exactly like NaN and < 1.
        FluentActions.Invoking(() => new StationaryBootstrapResampler<double>(double.PositiveInfinity, new Pcg64RandomSource(1UL)))
            .Should().Throw<ArgumentOutOfRangeException>("+∞ meanBlockLength is rejected");
        FluentActions.Invoking(() => new StationaryBootstrapResampler<double>(double.NegativeInfinity, new Pcg64RandomSource(1UL)))
            .Should().Throw<ArgumentOutOfRangeException>("−∞ meanBlockLength is rejected");

        // The non-generic decimal facade routes through the same guard.
        FluentActions.Invoking(() => new StationaryBootstrapResampler(double.PositiveInfinity, new Pcg64RandomSource(1UL)))
            .Should().Throw<ArgumentOutOfRangeException>("+∞ meanBlockLength is rejected on the facade too");
    }

    [Fact]
    public void StationaryBootstrap_ResamplePaired_PreservesPairingAndLength()
    {
        var rs = StationaryBootstrapResampler<double>.FromSeed(3.0, seed: 17);
        double[] a = [10.0, 20.0, 30.0, 40.0, 50.0];
        double[] b = [-1.0, -2.0, -3.0, -4.0, -5.0];

        var (ra, rb) = rs.ResamplePaired(a, b);
        ra.Length.Should().Be(a.Length);
        rb.Length.Should().Be(b.Length);

        // Pairing invariant: a[i] = 10*k  <=>  b[i] = -k, so ra[t] = -10*rb[t].
        for (var t = 0; t < ra.Length; t++)
        {
            ra[t].Should().Be(-10.0 * rb[t]);
        }
    }

    [Fact]
    public void StationaryBootstrap_ResamplePaired_RejectsMismatchedAndEmpty()
    {
        var rs = StationaryBootstrapResampler<double>.FromSeed(2.0, seed: 1);
        FluentActions.Invoking(() => rs.ResamplePaired(new double[] { 1.0, 2.0 }, new double[] { 1.0 }))
            .Should().Throw<ArgumentException>("mismatched lengths are rejected");
        FluentActions.Invoking(() => rs.ResamplePaired(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty))
            .Should().Throw<ArgumentException>("empty paired series are rejected");
    }

    // ===================================================================
    // B12 — Subsampler.Run guards + SuggestSubsampleLength.
    // ===================================================================

    [Fact]
    public void Subsampler_Run_Guards()
    {
        double[] series = [1.0, 2.0, 3.0, 4.0];
        FluentActions.Invoking(() => Subsampler.Run(series, 0, s => s[0]))
            .Should().Throw<ArgumentOutOfRangeException>("subsampleLength < 1 is rejected");
        FluentActions.Invoking(() => Subsampler.Run(series, 4, s => s[0]))
            .Should().Throw<ArgumentOutOfRangeException>("subsampleLength >= series length is rejected");
        FluentActions.Invoking(() => Subsampler.Run(series, 2, null!))
            .Should().Throw<ArgumentNullException>("null statistic is rejected");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    public void Subsampler_SuggestSubsampleLength_IsAlwaysUsableByRun(int seriesLength)
    {
        var b = Subsampler.SuggestSubsampleLength(seriesLength);
        // Polish: the suggestion must always satisfy Run's 1 <= b < seriesLength.
        b.Should().BeGreaterThanOrEqualTo(1);
        b.Should().BeLessThan(seriesLength);

        // And it must actually be accepted by Run without throwing.
        var series = new double[seriesLength];
        for (var i = 0; i < seriesLength; i++)
        {
            series[i] = i;
        }

        FluentActions.Invoking(() => Subsampler.Run<double>(series, b, s => s[0])).Should().NotThrow();
    }

    // ===================================================================
    // B13 — PolitisWhiteBlockLength.EstimateDouble guard.
    // ===================================================================

    [Fact]
    public void PolitisWhite_RejectsTooShortSeries()
    {
        FluentActions.Invoking(() => PolitisWhiteBlockLength.EstimateDouble(new double[31]))
            .Should().Throw<ArgumentException>("fewer than 32 observations is rejected");
        // 32 is the minimum accepted length.
        FluentActions.Invoking(() => PolitisWhiteBlockLength.EstimateDouble(new double[32]))
            .Should().NotThrow();
    }

    [Fact]
    public void PolitisWhite_ZeroVarianceSeries_ReturnsOne()
    {
        var flat = new double[64]; // all zero → zero variance.
        PolitisWhiteBlockLength.EstimateDouble(flat).Should().Be(1.0);
    }

    // ===================================================================
    // B14 — HarrellDavisPercentile boundaries + reference value.
    // ===================================================================

    [Fact]
    public void HarrellDavis_Boundaries()
    {
        HarrellDavisPercentile.Compute<double>(ReadOnlySpan<double>.Empty, 0.5).Should().Be(0.0);
        HarrellDavisPercentile.Compute<double>(new double[] { 42.0 }, 0.5).Should().Be(42.0);

        double[] sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        HarrellDavisPercentile.Compute<double>(sorted, 0.0).Should().Be(1.0);
        HarrellDavisPercentile.Compute<double>(sorted, 1.0).Should().Be(5.0);
        FluentActions.Invoking(() => HarrellDavisPercentile.Compute<double>(sorted, -0.5))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName("p");
        FluentActions.Invoking(() => HarrellDavisPercentile.Compute<double>(sorted, 1.5))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName("p");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void HarrellDavis_RejectsNonFinitePercentileBeforeLengthShortcuts(double p)
    {
        FluentActions.Invoking(() => HarrellDavisPercentile.Compute<double>([], p))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(p));
        FluentActions.Invoking(() => HarrellDavisPercentile.Compute<double>([42.0], p))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(p));
    }

    [Fact]
    public void HarrellDavis_RejectsUnsortedInput()
    {
        double[] unsorted = [3.0, 1.0, 2.0];

        FluentActions.Invoking(() => HarrellDavisPercentile.Compute<double>(unsorted, 0.5))
            .Should().Throw<ArgumentException>().WithParameterName("sorted");
    }

    [Fact]
    public void HarrellDavis_FivePointSample_MatchesIndependentBetaWeights()
    {
        // The median of {1,2,3,4,5} is exactly 3 by beta-weight symmetry.
        // For q25, independently integrate the Beta(3/2,9/2) density
        // 256/(7*pi) * sqrt(u) * (1-u)^(7/2) over the five intervals of width 1/5,
        // then weight those masses by the observations 1..5.
        double[] sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        HarrellDavisPercentile.Compute<double>(sorted, 0.50).Should().BeApproximately(3.0, 1e-9);

        var q25 = HarrellDavisPercentile.Compute<double>(sorted, 0.25);
        // mpmath 80-digit direct quadrature gives 1.769924414314333208036562054656...
        // scipy.special.betainc and scipy.stats.mstats.hdquantiles independently agree.
        q25.Should().BeApproximately(1.769924414314333208, 1e-12);
        q25.Should().BeLessThan(2.0, "HD q25 sits below the linear-interpolation quantile here");
        q25.Should().BeGreaterThan(1.0);
    }

    // ===================================================================
    // B22 — RNG reference vectors (independent ports of the published algorithms).
    // ===================================================================

    [Fact]
    public void Pcg64_MatchesIndependentReferencePort()
    {
        // Independent re-implementation of PCG-XSL-RR 128/64 (O'Neill 2014) from the
        // published spec: state=0; advance; state += seed; advance; then each output
        // advances the LCG and applies the XSL-RR permutation to the advanced state
        // (pcg-c step-then-output). Constants are the
        // pcg-cpp default multiplier/increment. If the library and this independent
        // port agree on the first outputs, the seeding + permutation are correct.
        const ulong seed = 0xDEADBEEFCAFEF00DUL;
        var lib = new Pcg64RandomSource(seed);
        var reference = new PcgReference(seed);

        for (var i = 0; i < 32; i++)
        {
            lib.NextULong().Should().Be(reference.Next(), $"PCG output {i} must match the independent reference port");
        }
    }

    [Fact]
    public void Xoshiro256_MatchesIndependentReferencePort()
    {
        // Independent re-implementation of SplitMix64 seed expansion + xoshiro256**
        // (Blackman-Vigna 2018) from the published spec.
        const ulong seed = 0x0123456789ABCDEFUL;
        var lib = new Xoshiro256StarStarRandomSource(seed);
        var reference = new XoshiroReference(seed);

        for (var i = 0; i < 32; i++)
        {
            lib.NextULong().Should().Be(reference.Next(), $"xoshiro256** output {i} must match the independent reference port");
        }
    }

    // ===================================================================
    // B23 — Sobol / Halton sequences: guards, determinism, skip, span, generic.
    // ===================================================================

    [Fact]
    public void Sobol_Guards()
    {
        FluentActions.Invoking(() => new SobolSequence(0)).Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new SobolSequence(33)).Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new SobolSequence(2, skip: -1)).Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new SobolSequence(2, skip: 1L << 30)).Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Sobol_DeterministicAndSkipConsistent()
    {
        var a = new SobolSequence(3);
        var b = new SobolSequence(3);
        for (var i = 0; i < 50; i++)
        {
            a.Next().Should().Equal(b.Next(), "same config → same points");
        }

        // skip = N must align the sequence with N leading Next() calls.
        var plain = new SobolSequence(3);
        for (var i = 0; i < 64; i++)
        {
            plain.Next();
        }

        var skipped = new SobolSequence(3, skip: 64);
        for (var i = 0; i < 10; i++)
        {
            skipped.Next().Should().Equal(plain.Next(), "skip must fast-forward to the same state as repeated Next()");
        }
    }

    [Fact]
    public void Sobol_SpanOverload_MatchesAllocatingOverload_AndRejectsShortSpan()
    {
        var a = new SobolSequence(4);
        var b = new SobolSequence(4);
        Span<double> buf = stackalloc double[4];
        for (var i = 0; i < 20; i++)
        {
            var alloc = a.Next();
            b.Next(buf);
            for (var d = 0; d < 4; d++)
            {
                buf[d].Should().Be(alloc[d]);
            }
        }

        var c = new SobolSequence(4);
        var tooShort = new double[3];
        FluentActions.Invoking(() => c.Next(tooShort)).Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Halton_Guards_GenericAndSpan()
    {
        FluentActions.Invoking(() => new HaltonSequence<double>(0)).Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new HaltonSequence<double>(51)).Should().Throw<ArgumentOutOfRangeException>();
        FluentActions.Invoking(() => new HaltonSequence<double>(2, skip: -1)).Should().Throw<ArgumentOutOfRangeException>();

        // Generic HaltonSequence<T> produces valid unit-interval coordinates and is
        // deterministic for a fixed config.
        var a = new HaltonSequence<double>(3);
        var b = new HaltonSequence<double>(3);
        for (var i = 0; i < 30; i++)
        {
            var pa = a.Next();
            var pb = b.Next();
            pa.Should().Equal(pb);
            foreach (var c in pa)
            {
                c.Should().BeGreaterThanOrEqualTo(0.0).And.BeLessThan(1.0);
            }
        }

        // Span overload must reject a short destination.
        var h = new HaltonSequence<double>(3);
        FluentActions.Invoking(() => h.Next(new double[2])).Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Halton_SkipConsistent()
    {
        var plain = new HaltonSequence<double>(3);
        for (var i = 0; i < 64; i++)
        {
            plain.Next();
        }

        var skipped = new HaltonSequence<double>(3, skip: 64);
        for (var i = 0; i < 10; i++)
        {
            skipped.Next().Should().Equal(plain.Next());
        }
    }

    // ===================================================================
    // B25 — T = float / T = Half parity / smoke tests.
    // ===================================================================

    [Fact]
    public void Percentile_Float_MatchesDoubleWithinFloatPrecision()
    {
        float[] sortedF = [1f, 2f, 3f, 4f, 5f];
        double[] sortedD = [1.0, 2.0, 3.0, 4.0, 5.0];
        var pf = Percentile.Compute<float>(sortedF, 0.5);
        var pd = Percentile.Compute<double>(sortedD, 0.5);
        ((double)pf).Should().BeApproximately(pd, 1e-5);
    }

    [Fact]
    public void NormalDistribution_Float_AndHalf_Smoke()
    {
        // Pdf(0) = 1/sqrt(2π) ≈ 0.39894 in any precision.
        var pf = NormalDistribution<float>.Pdf(0f);
        ((double)pf).Should().BeApproximately(0.39894, 1e-3);

        var ph = NormalDistribution<Half>.Pdf((Half)0);
        ((double)(float)ph).Should().BeApproximately(0.39894, 1e-2);

        // CDF(0) = 0.5 in any precision.
        ((double)CumulativeNormal<float>.Evaluate(0f)).Should().BeApproximately(0.5, 1e-5);
    }

    // ===================================================================
    // B26 — RollingWindow<T> edge cases.
    // ===================================================================

    [Fact]
    public void RollingWindow_CapacityOne_KeepsOnlyMostRecent()
    {
        var w = new RollingWindow<int>(1);
        w.Count.Should().Be(0);
        w.ToArray().Should().BeEmpty();

        w.Add(10);
        w.IsFull.Should().BeTrue();
        w.Count.Should().Be(1);
        w[0].Should().Be(10);

        w.Add(20);
        w.Count.Should().Be(1);
        w[0].Should().Be(20, "capacity-1 window (mod-1 index) keeps only the most recent element");

        w.Add(30);
        w[0].Should().Be(30);
        w.ToArray().Should().Equal(30);
    }

    [Fact]
    public void RollingWindow_EmptyEnumerationAndToArray()
    {
        var w = new RollingWindow<double>(4);
        w.ToArray().Should().BeEmpty();
        ((IEnumerable<double>)w).Should().BeEmpty();

        // Wrap-around then ToArray returns chronological order.
        for (var i = 1; i <= 6; i++)
        {
            w.Add(i);
        }

        w.ToArray().Should().Equal(3.0, 4.0, 5.0, 6.0);
        w.Should().Equal(3.0, 4.0, 5.0, 6.0);
    }

    // ===================================================================
    // B27 — NumericPrecision<T>.Sqrt contracts.
    // ===================================================================

    [Fact]
    public void NumericPrecision_Sqrt_NegativeContract()
    {
        // IEEE types: sqrt of negative is NaN (hardware semantics).
        double.IsNaN(NumericPrecision<double>.Sqrt(-1.0)).Should().BeTrue();
        float.IsNaN(NumericPrecision<float>.Sqrt(-1f)).Should().BeTrue();

        // decimal: negative throws (no NaN representation).
        FluentActions.Invoking(() => NumericPrecision<decimal>.Sqrt(-1m))
            .Should().Throw<ArgumentException>();
    }

    [Fact]
    public void NumericPrecision_Sqrt_LargeDecimal_NewtonConverges()
    {
        // > 100 million: Newton-Raphson decimal sqrt must converge to full precision.
        const decimal value = 123_456_789m;
        var r = NumericPrecision<decimal>.Sqrt(value);
        var err = Math.Abs(r * r - value);
        err.Should().BeLessThan(1e-6m);

        // A perfect square must round-trip exactly (123456² = 15241383936).
        NumericPrecision<decimal>.Sqrt(15_241_383_936m).Should().Be(123_456m);
    }

    [Fact]
    public void NumericPrecision_Sqrt_IeeeMatchesNativeExactly()
    {
        NumericPrecision<double>.Sqrt(2.0).Should().Be(Math.Sqrt(2.0));
        NumericPrecision<float>.Sqrt(2f).Should().Be(MathF.Sqrt(2f));
        NumericPrecision<Half>.Sqrt((Half)4).Should().Be((Half)2);
    }

    // ===================================================================
    // B28 — CumulativeNormal<T> direct unit tests incl. IEEE edge behavior.
    // ===================================================================

    [Theory]
    [InlineData(0.0, 0.5)]
    [InlineData(1.0, 0.8413447460685429)]
    [InlineData(-1.0, 0.15865525393145707)]
    [InlineData(1.96, 0.9750021048517795)]
    [InlineData(-1.96, 0.024997895148220484)]
    public void CumulativeNormal_KnownValues(double x, double expected)
    {
        CumulativeNormal<double>.Evaluate(x).Should().BeApproximately(expected, 1e-9);
    }

    [Fact]
    public void CumulativeNormal_IeeeEdgeBehavior()
    {
        CumulativeNormal<double>.Evaluate(double.PositiveInfinity).Should().Be(1.0);
        CumulativeNormal<double>.Evaluate(double.NegativeInfinity).Should().Be(0.0);
        double.IsNaN(CumulativeNormal<double>.Evaluate(double.NaN)).Should().BeTrue();

        // Symmetry: N(x) + N(-x) = 1.
        for (var x = 0.1; x < 4.0; x += 0.37)
        {
            (CumulativeNormal<double>.Evaluate(x) + CumulativeNormal<double>.Evaluate(-x))
                .Should().BeApproximately(1.0, 1e-12);
        }
    }

    // ===================================================================
    // Independent reference ports for B22.
    // ===================================================================

    /// <summary>
    /// Independent PCG-XSL-RR 128/64 reference, ported from O'Neill (2014) using
    /// the pcg-cpp default multiplier and increment. Deliberately written from the
    /// spec rather than reused from the library so a divergence is meaningful.
    /// </summary>
    private sealed class PcgReference
    {
        private static readonly UInt128 s_multiplier =
            new(2549297995355413924UL, 4865540595714422341UL);
        private static readonly UInt128 s_increment =
            new(6364136223846793005UL, 1442695040888963407UL);

        private UInt128 _state;

        public PcgReference(ulong seed)
        {
            _state = UInt128.Zero;
            _ = Next();
            _state += new UInt128(0UL, seed);
            _ = Next();
        }

        // pcg-c pcg_setseq_128_xsl_rr_64_random_r: step the LCG, then permute the
        // NEW state (pcg-cpp: output_previous = false for 128-bit state).
        public ulong Next()
        {
            _state = _state * s_multiplier + s_increment;
            var xorshifted = (ulong)(_state >> 64) ^ (ulong)_state;
            var rot = (int)((_state >> 122) & (UInt128)0x3FUL);
            return System.Numerics.BitOperations.RotateRight(xorshifted, rot);
        }
    }

    /// <summary>
    /// Independent SplitMix64 + xoshiro256** reference, ported from Blackman-Vigna
    /// (2018, arXiv:1805.01407).
    /// </summary>
    private sealed class XoshiroReference
    {
        private ulong _s0, _s1, _s2, _s3;

        public XoshiroReference(ulong seed)
        {
            var z = seed;
            _s0 = SplitMix64(ref z);
            _s1 = SplitMix64(ref z);
            _s2 = SplitMix64(ref z);
            _s3 = SplitMix64(ref z);
        }

        public ulong Next()
        {
            var result = System.Numerics.BitOperations.RotateLeft(_s1 * 5UL, 7) * 9UL;
            var t = _s1 << 17;
            _s2 ^= _s0;
            _s3 ^= _s1;
            _s1 ^= _s2;
            _s0 ^= _s3;
            _s2 ^= t;
            _s3 = System.Numerics.BitOperations.RotateLeft(_s3, 45);
            return result;
        }

        private static ulong SplitMix64(ref ulong z)
        {
            z += 0x9e3779b97f4a7c15UL;
            var x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
            return x ^ (x >> 31);
        }
    }
}
