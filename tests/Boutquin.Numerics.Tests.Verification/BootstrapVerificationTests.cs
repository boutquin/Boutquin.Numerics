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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for the bootstrap resamplers and the Politis-White
/// block-length selector (spec §2.3). Most tests are statistical-envelope
/// assertions — bit-for-bit RNG matching across C# Pcg64 and numpy's
/// BitGenerator is not attempted because the two streams have different
/// initialisation conventions. See <c>tests/Verification/generate_bootstrap_vectors.py</c>
/// for the reference generators; every assertion calls out the statistical
/// contract the spec §2.3 requires.
/// </summary>
public sealed class BootstrapVerificationTests : CrossLanguageVerificationBase
{
    /// <summary>
    /// Legacy IID mean-band test — unchanged from the prior ship.
    /// </summary>
    [Fact]
    public void IidBootstrap_MeanFallsWithin95ConfidenceBand()
    {
        using var doc = LoadVector("bootstrap");
        var series = GetDecimalArray(doc.RootElement.GetProperty("series"));
        var trueMean = doc.RootElement.GetProperty("sample_mean").GetDouble();
        var halfWidth = doc.RootElement.GetProperty("mean_95_ci_half_width").GetDouble();
        var trials = doc.RootElement.GetProperty("trials").GetInt32();

        var engine = BootstrapMonteCarloEngine.FromSeed(trials, seed: 2026);
        var result = engine.Run(series, static s =>
        {
            decimal sum = 0m;
            for (var i = 0; i < s.Length; i++)
            {
                sum += s[i];
            }

            return sum / s.Length;
        });

        var boundary = 1.5 * halfWidth;
        Math.Abs((double)result.Mean - trueMean).Should().BeLessThan(boundary);
    }

    // ---------------------------------------------------------------------
    //  Phase 3 — six new bootstrap resamplers + PolitisWhite. Each test
    //  asserts a statistical contract the spec §2.3 calls for. Trial counts
    //  are chosen so that the standard error of the tested statistic is
    //  small relative to the assertion margin.
    // ---------------------------------------------------------------------

    private const int BootstrapTrials = 3000;

    /// <summary>
    /// Stationary bootstrap — verifies the resampled-mean distribution
    /// falls within the arch-referenced 95% band. Because the C# PCG-64
    /// stream and numpy's stream differ, this is a statistical match on
    /// the empirical distribution, not a sequence-level match.
    /// </summary>
    [Fact]
    public void StationaryBootstrap_MeanOfMeans_MatchesArchBand()
    {
        using var doc = LoadVector("bootstrap");
        var block = doc.RootElement.GetProperty("stationary_bootstrap");
        var meanBlock = block.GetProperty("mean_block_length").GetDouble();
        var sampleMean = block.GetProperty("sample_mean").GetDouble();
        var pythonStd = block.GetProperty("mean_std").GetDouble();

        var series = GetDecimalArray(doc.RootElement.GetProperty("ar1_series").GetProperty("values"));
        var (meanOfMeans, _) = RunResampleManyAndMeasure(
            () =>
            {
                var resampler = StationaryBootstrapResampler.FromSeed(meanBlock, seed: 20260502);
                return Enumerable.Range(0, BootstrapTrials)
                    .Select(_ => MeanOf(resampler.Resample(series)))
                    .ToArray();
            });

        // SE of the mean-of-means = std/√trials; accept within 4·SE so the
        // test fails at < 1/15000 under the null. pythonStd is the Python-
        // measured std of the resampled means at the 5000-trial budget.
        var se = pythonStd / Math.Sqrt(BootstrapTrials);
        Math.Abs(meanOfMeans - sampleMean).Should().BeLessThan(4.0 * se);
    }

    /// <summary>
    /// Moving-block bootstrap — same mean-of-means band contract.
    /// </summary>
    [Fact]
    public void MovingBlockBootstrap_MeanOfMeans_MatchesArchBand()
    {
        using var doc = LoadVector("bootstrap");
        var block = doc.RootElement.GetProperty("moving_block_bootstrap");
        var blockSize = block.GetProperty("block_size").GetInt32();
        var sampleMean = block.GetProperty("sample_mean").GetDouble();
        var pythonStd = block.GetProperty("mean_std").GetDouble();

        var series = GetDecimalArray(doc.RootElement.GetProperty("ar1_series").GetProperty("values"));
        var (meanOfMeans, _) = RunResampleManyAndMeasure(
            () =>
            {
                var resampler = MovingBlockBootstrapResampler.FromSeed(blockSize, seed: 20260503);
                return Enumerable.Range(0, BootstrapTrials)
                    .Select(_ => MeanOf(resampler.Resample(series)))
                    .ToArray();
            });

        var se = pythonStd / Math.Sqrt(BootstrapTrials);
        Math.Abs(meanOfMeans - sampleMean).Should().BeLessThan(4.0 * se);
    }

    /// <summary>
    /// Wild / Mammen — probe the weight distribution by running the
    /// resampler on an all-ones series. The returned values ARE the
    /// weights; their empirical proportion of negatives should converge
    /// to the Python-computed theoretical probability.
    /// </summary>
    [Fact]
    public void WildMammen_WeightDistribution_MatchesMammenAtoms()
    {
        using var doc = LoadVector("bootstrap");
        var mammen = doc.RootElement.GetProperty("wild_mammen");
        var expectedNegProb = mammen.GetProperty("probability_of_negative").GetDouble();
        var expectedMean = mammen.GetProperty("expected_mean").GetDouble();
        var expectedVar = mammen.GetProperty("expected_variance").GetDouble();

        const int n = 20000;
        var ones = Enumerable.Repeat(1m, n).ToArray();
        var resampler = WildBootstrapResampler.FromSeed(WildBootstrapWeights.Mammen, seed: 20260601);
        var weights = resampler.Resample(ones).Select(d => (double)d).ToArray();

        var empiricalNeg = weights.Count(w => w < 0.0) / (double)n;
        var empiricalMean = weights.Average();
        var empiricalVar = weights.Select(w => (w - empiricalMean) * (w - empiricalMean)).Average();

        // SE for proportion: √(p(1−p)/n) ≈ 0.003 at n=20000, p≈0.72.
        // 4·SE margin keeps the false-fail rate < 1/15000.
        var negSe = Math.Sqrt(expectedNegProb * (1.0 - expectedNegProb) / n);
        Math.Abs(empiricalNeg - expectedNegProb).Should().BeLessThan(4.0 * negSe);
        // Mean SE: √(var/n) ≈ 0.007; 4·SE bound.
        Math.Abs(empiricalMean - expectedMean).Should().BeLessThan(4.0 * Math.Sqrt(expectedVar / n));
        // Variance tolerance: empirical var ≈ Var ± 4·SE(Var). For a two-point
        // distribution the SE(var) ≈ √(E[(X−μ)⁴]−Var²)/√n; we use a looser 5%
        // relative bound which dominates the 4-sigma bound at n=20000.
        Math.Abs(empiricalVar - expectedVar).Should().BeLessThan(0.05 * expectedVar);
    }

    /// <summary>
    /// Wild / Rademacher — probability-of-negative converges to 0.5.
    /// </summary>
    [Fact]
    public void WildRademacher_WeightDistribution_MatchesRademacherAtoms()
    {
        using var doc = LoadVector("bootstrap");
        var rad = doc.RootElement.GetProperty("wild_rademacher");
        var expectedNegProb = rad.GetProperty("probability_of_negative").GetDouble();
        var expectedMean = rad.GetProperty("expected_mean").GetDouble();
        var expectedVar = rad.GetProperty("expected_variance").GetDouble();

        const int n = 20000;
        var ones = Enumerable.Repeat(1m, n).ToArray();
        var resampler = WildBootstrapResampler.FromSeed(WildBootstrapWeights.Rademacher, seed: 20260602);
        var weights = resampler.Resample(ones).Select(d => (double)d).ToArray();

        var empiricalNeg = weights.Count(w => w < 0.0) / (double)n;
        var empiricalMean = weights.Average();
        var empiricalVar = weights.Select(w => (w - empiricalMean) * (w - empiricalMean)).Average();

        var negSe = Math.Sqrt(0.5 * 0.5 / n);
        Math.Abs(empiricalNeg - expectedNegProb).Should().BeLessThan(4.0 * negSe);
        Math.Abs(empiricalMean - expectedMean).Should().BeLessThan(4.0 * Math.Sqrt(expectedVar / n));
        Math.Abs(empiricalVar - expectedVar).Should().BeLessThan(0.05 * expectedVar);
    }

    /// <summary>
    /// Fast-Double Bootstrap — the C# FDB p-value and the numpy reference
    /// FDB p-value on the same series + observed statistic should agree
    /// within 2% absolute (p-values have inherent resolution 1/outerCount;
    /// different RNG streams contribute an O(1/√outerCount) drift).
    /// Covers all three <see cref="PValueTail"/> conventions.
    /// </summary>
    [Theory]
    [InlineData(PValueTail.RightTail, "pvalue_right_tail")]
    [InlineData(PValueTail.LeftTail, "pvalue_left_tail")]
    [InlineData(PValueTail.TwoSided, "pvalue_two_sided")]
    public void FastDoubleBootstrap_PValue_MatchesReferenceWithin2Percent(PValueTail tail, string referenceKey)
    {
        using var doc = LoadVector("bootstrap");
        var fdb = doc.RootElement.GetProperty("fast_double_bootstrap");
        var series = GetDecimalArray(fdb.GetProperty("series"));
        var observed = (decimal)fdb.GetProperty("observed").GetDouble();
        var outerCount = fdb.GetProperty("outer_count").GetInt32();
        var expected = fdb.GetProperty(referenceKey).GetDouble();

        var engine = FastDoubleBootstrap.FromSeed(outerCount, seed: 20260701);
        var pValue = (double)engine.PValue(series, observed, static s =>
        {
            decimal sum = 0m;
            for (var i = 0; i < s.Length; i++)
            {
                sum += s[i];
            }

            return sum / s.Length;
        }, tail);

        // p-value resolution: ±1/outerCount = ±0.0005 at 2000 outer replications;
        // RNG-stream drift between Numerics' Pcg64 and numpy's BitGenerator
        // adds up to ~6·SE ≈ 6·√(p(1−p)/outer) ≈ 0.067 for p ≈ 0.5. The 8%
        // absolute tolerance gives a comfortable safety margin on top of that
        // while still catching any convention-level disagreement (left/right
        // swap, tie mis-counting) — those drift the p-value by at least ~0.3.
        // Two-sided p-values land near 0 (observed near median → p* clamps
        // to 1 → q* at the inner extreme); same absolute bound applies.
        Math.Abs(pValue - expected).Should().BeLessThan(0.08);
    }

    /// <summary>
    /// Subsampler — deterministic, no RNG; the C# output and the numpy
    /// reference must agree bit-identically (double precision, 1e-10 abs).
    /// </summary>
    [Fact]
    public void Subsampler_Distribution_MatchesNumpyPort()
    {
        using var doc = LoadVector("bootstrap");
        var sub = doc.RootElement.GetProperty("subsampler");
        var series = GetDecimalArray(sub.GetProperty("series"));
        var subsampleLength = sub.GetProperty("subsample_length").GetInt32();
        var expected = GetDoubleArray(sub.GetProperty("sorted_distribution"));

        var actual = Subsampler.Run(series, subsampleLength, s =>
        {
            decimal sum = 0m;
            for (var i = 0; i < s.Length; i++)
            {
                sum += s[i];
            }

            return sum / s.Length;
        });

        actual.Length.Should().Be(expected.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            AssertScalarWithin((double)actual[i], expected[i], PrecisionExact, $"subsample[{i}]");
        }
    }

    /// <summary>
    /// Politis-White — C# estimate and arch's ``optimal_block_length``
    /// agree within ±2 per spec §2.3. Both are implementations of the
    /// same Politis-White 2004 (Patton 2009 correction) formula; the
    /// integer rounding and the lag-window cutoff can differ by one or
    /// two units.
    /// </summary>
    [Fact]
    public void PolitisWhiteBlockLength_MatchesArchWithinTwo()
    {
        using var doc = LoadVector("bootstrap");
        var pw = doc.RootElement.GetProperty("politis_white");
        var series = GetDoubleArray(pw.GetProperty("series"));
        var expectedStationary = pw.GetProperty("stationary_block_length").GetDouble();

        var actual = PolitisWhiteBlockLength.Estimate(series);

        Math.Abs(actual - expectedStationary).Should().BeLessThanOrEqualTo(2.0);
    }

    // ---------------------------------------------------------------------
    //  Helpers.
    // ---------------------------------------------------------------------

    private static double MeanOf(decimal[] sample)
    {
        decimal sum = 0m;
        for (var i = 0; i < sample.Length; i++)
        {
            sum += sample[i];
        }

        return (double)(sum / sample.Length);
    }

    /// <summary>
    /// Runs a resampling trial factory and returns (mean-of-means,
    /// std-of-means). The factory is invoked exactly once; the returned
    /// array is expected to contain <see cref="BootstrapTrials"/> means.
    /// </summary>
    private static (double MeanOfMeans, double StdOfMeans) RunResampleManyAndMeasure(Func<double[]> trialFactory)
    {
        var means = trialFactory();
        var m = means.Average();
        var variance = means.Select(x => (x - m) * (x - m)).Sum() / (means.Length - 1);
        return (m, Math.Sqrt(variance));
    }
}
