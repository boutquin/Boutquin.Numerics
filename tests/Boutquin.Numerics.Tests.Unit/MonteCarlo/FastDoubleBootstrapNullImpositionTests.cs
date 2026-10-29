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
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

/// <summary>
/// Size of the fast double bootstrap when the null is imposed at both levels.
/// Davidson (2023, "An Improved Fast Double Bootstrap", §4) draws the outer samples
/// from a null-imposed bootstrap DGP and each inner sample from the null-imposed DGP
/// estimated on that outer sample. For a location test with H₀: μ = 0 and the statistic
/// <c>|mean|</c>, the null-imposed DGP of a sample is the empirical law of that sample
/// recentred to mean zero. Under H₀ a correctly sized test rejects at nominal 5% about
/// 5% of the time; drawing the inner sample from the raw (un-recentred) outer sample
/// instead rejects about 17% of the time at n = 20.
/// </summary>
public sealed class FastDoubleBootstrapNullImpositionTests
{
    private const int SampleSize = 20;
    private const int OuterCount = 199;
    private const int Replications = 400;

    [Fact]
    public void PValue_WithNullImposition_HasNominalSizeUnderTheNull()
    {
        var sampler = new GaussianSampler(new Xoshiro256StarStarRandomSource(20261002UL));
        var rejections = 0;
        for (var r = 0; r < Replications; r++)
        {
            var x = new decimal[SampleSize];
            for (var i = 0; i < SampleSize; i++)
            {
                x[i] = (decimal)sampler.Next();
            }

            var observed = AbsMean(x);
            var engine = new FastDoubleBootstrap(OuterCount, new Pcg64RandomSource((ulong)(1000 + r)));

            var p = engine.PValueImposingNull(x, observed, AbsMean, RecentreToZero, PValueTail.RightTail);

            if (p <= 0.05m)
            {
                rejections++;
            }
        }

        var size = (double)rejections / Replications;

        // Monte Carlo SE at 5% with 400 replications is ≈ 0.011.
        size.Should().BeInRange(0.02, 0.10,
            "with the null imposed on the outer and every inner DGP, the FDB test has nominal 5% size");
    }

    [Fact]
    public void PValue_WithNullImposition_TransformsOnlyThePopulations_NotTheStatisticInput()
    {
        // The statistic sees each drawn sample as drawn. With imposeNull adding 100 to every
        // element, every outer and inner sample is drawn from values ≥ 100.
        decimal[] x = [1m, 2m, 3m, 4m, 5m];
        var minimumSeen = decimal.MaxValue;
        var calls = 0;
        var engine = new FastDoubleBootstrap(50, new Pcg64RandomSource(3UL));

        engine.PValueImposingNull(
            x,
            observed: 0m,
            statistic: s =>
            {
                calls++;
                foreach (var v in s)
                {
                    minimumSeen = Math.Min(minimumSeen, v);
                }

                return 0m;
            },
            imposeNull: s =>
            {
                for (var i = 0; i < s.Length; i++)
                {
                    s[i] += 100m;
                }
            });

        calls.Should().Be(100, "one outer and one inner statistic per outer draw");
        minimumSeen.Should().BeGreaterThanOrEqualTo(101m, "outer samples come from the null-imposed data");
        x.Should().Equal([1m, 2m, 3m, 4m, 5m], "the caller's observations are not mutated");
    }

    [Fact]
    public void PValueImposingNull_NullTransform_IsRejected()
    {
        // A null transform must not silently fall back to raw resampling at both levels.
        var engine = new FastDoubleBootstrap(50, new Pcg64RandomSource(5UL));

        var act = () => engine.PValueImposingNull([1m, 2m, 3m], 2m, AbsMean, imposeNull: null!);

        act.Should().Throw<ArgumentNullException>().WithParameterName("imposeNull");
    }

    private static decimal AbsMean(ReadOnlySpan<decimal> s)
    {
        var sum = 0m;
        foreach (var v in s)
        {
            sum += v;
        }

        return Math.Abs(sum / s.Length);
    }

    private static void RecentreToZero(Span<decimal> s)
    {
        var sum = 0m;
        foreach (var v in s)
        {
            sum += v;
        }

        var mean = sum / s.Length;
        for (var i = 0; i < s.Length; i++)
        {
            s[i] -= mean;
        }
    }
}
