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

using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Random;

public sealed class RandomSourceTests
{
    [Fact]
    public void Pcg64_SameSeedProducesIdenticalSequence()
    {
        var a = new Pcg64RandomSource(42UL);
        var b = new Pcg64RandomSource(42UL);
        for (var i = 0; i < 100; i++)
        {
            a.NextULong().Should().Be(b.NextULong());
        }
    }

    [Fact]
    public void Pcg64_DifferentStreamsDiverge()
    {
        var a = new Pcg64RandomSource(42UL, streamId: 0UL);
        var b = new Pcg64RandomSource(42UL, streamId: 1UL);
        var divergences = 0;
        for (var i = 0; i < 100; i++)
        {
            if (a.NextULong() != b.NextULong())
            {
                divergences++;
            }
        }

        divergences.Should().BeGreaterThan(95, "independent streams should produce essentially uncorrelated output");
    }

    [Fact]
    public void Xoshiro256_SameSeedProducesIdenticalSequence()
    {
        var a = new Xoshiro256StarStarRandomSource(7UL);
        var b = new Xoshiro256StarStarRandomSource(7UL);
        for (var i = 0; i < 100; i++)
        {
            a.NextULong().Should().Be(b.NextULong());
        }
    }

    [Fact]
    public void Xoshiro256_JumpPartitionsTheStream()
    {
        var a = new Xoshiro256StarStarRandomSource(7UL);
        var b = new Xoshiro256StarStarRandomSource(7UL);
        b.Jump();

        var divergences = 0;
        for (var i = 0; i < 50; i++)
        {
            if (a.NextULong() != b.NextULong())
            {
                divergences++;
            }
        }

        divergences.Should().BeGreaterThan(45);
    }

    [Fact]
    public void NextDouble_LiesInUnitInterval()
    {
        IRandomSource rng = new Pcg64RandomSource(12345UL);
        for (var i = 0; i < 10_000; i++)
        {
            var d = rng.NextDouble();
            d.Should().BeGreaterThanOrEqualTo(0.0).And.BeLessThan(1.0);
        }
    }

    [Fact]
    public void NextInt_IsUniformlyDistributedAcrossBuckets()
    {
        IRandomSource rng = new Pcg64RandomSource(99UL);
        const int buckets = 10;
        const int draws = 100_000;
        var counts = new int[buckets];
        for (var i = 0; i < draws; i++)
        {
            counts[rng.NextInt(buckets)]++;
        }

        // Expected per bucket: 10000; 6σ ≈ 190 ~ allow a wide band.
        foreach (var c in counts)
        {
            c.Should().BeGreaterThan(9500);
            c.Should().BeLessThan(10500);
        }
    }

    [Fact]
    public void NextInt_RejectsNonPositiveBound()
    {
        IRandomSource rng = new Pcg64RandomSource(1UL);
        Action zero = () => rng.NextInt(0);
        Action negative = () => rng.NextInt(-5);
        zero.Should().Throw<ArgumentOutOfRangeException>();
        negative.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GaussianSampler_MeanAndVarianceConverge()
    {
        var sampler = new GaussianSampler(new Pcg64RandomSource(2026UL));
        const int n = 100_000;
        var samples = sampler.NextBatch(n);

        double mean = 0.0;
        for (var i = 0; i < n; i++)
        {
            mean += samples[i];
        }

        mean /= n;

        double sumSq = 0.0;
        for (var i = 0; i < n; i++)
        {
            var d = samples[i] - mean;
            sumSq += d * d;
        }

        var variance = sumSq / (n - 1);

        Math.Abs(mean).Should().BeLessThan(0.02);
        Math.Abs(variance - 1.0).Should().BeLessThan(0.02);
    }

    [Fact]
    public void GaussianSampler_ShiftAndScale()
    {
        var sampler = new GaussianSampler(new Pcg64RandomSource(1UL));
        var raw = sampler.Next();
        var shifted = new GaussianSampler(new Pcg64RandomSource(1UL)).Next(5.0, 2.0);

        // Shift/scale: 5 + 2 * raw.
        Math.Abs(shifted - (5.0 + 2.0 * raw)).Should().BeLessThan(1e-12);
    }
}
