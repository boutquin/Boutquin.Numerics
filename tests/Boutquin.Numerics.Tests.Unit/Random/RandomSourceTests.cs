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

using System.Numerics;
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
        // Independently derived by squaring the 256-bit GF(2) transition matrix
        // 128 times, then applying it to the SplitMix64-expanded seed 7. This oracle
        // uses no jump-polynomial constants and advances exactly 2^128 steps.
        ulong[] expected =
        [
            0x156617fd83df2a74UL, 0x1ccb4975f3ae6cbcUL,
            0xc6b79bd4fd3989f0UL, 0x335a2f71da09ba0eUL,
            0xcae07fd267535616UL, 0x3cd5ab0d0dab15f7UL,
            0xa8ecdfbee86de85cUL, 0x83139c6298ae2616UL,
        ];
        var concrete = new Xoshiro256StarStarRandomSource(7UL);
        var generic = new Xoshiro256StarStarRandomSource<double>(7UL);
        concrete.Jump();
        generic.Jump();

        foreach (var word in expected)
        {
            concrete.NextULong().Should().Be(word);
            generic.NextULong().Should().Be(word);
        }
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
    public void NextT_MaximumDrawStaysBelowExclusiveUpperBound()
    {
        IRandomSource<Half> halfSource = new MaximumRandomSource<Half>();
        IRandomSource<float> floatSource = new MaximumRandomSource<float>();

        Assert.True(halfSource.NextT() < Half.One, "Half NextT must remain below one");
        Assert.True(floatSource.NextT() < 1f, "float NextT must remain below one");
    }

    private sealed class MaximumRandomSource<T> : IRandomSource<T>
        where T : IFloatingPoint<T>
    {
        public ulong NextULong() => ulong.MaxValue;
    }

    public enum RngKind
    {
        Pcg64,
        Xoshiro256,
    }

    private static IRandomSource MakeRng(RngKind kind, ulong seed) => kind switch
    {
        RngKind.Pcg64 => new Pcg64RandomSource(seed),
        RngKind.Xoshiro256 => new Xoshiro256StarStarRandomSource(seed),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
    };

    [Theory]
    [InlineData(RngKind.Pcg64)]
    [InlineData(RngKind.Xoshiro256)]
    public void NextInt_IsUniformlyDistributedAcrossBuckets(RngKind kind)
    {
        var rng = MakeRng(kind, 99UL);
        const int buckets = 10;
        const int draws = 100_000;
        var counts = new int[buckets];
        for (var i = 0; i < draws; i++)
        {
            counts[rng.NextInt(buckets)]++;
        }

        // Expected per bucket: 10000; σ = sqrt(draws · p · (1−p)) = sqrt(9000) ≈ 94.9.
        // A ±300 band is ≈ 3.16σ per bucket — tight enough to catch a biased
        // generator, loose enough that 10 buckets won't false-fail (the old ±500
        // band was ~5.3σ and far too forgiving). Exercised over BOTH shipped RNGs.
        foreach (var c in counts)
        {
            c.Should().BeGreaterThan(10000 - 300);
            c.Should().BeLessThan(10000 + 300);
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

    [Fact]
    public void GaussianSampler_ShiftAndScalePreservesFiniteCancellation()
    {
        // NR point(0,1/4) returns sqrt(log256), between2.35 and2.36.
        // Every selected scale overflows before cancellation, but the result fits.
        var standard = new GaussianSampler(new CancellationUniformSource()).Next();
        var expected = Math.FusedMultiplyAdd(1e308, standard, -1e308);

        var actual = new GaussianSampler(new CancellationUniformSource()).Next(-1e308, 1e308);

        double.IsFinite(actual).Should().BeTrue();
        actual.Should().Be(expected);
    }

    [Fact]
    public void GenericGaussianSampler_FloatAndHalfShiftAndScalePreservesFiniteCancellation()
    {
        using var scope = new FluentAssertions.Execution.AssertionScope();
        // NR point(0,1/4) returns sqrt(log256), between2.35 and2.36.
        // Every selected scale overflows before cancellation, but the result fits.
        var floatNormal = new GaussianSampler<float>(new CancellationUniformSource()).Next();
        var expectedFloat = (float)((double)2e38f * floatNormal - 2e38f);
        var actualFloat = new GaussianSampler<float>(new CancellationUniformSource()).Next(-2e38f, 2e38f);
        actualFloat.Should().Be(expectedFloat);

        var halfNormal = new GaussianSampler<Half>(new CancellationUniformSource()).Next();
        var expectedHalf = (Half)(40000.0 * (double)halfNormal - 40000.0);
        var actualHalf = new GaussianSampler<Half>(new CancellationUniformSource()).Next((Half)(-40000), (Half)40000);
        actualHalf.Should().Be(expectedHalf);
    }

    [Fact]
    public void GenericGaussianSampler_DecimalShiftAndScalePreservesFiniteCancellation()
    {
        // NR point(0,1/4) returns sqrt(log256), between2.35 and2.36.
        // Every selected scale overflows before cancellation, but the result fits.
        var standard = new GaussianSampler<decimal>(new CancellationUniformSource()).Next();
        var expected = 5e28m * (standard - 1m);

        decimal actual = 0m;
        var exception = Record.Exception(() => actual = new GaussianSampler<decimal>(new CancellationUniformSource()).Next(-5e28m, 5e28m));

        exception.Should().BeNull();
        actual.Should().Be(expected);
    }

    [Theory]
    [InlineData(0.0, -1.0)]
    [InlineData(0.0, double.NaN)]
    [InlineData(0.0, double.PositiveInfinity)]
    [InlineData(double.NaN, 1.0)]
    [InlineData(double.NegativeInfinity, 1.0)]
    public void GaussianSampler_RejectsInvalidShiftOrScale(double mean, double stdDev)
    {
        var legacy = new GaussianSampler(new Pcg64RandomSource(1UL));
        Assert.Throws<ArgumentOutOfRangeException>(() => legacy.Next(mean, stdDev));
    }

    [Theory]
    [InlineData(0.0, -1.0)]
    [InlineData(0.0, double.NaN)]
    [InlineData(0.0, double.PositiveInfinity)]
    [InlineData(double.NaN, 1.0)]
    [InlineData(double.NegativeInfinity, 1.0)]
    public void GenericGaussianSampler_RejectsInvalidShiftOrScale(double mean, double stdDev)
    {
        var generic = new GaussianSampler<double>(new Pcg64RandomSource(1UL));
        Assert.Throws<ArgumentOutOfRangeException>(() => generic.Next(mean, stdDev));
    }

    [Fact]
    public void GaussianSampler_ZeroScaleReturnsFiniteMean()
    {
        var legacy = new GaussianSampler(new Pcg64RandomSource(1UL));
        var generic = new GaussianSampler<double>(new Pcg64RandomSource(1UL));
        legacy.Next(3.0, 0.0).Should().Be(3.0);
        generic.Next(3.0, 0.0).Should().Be(3.0);
    }
    private sealed class CancellationUniformSource : IRandomSource
    {
        private int _draw;
        public ulong NextULong() => throw new NotSupportedException();
        public double NextDouble() => _draw++ == 0 ? .5 : .625;
    }

}
