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

[Trait("Tier", "Unit")]
public sealed class C3CoreRandomControls
{
    [Fact]
    public void Lemire_RepeatedRejection_RecomputesBothProductWords()
    {
        // Exact integer products with bound10 are 3*2^32+2, 1*2^32+4,
        // and 4*2^32+6. Threshold is 2^32 mod10=6: reject twice, accept equality.
        var source = new WordSource(1288490189U, 429496730U, 1717986919U);
        ((IRandomSource)source).NextInt(10).Should().Be(4);
        source.Reads.Should().Be(3);
    }

    [Fact]
    public void Lemire_ThresholdEquality_IsAcceptedWithoutAnotherDraw()
    {
        var source = new WordSource(1717986919U, uint.MaxValue);
        ((IRandomSource)source).NextInt(10).Should().Be(4);
        source.Reads.Should().Be(1);
    }

    [Fact]
    public void Lemire_LowWordEqualToRange_IsAccepted()
    {
        var source = new WordSource(1U, uint.MaxValue);
        ((IRandomSource)source).NextInt(10).Should().Be(0);
        source.Reads.Should().Be(1);
    }

    [Theory]
    [InlineData(1, 4294967295U, 0)]
    [InlineData(16, 4026531840U, 15)]
    [InlineData(2147483647, 4294967295U, 2147483646)]
    public void Lemire_FullWidthProduct_HasExactIntegerQuotient(int bound, uint word, int expected)
    {
        // Independent arbitrary-precision division checks fixed expected quotient.
        ((BigInteger)word * bound / (BigInteger.One << 32)).Should().Be(expected);
        var source = new WordSource(word);
        ((IRandomSource)source).NextInt(bound).Should().Be(expected);
        source.Reads.Should().Be(1);
    }

    [Theory]
    [InlineData(0.5, 0.5)]
    [InlineData(0.0, 0.5)]
    [InlineData(0.0, 0.0)]
    public void Polar_RejectedPoint_IsSkippedBeforeAcceptedPair(double first, double second)
    {
        var source = new UniformSource(first, second, .75, .75);
        var sampler = new GaussianSampler(source);
        // Accepted squared radius1/2, each squared output log2.
        sampler.Next().Should().BeApproximately(0.83255461115769775635, 2e-15);
        source.Reads.Should().Be(4);
        sampler.Next().Should().BeApproximately(0.83255461115769775635, 2e-15);
        source.Reads.Should().Be(4);
    }

    [Fact]
    public void Polar_SignedPair_SatisfiesIndependentRadiusAndDirection()
    {
        var source = new UniformSource(.25, .625);
        var sampler = new GaussianSampler(source);
        var pair = new[] { sampler.Next(), sampler.Next() };
        // Point(-1/2,1/4): squared radius5/16. Pair order is deliberately
        // immaterial here; the separate author-stream control fixes return order.
        Array.Sort(pair);
        pair[0].Should().BeApproximately(-1.3641998738048208995, 3e-15);
        pair[1].Should().BeApproximately(0.68209993690241044975, 2e-15);
        source.Reads.Should().Be(2);
    }

    [Fact]
    public void Polar_AffineTransform_MatchesIndependentScalarAcrossTypes()
    {
        const double expected = 2.6651092223153955127;
        new GaussianSampler(new UniformSource(.75, .75)).Next(1, 2)
            .Should().BeApproximately(expected, 2e-15);
        new GaussianSampler<double>(new UniformSource(.75, .75)).Next(1, 2)
            .Should().BeApproximately(expected, 2e-15);
        new GaussianSampler<float>(new UniformSource(.75, .75)).Next(1, 2)
            .Should().BeApproximately((float)expected, 3e-7f);
        ((double)new GaussianSampler<Half>(new UniformSource(.75, .75)).Next((Half)1, (Half)2))
            .Should().BeApproximately(expected, .002);
        new GaussianSampler<decimal>(new UniformSource(.75, .75)).Next(1m, 2m)
            .Should().BeApproximately(2.665109222315396m, .000000000000004m);
    }

    [Fact]
    public void DefaultAdapters_UseUpperBitsAndExclusiveTypedEndpoints()
    {
        IRandomSource words = new RawWordSource(0x12345678abcdef01UL, 0x80000000000007ffUL, ulong.MaxValue);
        words.NextUInt().Should().Be(0x12345678U);
        words.NextDouble().Should().Be(.5);
        words.NextDouble().Should().Be(Math.BitDecrement(1.0));
        IRandomSource<float> single = new TypedMaximum<float>();
        IRandomSource<Half> half = new TypedMaximum<Half>();
        single.NextT().Should().Be(MathF.BitDecrement(1f));
        half.NextT().Should().Be(Half.BitDecrement(Half.One));
    }

    private sealed class WordSource(params uint[] words) : IRandomSource
    {
        public int Reads { get; private set; }
        public ulong NextULong() => (ulong)words[Math.Min(Reads++, words.Length - 1)] << 32;
    }

    private sealed class RawWordSource(params ulong[] words) : IRandomSource
    {
        private int _index;
        public ulong NextULong() => words[_index++];
    }

    private sealed class UniformSource(params double[] values) : IRandomSource
    {
        public int Reads { get; private set; }
        public ulong NextULong() => throw new NotSupportedException();
        public double NextDouble() => values[Reads++];
    }

    private sealed class TypedMaximum<T> : IRandomSource<T> where T : IFloatingPoint<T>
    {
        public ulong NextULong() => ulong.MaxValue;
    }
}
