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

namespace Boutquin.Numerics.Tests.Unit.Random;

[Trait("Tier", "Unit")]
public sealed class C3RandomLawTests
{
    [Theory]
    [InlineData(10, 9)]
    [InlineData(1500000000, 1499999999)]
    public void Lemire_RejectsZeroWord_AndReturnsMultiplyHigh(int bound, int expected)
    {
        // Lemire Algorithm 5: 2^32 mod bound > 0, so word zero must be rejected.
        var source = new ScriptedSource([0, ulong.MaxValue], []);
        ((IRandomSource)source).NextInt(bound).Should().Be(expected);
        source.WordsRead.Should().Be(2);
    }

    [Fact]
    public void Polar_RejectsBothBoundaries_AndCachesFirstPrimaryVariate()
    {
        var source = new ScriptedSource([], [0.5, 0.5, 0, 0.5, 0.75, 0.625]);
        var sampler = new GaussianSampler(source);

        // NR3 §7.3.4: accepted (u,v)=(1/2,1/4), s=5/16.
        // Independent 60-digit evaluation; return v first, then stored u.
        sampler.Next().Should().BeApproximately(0.6820999369024104, 2e-15);
        source.UniformsRead.Should().Be(6);
        sampler.Next().Should().BeApproximately(1.3641998738048209, 3e-15);
        source.UniformsRead.Should().Be(6);
    }

    [Fact]
    public void Mammen_ExactAtomsAndProbabilityWeightedThirdMoment_MatchPrimaryLaw()
    {
        const double negative = -0.6180339887498948482;
        const double positive = 1.6180339887498948482;
        const double probability = 0.7236067977499789696;
        var source = new ScriptedSource([], [0, Math.BitDecrement(probability), probability, Math.BitDecrement(1.0)]);
        var sampler = new WildBootstrapResampler<double>(WildBootstrapWeights.Mammen, source);
        var values = sampler.Resample([1.0, 1.0, 1.0, 1.0]);

        values[0].Should().BeApproximately(negative, 3e-16);
        values[1].Should().BeApproximately(negative, 3e-16);
        values[2].Should().BeApproximately(positive, 3e-16);
        values[3].Should().BeApproximately(positive, 3e-16);
        var thirdMoment = probability * Math.Pow(values[0], 3) + (1 - probability) * Math.Pow(values[2], 3);
        thirdMoment.Should().BeApproximately(1, 8e-16);
        var decimalSampler = new WildBootstrapResampler<decimal>(WildBootstrapWeights.Mammen,
            new ScriptedSource([], [0, Math.BitDecrement(1.0)]));
        var decimals = decimalSampler.Resample([1m, 1m]);
        decimals[0].Should().BeApproximately(-0.618033988749895m, 0.000000000000002m);
        decimals[1].Should().BeApproximately(1.61803398874989m, 0.00000000000001m);
    }

    private sealed class ScriptedSource(ulong[] words, double[] uniforms) : IRandomSource
    {
        public int WordsRead { get; private set; }
        public int UniformsRead { get; private set; }
        public ulong NextULong() => words[WordsRead++];
        public double NextDouble() => uniforms[UniformsRead++];
    }
}
