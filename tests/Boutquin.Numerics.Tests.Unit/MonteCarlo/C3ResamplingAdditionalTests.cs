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
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingAdditionalTests
{
    [Fact]
    public void Halton_ExactRationalDigitReversal_AllFiftyPrimeBases()
    {
        // Independent integer arithmetic oracle: reverse radix digits as an integer,
        // divide once by base^digitCount, rather than accumulating floating fractions.
        var primes = Enumerable.Range(2, 228)
            .Where(p => !Enumerable.Range(2, Math.Max(0, (int)Math.Sqrt(p) - 1)).Any(d => p % d == 0))
            .Take(50).ToArray();
        primes.Length.Should().Be(50);
        foreach (var index in new[] { 1, 2, 5, 8, 31, 255, 1023, 4095 })
        {
            var actual = new HaltonSequence<double>(50, index - 1).Next();
            for (var d = 0; d < 50; d++)
            {
                var digits = new List<int>();
                var value = index;
                while (value != 0)
                {
                    digits.Add(value % primes[d]);
                    value /= primes[d];
                }
                BigInteger numerator = 0;
                foreach (var digit in digits)
                {
                    numerator = numerator * primes[d] + digit;
                }
                var denominator = BigInteger.Pow(primes[d], digits.Count);
                actual[d].Should().BeApproximately((double)numerator / (double)denominator, 4e-16);
                actual[d].Should().BeGreaterThan(0).And.BeLessThan(1);
            }
        }
    }

    [Fact]
    public void Engine_MultipleStatistics_ShareExactSamples()
    {
        var rng = new Choices();
        var samples = new List<decimal[]>();
        var result = new BootstrapMonteCarloEngine<decimal>(4, rng).RunMulti([2m, 8m], 2, (s, o) =>
        {
            samples.Add(s.ToArray());
            o[0] = s[0] + s[1];
            o[1] = s[0] - s[1];
        });
        samples.Count.Should().Be(4);
        samples[0].Should().Equal(2m, 2m);
        samples[1].Should().Equal(2m, 8m);
        samples[2].Should().Equal(8m, 2m);
        samples[3].Should().Equal(8m, 8m);
        result.Statistics[0].Should().Equal(4m, 10m, 10m, 16m);
        result.Statistics[1].Should().Equal(-6m, 0m, 0m, 6m);
        result.Means.Should().Equal(10m, 0m);
        result.Medians.Should().Equal(10m, 0m);
    }

    [Fact]
    public void Rademacher_TwoAtoms_IndependentCoordinateTrace()
    {
        // Four atoms equally likely: E=0, second moment1, third moment0.
        var result = new WildBootstrapResampler<decimal>(WildBootstrapWeights.Rademacher, new Choices())
            .Resample([1m, 2m, 3m, 4m]);
        result.Should().Equal(-1m, 2m, 3m, -4m);
    }

    [Fact]
    public void Stationary_UnitMean_IsIidAtEveryStep()
    {
        new StationaryBootstrapResampler<decimal>(1, new Choices()).Resample([1m, 2m, 4m, 8m])
            .Should().Equal(1m, 1m, 1m, 2m);
        Action invalid = () => new StationaryBootstrapResampler<decimal>(.5, new Choices());
        invalid.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GaussianMultiplier_ExactPolarInput()
    {
        // Both uniforms3/4 give polar coordinates1/2,1/2 and normals sqrt(log2).
        var actual = new WildBootstrapResampler<double>(WildBootstrapWeights.Gaussian, new ConstantUniform())
            .Resample([2.0, -3.0]);
        actual[0].Should().BeApproximately(1.66510922231539551270632928979040209526117770452888145833366, 2e-15);
        actual[1].Should().BeApproximately(-2.49766383347309326905949393468560314289176655679332218750049, 3e-15);
    }

    private sealed class ConstantUniform : IRandomSource
    {
        public ulong NextULong() => throw new NotSupportedException();
        public double NextDouble() => .75;
    }

    private sealed class Choices : IRandomSource
    {
        private int _i;
        private int _d;
        private readonly int[] _indices = [0, 0, 0, 1, 1, 0, 1, 1];
        private readonly double[] _uniforms = [0, .5, .9, .25];
        public ulong NextULong() => throw new NotSupportedException();
        public int NextInt(int upperExclusive) => _indices[_i++ % _indices.Length];
        public double NextDouble() => _uniforms[_d++ % _uniforms.Length];
    }
}
