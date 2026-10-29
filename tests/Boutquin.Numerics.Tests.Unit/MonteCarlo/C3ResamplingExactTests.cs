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

public sealed class C3ResamplingExactTests
{
    [Fact]
    public void Circular_AllStarts_ExactConditionalMeanAndWrap()
    {
        // Exhaust every equally likely circular start: the column sums are all 16.
        decimal[] x = [1, 2, 5, 8];
        decimal[][] expected = [[1, 2, 5, 8], [2, 5, 8, 1], [5, 8, 1, 2], [8, 1, 2, 5]];
        for (var start = 0; start < 4; start++)
        {
            var rng = new Script([start]);
            var actual = new BootstrapResampler<decimal>(4, rng).Resample(x);
            actual.Should().Equal(expected[start]);
            actual.Average().Should().Be(4m);
            rng.Bounds.Should().Equal(4);
        }
        new BootstrapResampler<decimal>(1, new Script([3, 3, 0, 1]))
            .Resample(x).Should().Equal(8m, 8m, 1m, 2m);
    }

    [Fact]
    public void Moving_AllNineBlockPairs_ExactMeanAndVariance()
    {
        // n=4,l=2. Block means 1.5,3.5,6.5. E*=23/6; Var*=19/9.
        decimal[] x = [1, 2, 5, 8];
        decimal[][] blocks = [[1, 2], [2, 5], [5, 8]];
        var means = new List<decimal>();
        for (var a = 0; a < 3; a++)
        {
            for (var b = 0; b < 3; b++)
            {
                var rng = new Script([a, b]);
                var actual = new MovingBlockBootstrapResampler<decimal>(2, rng).Resample(x);
                actual.Should().Equal(blocks[a].Concat(blocks[b]));
                rng.Bounds.Should().Equal(3, 3);
                means.Add(actual.Average());
            }
        }
        means.Average().Should().BeApproximately(23m / 6, 1e-25m);
        means.Select(v => (v - 23m / 6) * (v - 23m / 6)).Average()
            .Should().BeApproximately(19m / 9, 1e-25m);
    }

    [Fact]
    public void Moving_UnitBlocks_AreExactIidDraws()
    {
        var random = new Script([3, 3, 0, 1]);
        new MovingBlockBootstrapResampler<decimal>(1, random)
            .Resample([1m, 2m, 5m, 8m]).Should().Equal(8m, 8m, 1m, 2m);
        random.Bounds.Should().Equal(4, 4, 4, 4);
    }

    [Fact]
    public void Stationary_ExactTransitionTrace()
    {
        // p=1/2: advance at .5 and .9, restart at .4. Starting at final item wraps.
        var rng = new Script([3, 2], [.5, .4, .9, .9]);
        new StationaryBootstrapResampler<decimal>(2, rng).Resample([1m, 2m, 5m, 8m])
            .Should().Equal(8m, 1m, 5m, 8m);
        rng.Bounds.Should().Equal(4, 4);
    }

    [Fact]
    public void Wild_Mammen_ExactAtomsAndWeightedMoments()
    {
        // Algebraic atoms solve w²=w+1. High precision constants independent of Math.Sqrt.
        const double low = -0.6180339887498948482;
        const double high = 1.6180339887498948482;
        const double p = 0.7236067977499789696;
        var rng = new Script([], [0, .7236067977499788, .7236067977499791, .99]);
        var actual = new WildBootstrapResampler<double>(WildBootstrapWeights.Mammen, rng)
            .Resample([2.0, -3.0, 4.0, 1.0]);
        actual[0].Should().BeApproximately(2 * low, 2e-15);
        actual[1].Should().BeApproximately(-3 * low, 2e-15);
        actual[2].Should().BeApproximately(4 * high, 3e-15);
        actual[3].Should().BeApproximately(high, 1e-15);
        var a = actual[0] / 2;
        var b = actual[3];
        (p * a + (1 - p) * b).Should().BeApproximately(0, 1e-15);
        (p * a * a + (1 - p) * b * b).Should().BeApproximately(1, 1e-15);
        (p * a * a * a + (1 - p) * b * b * b).Should().BeApproximately(1, 2e-15);
        rng.DoubleCalls.Should().Be(4);
    }

    [Fact]
    public void Subsampling_AllBlocks_ExactFunctional()
    {
        var seen = new List<decimal[]>();
        var actual = Subsampler.Run<decimal>([1m, 7m, -2m, 4m, 9m], 3, s =>
        {
            seen.Add(s.ToArray());
            return s[0] + 2 * s[1] + 3 * s[2];
        });
        seen.Count.Should().Be(3);
        seen[0].Should().Equal(1m, 7m, -2m);
        seen[1].Should().Equal(7m, -2m, 4m);
        seen[2].Should().Equal(-2m, 4m, 9m);
        actual.Should().Equal(9m, 15m, 33m);
    }

    [Fact]
    public void HarrellDavis_IntegralPolynomial_ExactRational()
    {
        // n=3,p=1/4 -> Beta(1,3), CDF=1-(1-x)^3.
        // Bin masses 19/27,7/27,1/27; weighted sum on [1,4,13] is 20/9.
        HarrellDavisPercentile.Compute<double>([1, 4, 13], .25)
            .Should().BeApproximately(20.0 / 9, 1e-10);
        // p=3/4 -> Beta(3,1), masses reversed: (1+28+247)/27=92/9.
        HarrellDavisPercentile.Compute<double>([1, 4, 13], .75)
            .Should().BeApproximately(92.0 / 9, 1e-10);
        HarrellDavisPercentile.Compute<decimal>([7m, 7m, 7m], .25).Should().Be(7m);
    }

    [Fact]
    public void TypeSeven_ExactKnotsAndInterpolation()
    {
        decimal[] x = [-8, -2, 4, 20, 36];
        Percentile.Compute<decimal>(x, .375).Should().Be(1m);
        Percentile.Compute<decimal>(x, .75).Should().Be(20m);
        Percentile.Compute<decimal>(x, 0).Should().Be(-8m);
        Percentile.Compute<decimal>(x, 1).Should().Be(36m);
        Percentile.Compute<decimal>([], .5).Should().Be(0m);
        Percentile.Compute<decimal>([9m], .125).Should().Be(9m);
        Percentile.Compute<decimal>([2m, 2m, 2m], .125).Should().Be(2m);
    }

    [Fact]
    public void Engine_ExactFinitePopulationSummary()
    {
        // All four samples from two equiprobable observations; sum statistics 4,10,10,16.
        var rng = new Script([0, 0, 0, 1, 1, 0, 1, 1]);
        var result = new BootstrapMonteCarloEngine<decimal>(4, rng)
            .Run([2m, 8m], s => s[0] + s[1]);
        result.Statistics.Should().Equal(4m, 10m, 10m, 16m);
        result.Mean.Should().Be(10m);
        result.Median.Should().Be(10m);
        result.Percentile5.Should().BeApproximately(4.9m, 1e-13m);
        result.Percentile95.Should().BeApproximately(15.1m, 1e-13m);
        rng.Bounds.Should().Equal(Enumerable.Repeat(2, 8));
    }

    private sealed class Script(int[] integers, double[]? doubles = null) : IRandomSource
    {
        private int _integer;
        public int DoubleCalls { get; private set; }
        public List<int> Bounds { get; } = [];
        public ulong NextULong() => throw new NotSupportedException();
        public int NextInt(int upperExclusive)
        {
            Bounds.Add(upperExclusive);
            return integers[_integer++ % integers.Length];
        }
        public double NextDouble() => doubles![DoubleCalls++ % doubles.Length];
    }
}
