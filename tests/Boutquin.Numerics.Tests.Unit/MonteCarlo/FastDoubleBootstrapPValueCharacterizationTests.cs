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
/// Strict-tail characterization harness for <see cref="FastDoubleBootstrap.ComputePValue"/>.
/// Pins the exact tie / boundary counting contract of the p-value computation so the
/// binary-search implementation excludes equality. Every expected value is computed
/// by hand from strict tail counts (rightCount = #{x > target}, leftCount = #{x &lt; target}),
/// giving an independent oracle rather than a snapshot of current output.
/// </summary>
public sealed class FastDoubleBootstrapPValueCharacterizationTests
{
    // Case A — target present with a tie run in the middle.
    // dist = [1,2,2,2,3,4], target = 2 → #{>2}=2, #{<2}=1, length 6.
    [Theory]
    [InlineData(PValueTail.RightTail, 2.0, 6.0)]
    [InlineData(PValueTail.LeftTail, 1.0, 6.0)]
    public void ComputePValue_TieRunMiddle_MatchesStrictTail(PValueTail tail, double num, double den)
    {
        var dist = new[] { 1m, 2m, 2m, 2m, 3m, 4m };
        var expected = (decimal)num / (decimal)den;
        FastDoubleBootstrap.ComputePValue(dist, 2m, tail).Should().Be(expected);
    }

    [Fact]
    public void ComputePValue_TieRunMiddle_TwoSided_ExcludesTies()
    {
        // min(2·min(2/6,1/6),1) = 1/3.
        var dist = new[] { 1m, 2m, 2m, 2m, 3m, 4m };
        FastDoubleBootstrap.ComputePValue(dist, 2m, PValueTail.TwoSided).Should().Be(2m * (1m / 6m));
    }

    // Case B — target absent, strictly between two elements. dist=[1,3,5,7], target=4.
    // #{>4}=2, #{<4}=2, length 4 → both 0.5.
    [Theory]
    [InlineData(PValueTail.RightTail)]
    [InlineData(PValueTail.LeftTail)]
    public void ComputePValue_AbsentBetween_IsHalf(PValueTail tail)
    {
        var dist = new[] { 1m, 3m, 5m, 7m };
        FastDoubleBootstrap.ComputePValue(dist, 4m, tail).Should().Be(0.5m);
    }

    // Case C — target below every element. dist=[2,4,6], target=1.
    [Fact]
    public void ComputePValue_BelowAll_RightIsOne_LeftIsZero()
    {
        var dist = new[] { 2m, 4m, 6m };
        FastDoubleBootstrap.ComputePValue(dist, 1m, PValueTail.RightTail).Should().Be(1m);
        FastDoubleBootstrap.ComputePValue(dist, 1m, PValueTail.LeftTail).Should().Be(0m);
        FastDoubleBootstrap.ComputePValue(dist, 1m, PValueTail.TwoSided).Should().Be(0m);
    }

    // Case D — target above every element. dist=[2,4,6], target=9.
    [Fact]
    public void ComputePValue_AboveAll_RightIsZero_LeftIsOne()
    {
        var dist = new[] { 2m, 4m, 6m };
        FastDoubleBootstrap.ComputePValue(dist, 9m, PValueTail.RightTail).Should().Be(0m);
        FastDoubleBootstrap.ComputePValue(dist, 9m, PValueTail.LeftTail).Should().Be(1m);
        FastDoubleBootstrap.ComputePValue(dist, 9m, PValueTail.TwoSided).Should().Be(0m);
    }

    // Case E — tie run at the low boundary. dist=[2,2,5], target=2 → #{>2}=1, #{<2}=0.
    [Fact]
    public void ComputePValue_TieAtLowBoundary()
    {
        var dist = new[] { 2m, 2m, 5m };
        FastDoubleBootstrap.ComputePValue(dist, 2m, PValueTail.RightTail).Should().Be(1m / 3m);
        FastDoubleBootstrap.ComputePValue(dist, 2m, PValueTail.LeftTail).Should().Be(0m);
        FastDoubleBootstrap.ComputePValue(dist, 2m, PValueTail.TwoSided).Should().Be(0m);
    }

    // Case F — tie run at the high boundary. dist=[1,5,5], target=5 → #{>5}=0, #{<5}=1.
    [Fact]
    public void ComputePValue_TieAtHighBoundary()
    {
        var dist = new[] { 1m, 5m, 5m };
        FastDoubleBootstrap.ComputePValue(dist, 5m, PValueTail.RightTail).Should().Be(0m);
        FastDoubleBootstrap.ComputePValue(dist, 5m, PValueTail.LeftTail).Should().Be(1m / 3m);
        FastDoubleBootstrap.ComputePValue(dist, 5m, PValueTail.TwoSided).Should().Be(0m);
    }

    // Case G — single element exactly equal.
    [Fact]
    public void ComputePValue_SingleElementEqual()
    {
        var dist = new[] { 5m };
        FastDoubleBootstrap.ComputePValue(dist, 5m, PValueTail.RightTail).Should().Be(0m);
        FastDoubleBootstrap.ComputePValue(dist, 5m, PValueTail.LeftTail).Should().Be(0m);
        FastDoubleBootstrap.ComputePValue(dist, 5m, PValueTail.TwoSided).Should().Be(0m);
    }

    // End-to-end determinism: the public PValue seam must reproduce bit-identically
    // for the same fixed seed.
    [Fact]
    public void PValue_FixedSeed_IsReproducible()
    {
        var observations = Enumerable.Range(1, 40).Select(i => (decimal)i).ToArray();
        static decimal Mean(ReadOnlySpan<decimal> s)
        {
            var sum = 0m;
            foreach (var v in s)
            {
                sum += v;
            }

            return sum / s.Length;
        }

        var a = new FastDoubleBootstrap(200, new Pcg64RandomSource(20260707UL))
            .PValue(observations, 25m, Mean);
        var b = new FastDoubleBootstrap(200, new Pcg64RandomSource(20260707UL))
            .PValue(observations, 25m, Mean);

        a.Should().Be(b);
        a.Should().BeInRange(0m, 1m);
    }
}
