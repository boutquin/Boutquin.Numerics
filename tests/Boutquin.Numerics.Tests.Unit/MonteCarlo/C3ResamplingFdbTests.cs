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

public sealed class C3ResamplingFdbTests
{
    [Fact]
    public void Fdb_NullDgpsAndPairing_ExactCallbackTrace()
    {
        decimal[] original = [2m, 8m];
        var nullCalls = new List<decimal[]>();
        var statisticCalls = new List<decimal[]>();
        var rng = new Indices();
        _ = new FastDoubleBootstrap(50, rng).PValueImposingNull(original, 99m, s =>
        {
            statisticCalls.Add(s.ToArray());
            return (s[0] + s[1]) / 2;
        }, s =>
        {
            nullCalls.Add(s.ToArray());
            var mean = (s[0] + s[1]) / 2;
            s[0] -= mean;
            s[1] -= mean;
        });
        original.Should().Equal(2m, 8m);
        nullCalls.Count.Should().Be(51);
        statisticCalls.Count.Should().Be(100);
        nullCalls[0].Should().Equal(2m, 8m);
        for (var b = 0; b < 50; b++)
        {
            nullCalls[b + 1].Should().Equal(b % 2 == 0 ? [-3m, -3m] : [-3m, 3m]);
            statisticCalls[2 * b].Should().Equal(b % 2 == 0 ? [-3m, -3m] : [-3m, 3m]);
            statisticCalls[2 * b + 1].Should().Equal(b % 2 == 0 ? [0m, 0m] : [3m, 3m]);
        }
        rng.Bounds.Should().Equal(Enumerable.Repeat(2, 200));
    }

    [Fact]
    public void Fdb_QuantilesEndpointsAndFold_ExactFiniteRanks()
    {
        // Distinct outer0..100; inner=-20..80. Observed89.5: first right11/101,
        // q=6900/101+? precisely 100*90/101-20=6980/101, final31/101.
        Evaluate(PValueTail.RightTail, 89.5m, false).Should().BeApproximately(31m / 101, 1e-25m);
        // p*=0 => max80 and20 outer values strictly greater. p*=1 => min-20 and101 greater.
        Evaluate(PValueTail.RightTail, 1000m, false).Should().BeApproximately(20m / 101, 1e-25m);
        Evaluate(PValueTail.RightTail, -1000m, false).Should().Be(1m);
        // Left p*=11/101, inner20..120 gives q=3120/101 and31/101 below.
        Evaluate(PValueTail.LeftTail, 10.5m, false).Should().BeApproximately(31m / 101, 1e-25m);
        // Tied outer1,2,2,2,3,4, constant inner2. Final right2/6,left1/6,
        // selected two-sided fold1/3. Neither one-sided tail is its complement.
        Evaluate(PValueTail.TwoSided, 2m, true).Should().BeApproximately(1m / 3, 1e-25m);
    }

    [Fact]
    public void Fdb_TypeSevenInterpolation_StraddlesOuterAtom()
    {
        // Outer30 copies of-1 and30 of1/4; observed0 gives both first tails1/2.
        // Inner30 copies0 and30 copies1 => Type7 median1/2.
        // Exact final ranks: right0,left1. Lower-order-statistic quantile0 gives1/2,1/2.
        foreach (var tail in new[] { PValueTail.RightTail, PValueTail.LeftTail })
        {
            var call = 0;
            var actual = new FastDoubleBootstrap(60, new Indices()).PValue([1m, 2m], 0m, _ =>
            {
                var b = call / 2;
                var inner = call++ % 2 == 1;
                return inner ? (b < 30 ? 0m : 1m) : (b < 30 ? -1m : .25m);
            }, tail);
            actual.Should().Be(tail == PValueTail.RightTail ? 0m : 1m);
        }
    }

    private static decimal Evaluate(PValueTail tail, decimal observed, bool ties)
    {
        var call = 0;
        decimal[] pattern = [1, 2, 2, 2, 3, 4];
        var count = ties ? 60 : 101;
        return new FastDoubleBootstrap(count, new Indices()).PValue([1m, 2m], observed, _ =>
        {
            var b = call / 2;
            var inner = call++ % 2 == 1;
            return ties ? (inner ? 2m : pattern[b % 6])
                : b + (inner ? (tail == PValueTail.LeftTail ? 20m : -20m) : 0m);
        }, tail);
    }

    private sealed class Indices : IRandomSource
    {
        private int _call;
        private readonly int[] _indices = [0, 0, 0, 1, 0, 1, 1, 1];
        public List<int> Bounds { get; } = [];
        public ulong NextULong() => throw new NotSupportedException();
        public int NextInt(int upperExclusive)
        {
            Bounds.Add(upperExclusive);
            return _indices[_call++ % _indices.Length];
        }
    }
}
