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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3RemainingPredicateControls
{
    [Fact]
    public void BoundedInteger_ZeroBoundIsExcludedBeforeDrawing()
    {
        var words = new MaximumWords();
        Action draw = () => ((IRandomSource)words).NextInt(0);
        draw.Should().Throw<ArgumentOutOfRangeException>();
        words.Reads.Should().Be(0);
        ((IRandomSource)words).NextInt(1).Should().Be(0);
        words.Reads.Should().Be(1);
    }

    [Fact]
    public void DistanceCorrelation_RequiresPairedEqualLengthObservations()
    {
        Action evaluate = () => DistanceCorrelation<double>.Compute([0, 1, 5], [0, 2]);
        evaluate.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void SubsampleRawValues_ComposeInclusiveCenteredEmpiricalCdf()
    {
        // Independently expanded weighted sums are9,15,33; the full-sample
        // value is70. At scale2 the threshold-110 includes exactly the first
        // two of all three observations, including the equality at15.
        var values = Subsampler.Run<decimal>([1m, 7m, -2m, 4m, 9m], 3,
            block => block[0] + 2m * block[1] + 3m * block[2]);
        values.Should().Equal(9m, 15m, 33m);
        var cumulativeProbability = (decimal)values.Count(value => 2m * (value - 70m) <= -110m) / values.Length;
        cumulativeProbability.Should().Be(2m / 3m);
    }

    private sealed class MaximumWords : IRandomSource
    {
        public int Reads { get; private set; }
        public ulong NextULong()
        {
            Reads++;
            return ulong.MaxValue;
        }
    }
}
