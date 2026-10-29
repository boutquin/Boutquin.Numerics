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

[Trait("Tier", "Unit")]
public sealed class C3GaussianAuthorListingTests
{
    [Fact]
    public void AuthorListing_ReturnsSecondCoordinateThenStoredFirst()
    {
        var source = new UniformSource(.75, .625);
        var sampler = new GaussianSampler(source);
        // NR3 p365 stores v1*factor, returns v2*factor. Point(1/2,1/4)
        // has squared radius5/16; these constants were evaluated independently.
        sampler.Next().Should().BeApproximately(.68209993690241044975, 2e-15);
        source.Reads.Should().Be(2);
        sampler.Next().Should().BeApproximately(1.3641998738048208995, 3e-15);
        source.Reads.Should().Be(2);
    }

    [Fact]
    public void AuthorListing_ZeroStoredCoordinateCausesFreshPair()
    {
        var source = new UniformSource(.5, .75, .75, .625);
        var sampler = new GaussianSampler(source);
        // First point(0,1/2) storeszero and returns sqrt(log16).
        sampler.Next().Should().BeApproximately(1.6651092223153955127, 3e-15);
        source.Reads.Should().Be(2);
        sampler.Next().Should().BeApproximately(.68209993690241044975, 2e-15);
        source.Reads.Should().Be(4);
    }

    [Fact]
    public void AuthorListing_StoredValueIsConsumedOnce()
    {
        var source = new UniformSource(.75, .625, .25, .625);
        var sampler = new GaussianSampler(source);
        sampler.Next().Should().BeApproximately(.68209993690241044975, 2e-15);
        sampler.Next().Should().BeApproximately(1.3641998738048208995, 3e-15);
        sampler.Next().Should().BeApproximately(.68209993690241044975, 2e-15);
        source.Reads.Should().Be(4);
        sampler.Next().Should().BeApproximately(-1.3641998738048208995, 3e-15);
        source.Reads.Should().Be(4);
    }

    private sealed class UniformSource(params double[] values) : IRandomSource
    {
        public int Reads { get; private set; }
        public ulong NextULong() => throw new NotSupportedException();
        public double NextDouble() => values[Reads++];
    }
}
