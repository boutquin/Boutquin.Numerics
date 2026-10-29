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

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingGaussianCurrentTests
{
    [Fact]
    public void CurrentPolar_RejectsOrigin() => AssertRejected(.5, .5);

    [Fact]
    public void CurrentPolar_RejectsUnitCircle() => AssertRejected(0, .5);

    [Fact]
    public void CurrentPolar_RejectsOutsideDisk() => AssertRejected(0, 0);

    private static void AssertRejected(double first, double second)
    {
        var source = new Uniforms(first, second, .25, .625);
        var sampler = new GaussianSampler(source);
        // Accepted u=-1/2,v=1/4,s=5/16; NR3 returns v*factor first.
        sampler.Next().Should().BeApproximately(.68209993690241044975, 2e-15);
        source.Reads.Should().Be(4);
        sampler.Next().Should().BeApproximately(-1.3641998738048208995, 3e-15);
        source.Reads.Should().Be(4);
    }

    [Fact]
    public void CurrentPolar_OrderedSignedPair_ExactIndependentValues()
    {
        var source = new Uniforms(.25, .625);
        var sampler = new GaussianSampler(source);
        // Squared radius5/16, factor=sqrt((32/5)*log(16/5)).
        sampler.Next().Should().BeApproximately(.68209993690241044975, 2e-15);
        sampler.Next().Should().BeApproximately(-1.3641998738048208995, 3e-15);
        source.Reads.Should().Be(2);
    }

    [Fact]
    public void CurrentPolar_AffineTransform_UsesReturnedSecondCoordinate()
    {
        // mean3, sd4; v*factor=.68209993690241044975.
        new GaussianSampler(new Uniforms(.25, .625)).Next(3, 4)
            .Should().BeApproximately(5.7283997476096417990, 4e-15);
    }

    private sealed class Uniforms(params double[] values) : IRandomSource
    {
        public int Reads { get; private set; }
        public ulong NextULong() => throw new NotSupportedException();
        public double NextDouble() => values[Reads++];
    }
}
