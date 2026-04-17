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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

public sealed class Random_ParityTests
{
    [Fact]
    public void Pcg64RandomSource_GenericMatchesLegacy_AtDouble()
    {
        var seed = 42ul;

        var legacy = new Pcg64RandomSource(seed);
        var generic = new Pcg64RandomSource<double>(seed);

        var legacyValue = legacy.NextULong();
        var genericValue = generic.NextULong();

        genericValue.Should().Be(legacyValue);
    }

    [Fact]
    public void Xoshiro256StarStarRandomSource_GenericMatchesLegacy_AtDouble()
    {
        var seed = 42ul;

        var legacy = new Xoshiro256StarStarRandomSource(seed);
        var generic = new Xoshiro256StarStarRandomSource<double>(seed);

        var legacyValue = legacy.NextULong();
        var genericValue = generic.NextULong();

        genericValue.Should().Be(legacyValue);
    }

    [Fact]
    public void GaussianSampler_GenericMatchesLegacy_AtDouble()
    {
        var seed = 42ul;

        var legacySource = new Pcg64RandomSource(seed);
        var genericSource = new Pcg64RandomSource<double>(seed);

        var legacy = new GaussianSampler(legacySource);
        var generic = new GaussianSampler<double>(genericSource);

        var legacyValue = legacy.Next();
        var genericValue = generic.Next();

        genericValue.Should().BeApproximately(legacyValue, 1e-12);
    }
}
