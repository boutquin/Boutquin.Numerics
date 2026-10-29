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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Tests for additive APIs from the 2026-06-20 review (findings #25, #36).
/// </summary>
public sealed class AdditiveApiTests
{
    [Fact]
    public void Halton_NextSpan_MatchesAllocatingNext()
    {
        var allocating = new HaltonSequence<double>(dimension: 3);
        var spanned = new HaltonSequence<double>(dimension: 3);
        Span<double> actual = stackalloc double[3];

        for (var k = 0; k < 5; k++)
        {
            var expected = allocating.Next();
            spanned.Next(actual);
            for (var d = 0; d < 3; d++)
            {
                actual[d].Should().Be(expected[d]);
            }
        }
    }

    [Fact]
    public void Halton_NextSpan_TooShort_Throws()
    {
        var seq = new HaltonSequence<double>(dimension: 3);
        var act = () =>
        {
            Span<double> tiny = stackalloc double[2];
            seq.Next(tiny);
        };
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Sobol_NextSpan_MatchesAllocatingNext()
    {
        var allocating = new SobolSequence<double>(dimension: 4);
        var spanned = new SobolSequence<double>(dimension: 4);
        Span<double> actual = stackalloc double[4];

        for (var k = 0; k < 5; k++)
        {
            var expected = allocating.Next();
            spanned.Next(actual);
            for (var d = 0; d < 4; d++)
            {
                actual[d].Should().Be(expected[d]);
            }
        }
    }

    [Fact]
    public void FisherZ_GenericConfidenceInterval_DefaultsTo95Percent()
    {
        var withDefault = FisherZTransform<double>.ConfidenceInterval(0.5, 30);
        var explicit95 = FisherZTransform<double>.ConfidenceInterval(0.5, 30, 0.95);
        withDefault.Lower.Should().Be(explicit95.Lower);
        withDefault.Upper.Should().Be(explicit95.Upper);
    }
}
