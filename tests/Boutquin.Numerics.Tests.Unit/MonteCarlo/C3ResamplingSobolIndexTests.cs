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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingSobolIndexTests
{
    [Theory]
    [InlineData(0L)]
    [InlineData(1L)]
    [InlineData(536870911L)]
    [InlineData(536870912L)]
    [InlineData(1073741822L)]
    [InlineData(1073741823L)]
    public void DirectSkip_FirstDimension_MatchesIndependentGrayDigitFraction(long index)
    {
        // Notes equation5: radical inverse of Gray digits. Accumulate dyadic
        // fractions directly, without using direction tables or recurrence.
        var gray = index ^ (index / 2);
        var fraction = 0.0;
        var place = 0.5;
        while (gray != 0)
        {
            fraction += (gray % 2) * place;
            gray /= 2;
            place /= 2;
        }

        new SobolSequence<double>(1, index).Next()[0].Should().Be(fraction);
    }

    [Fact]
    public void FinalAuthorIndex_EmitsExactDyadicPoint_ThenExhaustionIsSticky()
    {
        // Gray(2^30-1)=2^29, whose radical inverse is exactly2^-30.
        SobolSequence<double>? sequence = null;
        Action construct = () => sequence = new SobolSequence<double>(1, (1L << 30) - 1);
        construct.Should().NotThrow();
        sequence!.Next()[0].Should().Be(1.0 / 1073741824.0);
        Action next = () => sequence!.Next();
        next.Should().Throw<InvalidOperationException>();
        next.Should().Throw<InvalidOperationException>();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(17)]
    [InlineData(32)]
    public void DirectSkip_MatchesAuthorIndexedSequentialStream_AcrossDimensions(int dimension)
    {
        int[] indices = [0, 1, 2, 31, 63, 255, 1023];
        foreach (var index in indices)
        {
            var sequential = new SobolSequence<double>(dimension);
            for (var i = 0; i < index; i++)
            {
                sequential.Next();
            }

            var skipped = new SobolSequence<double>(dimension, index);
            for (var i = 0; i < 9; i++)
            {
                skipped.Next().Should().Equal(sequential.Next(),
                    "author index {0}, dimension {1}", index + i, dimension);
            }
        }
    }
}
