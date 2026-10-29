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

public sealed class C3SobolAuthorOriginTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(32)]
    public void AuthorIndexZero_IsEmittedInEverySelectedDimension(int dimension)
    {
        // Joe-Kuo Notes eq5 and author POINTS[0]: origin in every dimension.
        new SobolSequence<double>(dimension).Next().Should().Equal(new double[dimension]);
    }

    [Fact]
    public void AuthorGrayBlock_IncludesOriginAndAllEightDyadicCells()
    {
        // Notes section2: radical inverse of i XOR(i>>1), i=0..7.
        double[] expected = [0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];
        var sequence = new SobolSequence<double>(1);
        foreach (var point in expected)
        {
            sequence.Next()[0].Should().Be(point);
        }
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 0.5)]
    [InlineData(3, 0.25)]
    public void SkipCountsAuthorIndicesFromZero(int skip, double expected)
    {
        new SobolSequence<double>(1, skip).Next()[0].Should().Be(expected);
    }

    [Fact]
    public void RejectedDestination_DoesNotConsumeAuthorOrigin()
    {
        var sequence = new SobolSequence<double>(2);
        Action shortDestination = () => sequence.Next(new double[1]);
        shortDestination.Should().Throw<ArgumentException>();
        sequence.Next().Should().Equal(0, 0);
    }
}
