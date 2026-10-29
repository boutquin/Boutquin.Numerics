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

using Boutquin.Numerics.LinearAlgebra.Internal;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2LapackZeroTailTests
{
    [Theory]
    [InlineData(1d)]
    [InlineData(-1d)]
    [InlineData(1e-300)]
    [InlineData(-1e200)]
    public void OneEntryReflector_IsIdentityAndPreservesSignedAlpha(double alpha)
    {
        double[,] a = { { alpha } };
        double[] y = [7];
        var diagonal = new double[1];
        HouseholderQr<double>.Factor(a, y, diagonal, new int[1]).Should().BeTrue();
        diagonal[0].Should().Be(alpha);
        y.Should().Equal(7);
        double[] forward = [11];
        HouseholderQr<double>.ApplyQ(a, diagonal, forward);
        forward.Should().Equal(11);
        HouseholderQr<double>.ApplyQTranspose(a, diagonal, forward);
        forward.Should().Equal(11);
    }

    [Theory]
    [InlineData(2d)]
    [InlineData(-2d)]
    public void ZeroTailReflector_IsIdentityOnResponseAndBothQActions(double alpha)
    {
        double[,] a = { { alpha }, { 0 }, { 0 } };
        double[] y = [7, -3, 5];
        var diagonal = new double[1];
        HouseholderQr<double>.Factor(a, y, diagonal, new int[1]).Should().BeTrue();
        diagonal[0].Should().Be(alpha);
        y.Should().Equal(7, -3, 5);
        double[] forward = [11, 13, -17];
        HouseholderQr<double>.ApplyQ(a, diagonal, forward);
        forward.Should().Equal(11, 13, -17);
        HouseholderQr<double>.ApplyQTranspose(a, diagonal, forward);
        forward.Should().Equal(11, 13, -17);
    }
}
