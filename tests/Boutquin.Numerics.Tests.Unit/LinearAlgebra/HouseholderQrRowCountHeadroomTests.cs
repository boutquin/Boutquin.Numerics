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
using Boutquin.Numerics.Solvers;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

/// <summary>
/// The overflow headroom factor of the Householder QR path scales with the row count and must stay
/// positive and exact for every row count inside the supported length domain (below 2^30), where a
/// 32-bit product would wrap negative from 2^29 rows. Expected values are the exact 64-bit products.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class HouseholderQrRowCountHeadroomTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(1000)]
    [InlineData((1 << 29) - 1)]
    [InlineData(1 << 29)]
    [InlineData((1 << 30) - 1)]
    public void RowHeadroomFactor_IsExactAndPositive_ForEveryRowCountBelowTwoToThe30(int rows)
    {
        long expected = 4L * rows;

        HouseholderQr<double>.RowHeadroomFactor(rows).Should().Be(expected);
        HouseholderQr<decimal>.RowHeadroomFactor(rows).Should().Be(expected);
        expected.Should().BePositive();
    }

    [Fact]
    public void Fit_ZeroResponseWithModerateRows_StillReturnsZeroCoefficient()
    {
        var x = new double[1000, 1];
        var y = new double[1000];
        for (var i = 0; i < 1000; i++)
        {
            x[i, 0] = 1.0;
        }

        var result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false);

        result.Coefficients[0].Should().Be(0.0);
    }
}
