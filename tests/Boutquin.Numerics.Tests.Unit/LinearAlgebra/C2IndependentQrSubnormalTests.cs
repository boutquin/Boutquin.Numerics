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

public sealed class C2IndependentQrSubnormalTests
{
    [Theory]
    [InlineData(-1030)]
    [InlineData(-1070)]
    public void SubnormalPythagoreanColumn_PreservesNormAndOrthogonalAction(int exponent)
    {
        // Exact binary subnormal coordinates 3s,4s and representable norm5s.
        // Analytic H=[[-3/5,-4/5],[-4/5,3/5]], independent of s.
        var scale = Math.ScaleB(1, exponent);
        double[,] a = { { 3 * scale }, { 4 * scale } };
        double[] response = [1, 0];
        var diagonal = new double[1];
        HouseholderQr<double>.Factor(a, response, diagonal, new int[1]).Should().BeTrue();
        diagonal[0].Should().Be(-5 * scale);
        response[0].Should().BeApproximately(-.6, 2e-15);
        response[1].Should().BeApproximately(-.8, 2e-15);
        double[] vector = [7, 11];
        HouseholderQr<double>.ApplyQTranspose(a, diagonal, vector);
        vector[0].Should().BeApproximately(-13, 1e-14);
        vector[1].Should().BeApproximately(1, 1e-14);
        HouseholderQr<double>.ApplyQ(a, diagonal, vector);
        vector[0].Should().BeApproximately(7, 2e-14);
        vector[1].Should().BeApproximately(11, 2e-14);
    }
}
