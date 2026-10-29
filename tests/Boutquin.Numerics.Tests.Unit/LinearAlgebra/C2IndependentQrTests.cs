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

public sealed class C2IndependentQrTests
{
    [Theory]
    [InlineData(1d)]
    [InlineData(1e-200)]
    [InlineData(1e-300)]
    [InlineData(1e200)]
    public void NonzeroTailWithZeroHead_HasAnalyticReflection(double scale)
    {
        double[,] a = { { 0 }, { 3 * scale }, { 4 * scale } };
        double[] response = [1, 0, 0];
        var diagonal = new double[1];
        HouseholderQr<double>.Factor(a, response, diagonal, new int[1]).Should().BeTrue();
        (diagonal[0] / scale).Should().BeApproximately(-5, 1e-14);
        // The reflector maps x to -5e0; its first column is (0,-3/5,-4/5).
        response[0].Should().BeApproximately(0, 1e-14);
        response[1].Should().BeApproximately(-.6, 1e-14);
        response[2].Should().BeApproximately(-.8, 1e-14);
    }

    [Fact]
    public void EqualNormIdentity_PreservesFirstPivotAndLapackIdentity()
    {
        double[,] a = { { 1, 0 }, { 0, 1 } };
        double[] response = [7, 11];
        var diagonal = new double[2];
        var permutation = new int[2];
        HouseholderQr<double>.Factor(a, response, diagonal, permutation).Should().BeTrue();
        permutation.Should().Equal(0, 1);
        // DLARFG uses identity when the tail is zero, preserving both diagonal and response.
        diagonal.Should().Equal(1, 1);
        response.Should().Equal(7, 11);
    }

    [Fact]
    public void InverseGram_PivotedDesign_HasIndependentRationalEntries()
    {
        double[,] a = { { 1, 0 }, { 1, 1 }, { 1, 2 } };
        var diagonal = new double[2];
        var permutation = new int[2];
        HouseholderQr<double>.Factor(a, new double[3], diagonal, permutation).Should().BeTrue();
        permutation.Should().Equal(1, 0);
        var inverse = HouseholderQr<double>.BuildXtXInverse(a, diagonal, permutation);
        // A'A=[[3,3],[3,5]], determinant6; adjugate divided by6.
        inverse[0, 0].Should().BeApproximately(5d / 6, 1e-14);
        inverse[0, 1].Should().BeApproximately(-.5, 1e-14);
        inverse[1, 0].Should().BeApproximately(-.5, 1e-14);
        inverse[1, 1].Should().BeApproximately(.5, 1e-14);
    }
}
