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

public sealed class C2IndependentQrFactorTests
{
    [Fact]
    public void PivotedThreeByTwo_HasExactFactorActionStorageAndSolve()
    {
        double[,] a = { { 3, 3 }, { 0, 4 }, { 2, 0 } };
        double[] response = [3, 8, -2];
        var diagonal = new double[2];
        var permutation = new int[2];
        HouseholderQr<double>.Factor(a, response, diagonal, permutation).Should().BeTrue();
        var root = Math.Sqrt(61);
        permutation.Should().Equal(1, 0);
        diagonal[0].Should().BeApproximately(-5, 1e-14);
        diagonal[1].Should().BeApproximately(2 * root / 5, 1e-14);
        a[0, 1].Should().BeApproximately(-9d / 5, 1e-14);
        // Ratios identify the scale-free reflectors; raw and normalized storage
        // have identical action. v1=(1,1/2,0), v2=(1,5/(-6-sqrt61)).
        (a[1, 0] / a[0, 0]).Should().BeApproximately(.5, 1e-14);
        (a[2, 0] / a[0, 0]).Should().Be(0);
        (a[2, 1] / a[1, 1]).Should().BeApproximately(5 / (-6 - root), 1e-14);
        response[0].Should().BeApproximately(-41d / 5, 1e-14);
        response[1].Should().BeApproximately(-2 * root / 5, 1e-14);
        response[2].Should().BeApproximately(0, 1e-14);
        // Constructed independently from two planar reflections. The columns
        // are the orthonormal basis, not output from another factorization.
        double[,] expectedQ = { { -.6, 24 / (5 * root), -4 / root }, { -.8, -18 / (5 * root), 3 / root }, { 0, 5 / root, 6 / root } };
        for (var j = 0; j < 3; j++)
        {
            var unit = new double[3];
            unit[j] = 1;
            HouseholderQr<double>.ApplyQ(a, diagonal, unit);
            for (var i = 0; i < 3; i++) { unit[i].Should().BeApproximately(expectedQ[i, j], 1e-14); }
            var transposeUnit = new double[3];
            transposeUnit[j] = 1;
            HouseholderQr<double>.ApplyQTranspose(a, diagonal, transposeUnit);
            for (var i = 0; i < 3; i++) { transposeUnit[i].Should().BeApproximately(expectedQ[j, i], 1e-14); }
        }
        double[]? solution = null;
        FluentActions.Invoking(() => solution = HouseholderQr<double>.SolveUpperTriangular(a, diagonal, response, 2, permutation)).Should().NotThrow();
        solution![0].Should().BeApproximately(-1, 1e-14);
        solution[1].Should().BeApproximately(2, 1e-14);
    }

    [Fact]
    public void DependentAndZeroColumns_AreRejectedByRankPolicy()
    {
        double[,] near = { { 1, 1 }, { 1, 1 + Math.ScaleB(1, -51) } };
        HouseholderQr<double>.Factor(near, new double[2], new double[2], new int[2]).Should().BeFalse();
        HouseholderQr<double>.Factor(new double[2, 1], new double[2], new double[1], new int[1]).Should().BeFalse();
    }
}
