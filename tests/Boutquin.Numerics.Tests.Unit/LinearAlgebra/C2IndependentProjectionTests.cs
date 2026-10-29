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

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class C2IndependentProjectionTests
{
    [Fact]
    public void PrintedFourByFour_ConvergesToPublishedNearestCorrelation()
    {
        double[,] a = { { 2, -1, 0, 0 }, { -1, 2, -1, 0 }, { 0, -1, 2, -1 }, { 0, 0, -1, 2 } };
        double[,] printed = { { 1, -.8084, .1916, .1068 }, { -.8084, 1, -.6562, .1916 }, { .1916, -.6562, 1, -.8084 }, { .1068, .1916, -.8084, 1 } };
        double[,]? result = null;
        FluentActions.Invoking(() => result = NearestPsdProjection<double>.Higham(a, 1000, 1e-10)).Should().NotThrow();
        var squaredDistance = 0d;
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                result![i, j].Should().BeApproximately(printed[i, j], 5.01e-5);
                squaredDistance += Math.Pow(a[i, j] - result[i, j], 2);
            }
            result![i, i].Should().Be(1);
        }
        Math.Sqrt(squaredDistance).Should().BeApproximately(2.13, .005);
    }

    [Fact]
    public void PositiveFloor_RetainsLargerEigenvaluesAndRaisesOnlySmallerOnes()
    {
        decimal[,] a = { { -2, 0, 0 }, { 0, .25m, 0 }, { 0, 0, 3 } };
        var result = NearestPsdProjection<decimal>.EigenClip(a, .5m);
        decimal[,] expected = { { .5m, 0, 0 }, { 0, .5m, 0 }, { 0, 0, 3 } };
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++) { result[i, j].Should().BeApproximately(expected[i, j], 1e-25m); }
        }
    }

    [Fact]
    public void NumericalAcceptanceAndExactProjection_AreSeparateOperations()
    {
        decimal[,] a = { { -.0000000000005m, 0 }, { 0, 0 } };
        NearestPsdProjection<decimal>.IsPsd(a).Should().BeTrue();
        NearestPsdProjection<decimal>.IsPsd(a, 0).Should().BeFalse();
        var projected = NearestPsdProjection<decimal>.EigenClip(a);
        projected[0, 0].Should().Be(0);
        projected[1, 1].Should().Be(0);
    }

    [Fact]
    public void PositiveInput_FirstProjectionPreservesOffDiagonalBeforeUnitDiagonal()
    {
        // [[4,1],[1,4]] is positive definite, so its cone projection is itself.
        // One affine projection gives [[1,1],[1,1]], already PSD. A loose
        // tolerance allows precisely this first iterate to return.
        double[,] a = { { 4, 1 }, { 1, 4 } };
        double[,]? result = null;
        FluentActions.Invoking(() => result = NearestPsdProjection<double>.Higham(a, 1, 10)).Should().NotThrow();
        result![0, 0].Should().Be(1);
        result[1, 1].Should().Be(1);
        result[0, 1].Should().BeApproximately(1, 1e-14);
        result[1, 0].Should().BeApproximately(1, 1e-14);
    }
}
