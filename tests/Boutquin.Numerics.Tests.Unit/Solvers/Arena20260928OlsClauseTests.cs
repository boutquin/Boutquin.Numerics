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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class Arena20260928OlsClauseTests
{
    [Fact]
    public void PivotedQr_OrdersOrthogonalColumnNormsAndRestoresOriginalCoefficients()
    {
        // Orthogonal Hadamard columns have norms2,8,4: pivot order is uniquely1,2,0.
        double[,] design = { { 1, 4, 2 }, { 1, -4, 2 }, { -1, 4, 2 }, { -1, -4, 2 } };
        double[] response = [12, 14, -8, 22]; // A*(3,-2,5) +7*(1,-1,-1,1).
        var qr = (double[,])design.Clone(); var transformed = (double[])response.Clone();
        var diagonal = new double[3]; var permutation = new int[3];
        HouseholderQr<double>.Factor(qr, transformed, diagonal, permutation).Should().BeTrue();
        permutation.Should().Equal(1, 2, 0);
        var coefficients = HouseholderQr<double>.SolveUpperTriangular(qr, diagonal, transformed, 3, permutation);
        var result = OrdinaryLeastSquares<double>.Fit(design, response, includeIntercept: false);
        double[] expected = [3, -2, 5];
        for (var i = 0; i < 3; i++)
        {
            coefficients[i].Should().BeApproximately(expected[i], 1e-13);
            result.Coefficients[i].Should().BeApproximately(expected[i], 1e-13);
        }
        result.ResidualSumOfSquares.Should().BeApproximately(196, 1e-12);
    }

    [Fact]
    public void ArbitraryAugmentedRightHandSides_RecoverBothIndependentBlocks()
    {
        // Construct f=r+A*x and g=A'*r from independent integer r,x.
        // Higham20.15 requires both returned blocks, not merely a least-squares x.
        double[,] design = { { 1, 4, 2 }, { 1, -4, 2 }, { -1, 4, 2 }, { -1, -4, 2 } };
        double[] expectedResidual = [2, 3, -1, 4]; double[] expectedX = [3, -2, 5];
        double[] f = [7, 24, -2, 19]; double[] g = [2, -24, 16];
        var diagonal = new double[3]; var permutation = new int[3];
        HouseholderQr<double>.Factor(design, new double[4], diagonal, permutation).Should().BeTrue();
        var h = HouseholderQr<double>.SolveUpperTriangularTranspose(design, diagonal, g, 3, permutation);
        HouseholderQr<double>.ApplyQTranspose(design, diagonal, f);
        var c = new double[3];
        for (var i = 0; i < 3; i++) { c[i] = f[i] - h[i]; f[i] = h[i]; }
        var x = HouseholderQr<double>.SolveUpperTriangular(design, diagonal, c, 3, permutation);
        HouseholderQr<double>.ApplyQ(design, diagonal, f);
        for (var i = 0; i < 4; i++) { f[i].Should().BeApproximately(expectedResidual[i], 1e-13); }
        for (var i = 0; i < 3; i++) { x[i].Should().BeApproximately(expectedX[i], 1e-13); }
    }

    [Fact]
    public void BjorckCompatibleInverseHilbert8_UsesExactIntegerInputConstruction()
    {
        // Bjorck1996 Example2.9.1: last6columns of inverseHilbert8, x_j=1/(j+3).
        // Integer inverse entries and b were evaluated by exact rational arithmetic.
        double[,] design = {
            {20160,-92400,221760,-288288,192192,-51480},
            {-952560,4656960,-11642400,15567552,-10594584,2882880},
            {11430720,-58212000,149688000,-204324120,141261120,-38918880},
            {-58212000,304920000,-800415000,1109908800,-776936160,216216000},
            {149688000,-800415000,2134440000,-2996753760,2118916800,-594594000},
            {-204324120,1109908800,-2996753760,4249941696,-3030051024,856215360},
            {141261120,-776936160,2118916800,-3030051024,2175421248,-618377760},
            {-38918880,216216000,-594594000,856215360,-618377760,176679360}};
        double[] response = [945, -40320, 456120, -2236080, 5599440, -7495488, 5105100, -1389960];
        var result = OrdinaryLeastSquares<double>.Fit(design, response, includeIntercept: false);
        for (var i = 0; i < 6; i++) { result.Coefficients[i].Should().Be(1.0 / (i + 3)); }
        // Exact rational accumulation of b-A*round_double(x) gives this nonzero RSS.
        // The mathematical x has zero residual, but its returned double representation does not.
        result.ResidualSumOfSquares.Should().BeApproximately(4.250326613145962e-15, 1e-25);
        result.DegreesOfFreedom.Should().Be(2);
    }

    [Fact]
    public void BjorckIncompatibleInverseHilbert8_RecoversPublishedSolutionAndResidualNorm()
    {
        // Bjorck-Golub Stanford report CS68-83, printed p25, second B column.
        // Bjorck1996 Example2.9.1 uses this problem; historical 36-bit iteration
        // digits are not an IEEE-double trajectory contract. Exact arithmetic gives
        // r_i=8400000/i, A'*r=0 and x_j=1/(j+3). The book's displayed r1 is r3.
        double[,] design = {
            {20160,-92400,221760,-288288,192192,-51480},
            {-952560,4656960,-11642400,15567552,-10594584,2882880},
            {11430720,-58212000,149688000,-204324120,141261120,-38918880},
            {-58212000,304920000,-800415000,1109908800,-776936160,216216000},
            {149688000,-800415000,2134440000,-2996753760,2118916800,-594594000},
            {-204324120,1109908800,-2996753760,4249941696,-3030051024,856215360},
            {141261120,-776936160,2118916800,-3030051024,2175421248,-618377760},
            {-38918880,216216000,-594594000,856215360,-618377760,176679360}};
        double[] response = [8400945, 4159680, 3256120, -136080, 7279440, -6095488, 6305100, -339960];
        var result = OrdinaryLeastSquares<double>.Fit(design, response, includeIntercept: false);

        for (var i = 0; i < 6; i++)
        {
            result.Coefficients[i].Should().BeApproximately(1.0 / (i + 3), 1e-13);
        }

        for (var i = 0; i < 8; i++)
        {
            result.Residuals[i].Should().BeApproximately(8400000.0 / (i + 1), 1e-6);
        }

        // Exact rational sum of (8400000/i)^2; the coefficient representation's
        // extra RSS is below one ULP at this magnitude, unlike the compatible case.
        result.ResidualSumOfSquares.Should().BeApproximately(107774900000000, 0.25);
        result.DegreesOfFreedom.Should().Be(2);
    }
}
