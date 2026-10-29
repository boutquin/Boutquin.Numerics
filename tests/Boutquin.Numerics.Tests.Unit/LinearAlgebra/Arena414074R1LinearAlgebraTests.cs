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

/// <summary>
/// Regression pins for the linear-algebra range defects: a requested eigenvalue floor
/// that is lost to normalization, standardized-PCA scale quantization, Jacobi rotation
/// overflow and pivot collapse, and a Cholesky retry that normalizes unit entries to zero.
/// </summary>
public sealed class Arena414074R1LinearAlgebraTests
{
    [Fact]
    public void EigenClip_ConnectedDecimalMixedScale_PreservesRequestedPositiveFloor()
    {
        // Every symmetric matrix whose eigenvalues are >= delta has each diagonal >= delta, so a
        // zero diagonal violates the requested floor without needing an eigenvalue oracle.
        const decimal floor = 1e-28m;
        var result = NearestPsdProjection<decimal>.EigenClip(
            new[,] { { 1e28m, 1e-28m }, { 1e-28m, 1e-28m } }, floor);

        result[0, 0].Should().Be(1e28m);
        result[1, 1].Should().BeGreaterThanOrEqualTo(floor);
        result[0, 1].Should().Be(result[1, 0]);
    }

    [Fact]
    public void EigenClip_ConnectedDoubleMixedScale_PreservesRequestedPositiveFloor()
    {
        const double floor = 1e-300;
        var result = NearestPsdProjection<double>.EigenClip(
            new[,] { { 1e300, 1e-300 }, { 1e-300, 1e-300 } }, floor);

        result[0, 0].Should().Be(1e300);
        result[1, 1].Should().BeGreaterThanOrEqualTo(floor);
    }

    [Fact]
    public void FromReturns_StandardizedTinyAmplitude_ProjectsOntoUnitCorrelation()
    {
        // The correlation matrix of one standardized column has diagonal one, so the only
        // eigenvalue is one and the two training rows score -/+ 1/sqrt(2). Squaring each
        // deviation (1.5e-14) on decimal's 1e-28 grid first made the scale 2e-14 instead of
        // sqrt(4.5e-28) = 2.12e-14. The stored scale 2.1213203435596425732e-14 itself keeps only
        // 14 digits on that grid, so the projected scores are checked to 1e-13.
        var rows = new[,] { { 0m }, { 3e-14m } };
        var fit = PrincipalComponentAnalysis.FromReturns(rows, standardize: true);

        fit.Eigenvalues[0].Should().BeApproximately(1m, 1e-20m);
        var scores = fit.Project(rows, 1);
        scores[0, 0].Should().BeApproximately(-0.7071067811865475244m, 1e-13m);
        scores[1, 0].Should().BeApproximately(0.7071067811865475244m, 1e-13m);
    }

    [Fact]
    public void Jacobi_OppositeNearMaximumDoubleRotation_ReturnsRepresentableSpectrum()
    {
        var scale = Math.ScaleB(1d, 1023);
        var g = 1.99 * Math.Sin(Math.PI / 8) * scale;
        var h = 1.99 * Math.Cos(Math.PI / 8) * scale;
        var e = Math.ScaleB(1d, 963);
        var a = new[,] { { 0d, e, g }, { e, 0d, h }, { g, h, 0d } };

        EigenResult<double>? eigen = null;
        var decompose = () => eigen = JacobiEigenDecomposition<double>.Decompose(a);

        // Equal diagonals force a 45-degree first rotation whose unscaled h + tau*g is infinite
        // although every rotated entry and the spectrum +/-1.99*2^1023 are representable.
        decompose.Should().NotThrow();
        (eigen!.Values[0] / scale).Should().BeApproximately(1.99, 1e-12);
        (eigen.Values[2] / scale).Should().BeApproximately(-1.99, 1e-12);
        Math.Abs(eigen.Values[1] / scale).Should().BeLessThan(1e-12);
    }

    [Fact]
    public void Jacobi_OppositeNearMaximumDecimalRotation_ReturnsRepresentableSpectrum()
    {
        const decimal g = 3e28m;
        const decimal h = 7e28m;
        var a = new[,] { { 0m, 1m, g }, { 1m, 0m, h }, { g, h, 0m } };

        EigenResult<decimal>? eigen = null;
        var decompose = () => eigen = JacobiEigenDecomposition<decimal>.Decompose(a);

        decompose.Should().NotThrow();
        var norm = Math.Sqrt(((double)g * (double)g) + ((double)h * (double)h));
        ((double)eigen!.Values[0] / norm).Should().BeApproximately(1d, 1e-12);
        ((double)eigen.Values[2] / norm).Should().BeApproximately(-1d, 1e-12);
    }

    [Fact]
    public void PcaDecompose_OppositeNearMaximumDoubleRotation_IsReachable()
    {
        var scale = Math.ScaleB(1d, 1023);
        var g = 1.99 * Math.Sin(Math.PI / 8) * scale;
        var h = 1.99 * Math.Cos(Math.PI / 8) * scale;
        var e = Math.ScaleB(1d, 963);
        var a = new[,] { { 0d, e, g }, { e, 0d, h }, { g, h, 0d } };

        PcaResult<double>? result = null;
        var decompose = () => result = PrincipalComponentAnalysis<double>.Decompose(a);

        decompose.Should().NotThrow();
        (result!.Eigenvalues[0] / scale).Should().BeApproximately(1.99, 1e-12);
    }

    [Fact]
    public void CholeskySolve_DecimalOverflowingPartialSum_ReturnsRepresentableSolution()
    {
        // A = L*L' with L = [I; 1 1 1 1]. For b = 7e28*(1,-1,-1,-1) the forward partial sum
        // -7e28 - 7e28 overflows decimal although the exact solution (7e28, -7e28, -7e28, 0)
        // is representable. The retry must not divide the unit factor entries by 7e28.
        var a = new[,]
        {
            { 1m, 0m, 0m, 1m },
            { 0m, 1m, 0m, 1m },
            { 0m, 0m, 1m, 1m },
            { 1m, 1m, 1m, 4m },
        };
        var lower = CholeskyDecomposition<decimal>.Decompose(a);

        decimal[]? x = null;
        var solve = () => x = CholeskyDecomposition<decimal>.Solve(lower, [7e28m, -7e28m, -7e28m, -7e28m]);

        solve.Should().NotThrow();
        x.Should().Equal(7e28m, -7e28m, -7e28m, 0m);
    }

    [Fact]
    public void Jacobi_DecimalOffDiagonalBelowNormalizationGrid_DecomposesWithoutDividingByZero()
    {
        // 1e-14 / 1e15 = 1e-29 rounds to zero on decimal's 1e-28 grid while the diagonal gap
        // 1e-13 / 1e15 = 1e-28 does not, so the rotation angle divided by a zero pivot.
        var a = new[,] { { 1e15m, 1e-14m }, { 1e-14m, 1e15m + 1e-13m } };

        EigenResult<decimal>? eigen = null;
        var decompose = () => eigen = JacobiEigenDecomposition<decimal>.Decompose(a);

        decompose.Should().NotThrow();
        eigen!.Values[0].Should().BeApproximately(1e15m + 1e-13m, 2e-13m);
        eigen.Values[1].Should().BeApproximately(1e15m, 2e-13m);
    }

    [Fact]
    public void Jacobi_DecimalNearMaximumDiagonalGapBelowGrid_DecomposesWithoutDividingByZero()
    {
        // 3 / 7e28 = 4.3e-29 rounds to zero; the exact eigenvalues are 7e28 + 1.24 and 7e28 - 7.24,
        // which decimal can resolve only to whole units at this magnitude.
        var a = new[,] { { 7e28m, 3m }, { 3m, 7e28m - 6m } };

        EigenResult<decimal>? eigen = null;
        var decompose = () => eigen = JacobiEigenDecomposition<decimal>.Decompose(a);

        decompose.Should().NotThrow();
        eigen!.Values[0].Should().BeApproximately(7e28m + 1.24m, 2m);
        eigen.Values[1].Should().BeApproximately(7e28m - 7.24m, 2m);
    }
}
