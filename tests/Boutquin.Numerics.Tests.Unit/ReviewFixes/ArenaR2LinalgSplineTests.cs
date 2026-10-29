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

using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.LinearAlgebra;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Regression pins for Jacobi small-magnitude convergence, Jacobi input finiteness,
/// decimal overflow contracts in Cholesky / Gaussian elimination, and cubic-spline
/// abscissa scaling.
/// </summary>
public sealed class ArenaR2LinalgSplineTests
{
    // ── Jacobi: small-magnitude convergence ────────────────────────────────

    private static double[,] RandomSpd(int n, int seed, double scale)
    {
        var rng = new System.Random(seed);
        var b = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                b[i, j] = rng.NextDouble() - 0.5;
            }
        }

        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var k = 0; k < n; k++)
                {
                    s += b[i, k] * b[j, k];
                }

                a[i, j] = scale * ((s / n) + (i == j ? 0.1 : 0.0));
            }
        }

        return a;
    }

    private static double MaxReconstructionError(double[,] a, EigenResult<double> r)
    {
        var n = a.GetLength(0);
        var vectors = r.Vectors;
        var values = r.Values;
        var worst = 0.0;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var k = 0; k < n; k++)
                {
                    s += vectors[i, k] * values[k] * vectors[j, k];
                }

                worst = Math.Max(worst, Math.Abs(s - a[i, j]));
            }
        }

        return worst;
    }

    [Theory]
    [InlineData(1e-6)]
    [InlineData(1e-9)]
    [InlineData(1e-30)]
    public void Jacobi_LargeSmallMagnitudeMatrix_ConvergesAndReconstructs(double scale)
    {
        var a = RandomSpd(30, 17, scale);

        var act = () => JacobiEigenDecomposition<double>.Decompose(a);

        var result = act.Should().NotThrow().Subject;
        MaxReconstructionError(a, result).Should().BeLessThan(1e-10 * scale);
    }

    [Fact]
    public void Jacobi_DecimalSmallMagnitudeMatrix_Converges()
    {
        var source = RandomSpd(30, 5, 1e-7);
        var a = new decimal[30, 30];
        for (var i = 0; i < 30; i++)
        {
            for (var j = 0; j < 30; j++)
            {
                a[i, j] = (decimal)source[i, j];
            }
        }

        var act = () => JacobiEigenDecomposition<decimal>.Decompose(a);

        act.Should().NotThrow();
    }

    [Fact]
    public void PrincipalComponentAnalysis_LowVolatilityCovariance_Decomposes()
    {
        var a = RandomSpd(30, 23, 1e-7);

        var act = () => PrincipalComponentAnalysis<double>.Decompose(a);

        act.Should().NotThrow();
    }

    // ── Jacobi: non-finite input ───────────────────────────────────────────

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Jacobi_NonFiniteEntry_ThrowsArgumentException(double bad)
    {
        double[,] m = { { 1.0, bad }, { bad, 2.0 } };

        var act = () => JacobiEigenDecomposition<double>.Decompose(m);

        act.Should().Throw<ArgumentException>().WithMessage("*non-finite*");
    }

    [Fact]
    public void Jacobi_NonFiniteDiagonal_ThrowsArgumentException()
    {
        double[,] m = { { double.NaN, 0.5 }, { 0.5, 2.0 } };

        var act = () => JacobiEigenDecomposition<double>.Decompose(m);

        act.Should().Throw<ArgumentException>().WithMessage("*non-finite*");
    }

    [Fact]
    public void Jacobi_NonFiniteUnreadLowerTriangle_IsIgnored()
    {
        // The contract reads only the upper triangle, so a NaN below the diagonal is unused.
        double[,] m = { { 2.0, 1.0 }, { double.NaN, 2.0 } };

        var result = JacobiEigenDecomposition<double>.Decompose(m);

        result.Values[0].Should().BeApproximately(3.0, 1e-12);
        result.Values[1].Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void NearestPsdProjection_EigenClipAndIsPsd_NonFiniteEntry_ThrowArgumentException()
    {
        double[,] m = { { 1.0, double.NaN }, { double.NaN, 2.0 } };

        var clip = () => NearestPsdProjection<double>.EigenClip(m);
        var isPsd = () => NearestPsdProjection<double>.IsPsd(m, -1e-12);

        clip.Should().Throw<ArgumentException>().WithMessage("*non-finite*");
        isPsd.Should().Throw<ArgumentException>().WithMessage("*non-finite*");
    }

    // ── Gaussian elimination: back-substitution overflow ───────────────────

    [Fact]
    public void GaussianElimination_DoubleBackSubstitutionOverflow_ThrowsInvalidOperation()
    {
        double[,] a = { { 1.0, 1.0 }, { 1.0, 1.0 + 1e-13 } };
        double[] b = { 0.0, 1e300 };

        var act = () => GaussianElimination<double>.Solve(a, b, seed: 1);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void GaussianElimination_DecimalBackSubstitutionOverflow_ThrowsInvalidOperation()
    {
        decimal[,] a = { { 1m, 1m }, { 1m, 1m + 1e-13m } };
        decimal[] b = { 0m, 1e16m };

        var generic = () => GaussianElimination<decimal>.Solve(a, b, seed: 1);
        var legacy = () => GaussianElimination.Solve(a, b, seed: 1);

        generic.Should().Throw<InvalidOperationException>();
        legacy.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void GaussianElimination_DecimalWellConditioned_StillSolves()
    {
        decimal[,] a = { { 1m, 1m }, { 1m, 1m + 1e-13m } };
        decimal[] b = { 0m, 1m };

        var x = GaussianElimination<decimal>.Solve(a, b, seed: 1);

        (x[0] + x[1]).Should().Be(0m);
        (x[0] + ((1m + 1e-13m) * x[1])).Should().BeApproximately(1m, 1e-12m);
    }

    // ── Cholesky: decimal overflow on non-PD input ─────────────────────────

    [Fact]
    public void Cholesky_DecimalNonPdOverflow_ThrowsInvalidOperation()
    {
        decimal[,] a = { { 1e-10m, 1e20m }, { 1e20m, 1m } };

        var generic = () => CholeskyDecomposition<decimal>.Decompose(a);
        var legacy = () => CholeskyDecomposition.Decompose(a);

        generic.Should().Throw<InvalidOperationException>();
        legacy.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Cholesky_DecimalPivotedIndefiniteOverflow_ThrowsInvalidOperation()
    {
        decimal[,] a = { { 1m, 1e20m }, { 1e20m, 1m } };

        var generic = () => CholeskyDecomposition<decimal>.DecomposePivoted(a);
        var legacy = () => CholeskyDecomposition.DecomposePivoted(a);

        generic.Should().Throw<InvalidOperationException>();
        legacy.Should().Throw<InvalidOperationException>();
    }

    // ── Cubic spline: abscissa scaling ─────────────────────────────────────

    private static CubicSplineInterpolator<decimal> CubicOnScaledKnots(decimal h, SplineBoundary boundary)
    {
        var xs = new decimal[5];
        var ys = new decimal[5];
        for (var i = 0; i < 5; i++)
        {
            xs[i] = i * h;
            ys[i] = i * i * i;
        }

        return new CubicSplineInterpolator<decimal>(xs, ys, boundary);
    }

    [Theory]
    [InlineData(1e10)]
    [InlineData(1e14)]
    [InlineData(1e20)]
    public void CubicSpline_DecimalNotAKnot_WideSpacing_ReproducesCubic(double spacing)
    {
        var h = (decimal)spacing;
        var act = () => CubicOnScaledKnots(h, SplineBoundary.NotAKnot).Interpolate(1.5m * h);

        act.Should().NotThrow().Subject.Should().BeApproximately(3.375m, 1e-20m);
    }

    [Theory]
    [InlineData(1e8)]
    [InlineData(1e15)]
    public void CubicSpline_DecimalNatural_WideSpacing_MatchesUnitSpacing(double spacing)
    {
        var h = (decimal)spacing;
        var unit = CubicOnScaledKnots(1m, SplineBoundary.Natural).Interpolate(1.5m);
        var wide = CubicOnScaledKnots(h, SplineBoundary.Natural).Interpolate(1.5m * h);

        wide.Should().BeApproximately(unit, 1e-20m);
    }

    [Theory]
    [InlineData(1e155)]
    [InlineData(1e200)]
    public void CubicSpline_DoubleNotAKnotVariants_HugeSpacing_ReturnFiniteLinear(double h)
    {
        var xs = new[] { 0.0, h, 2 * h, 3 * h, 4 * h, 5 * h };
        var ys = new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };

        foreach (var boundary in new[] { SplineBoundary.NotAKnot, SplineBoundary.ModifiedNotAKnot, SplineBoundary.QSpline })
        {
            var spline = new CubicSplineInterpolator<double>(xs, ys, boundary);

            spline.Interpolate(1.5 * h).Should().BeApproximately(1.5, 1e-12, boundary.GetType().Name);
        }
    }

    [Fact]
    public void CubicSpline_DecimalClamped_WideSpacing_HonoursEndpointSlopes()
    {
        // y = (x/h)^2 has slope 0 at x=0 and 8/h at x=4h; a clamped spline reproduces it exactly.
        const decimal h = 1e12m;
        var xs = new decimal[5];
        var ys = new decimal[5];
        for (var i = 0; i < 5; i++)
        {
            xs[i] = i * h;
            ys[i] = i * i;
        }

        var spline = new CubicSplineInterpolator<decimal>(
            xs, ys, SplineBoundary<decimal>.CreateClamped(0m, 8m / h));

        spline.Interpolate(2.5m * h).Should().BeApproximately(6.25m, 1e-18m);
    }
}
