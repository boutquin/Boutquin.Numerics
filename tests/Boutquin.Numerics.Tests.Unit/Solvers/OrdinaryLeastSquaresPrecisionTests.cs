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

/// <summary>
/// Precision and scale contracts for <see cref="OrdinaryLeastSquares{T}"/> at
/// <c>T = double</c> and its internal <see cref="HouseholderQr{T}"/>. Every expected
/// value here comes from outside the implementation: OLS scale equivariance, the
/// closed-form single-regressor slope, an exactly consistent system, or an integer
/// system whose least-squares solution is known by construction.
/// </summary>
public sealed class OrdinaryLeastSquaresPrecisionTests
{
    [Fact]
    public void DecimalCovarianceOverflow_PreservesTheRangeException()
    {
        decimal[,] x = { { 1e-12m }, { 2e-12m }, { 3e-12m }, { 4e-12m } };
        decimal[] y = [1e5m, -1e5m, 3e5m, 1e5m];
        // This one-column design has condition number 1; covariance = 8e32 is
        // unrepresentable, so the correct refusal reports range rather than rank.
        FluentActions.Invoking(() => OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept: false))
            .Should().Throw<InvalidOperationException>().WithInnerException<OverflowException>();
    }

    [Fact]
    public void OverflowingCovariance_HasSignedInfinityInsteadOfNaN()
    {
        // All inputs are in the double operating domain. Exact rational inversion of
        // X'X gives signs [[+,-,-],[-,+,-],[-,-,+]]; positive sigma² preserves them.
        double[,] x = {
            { 1.1174694074812397e-110, -1.539578853041844e-41, 1.1092609088918017e-30 },
            { -2.0125363673935848e-111, -8.058776402446577e-41, 1.344911138697641e-31 },
            { 1.0988646282195039e-110, 5.371329926133317e-41, -6.3760448782210874e-31 },
            { 1.5117272462333298e-111, -2.7468283260259297e-40, -4.74850008839748e-31 }
        };
        double[] y = [5.373438437092569e129, -1.8492655357336015e130, -1.8453191793471456e130, 1.8478109004476085e128];
        var fit = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false);
        for (var i = 0; i < 3; i++)
        {
            double.IsFinite(fit.StandardErrors[i]).Should().BeTrue();
            for (var j = 0; j < 3; j++)
            {
                fit.CovarianceMatrix[i, j].Should().Be(i == j ? double.PositiveInfinity : double.NegativeInfinity);
            }
        }
    }

    [Theory]
    [InlineData("0.0001", false)]
    [InlineData("0.0000000001", false)]
    [InlineData("0.0001", true)]
    public void DecimalDependentColumns_AreRejectedAtSmallScales(string scaleText, bool includeIntercept)
    {
        // The third column is the exact sum of the first two, at every scale.
        var scale = decimal.Parse(scaleText, System.Globalization.CultureInfo.InvariantCulture);
        var random = new System.Random(7342);
        for (var trial = 0; trial < 30; trial++)
        {
            var x = new decimal[12, 3];
            var y = new decimal[12];
            for (var i = 0; i < 12; i++)
            {
                x[i, 0] = random.Next(-99, 99) * scale;
                x[i, 1] = random.Next(-99, 99) * scale;
                x[i, 2] = x[i, 0] + x[i, 1];
                y[i] = x[i, 0] + 2 * x[i, 1];
            }

            FluentActions.Invoking(() => OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept))
                .Should().Throw<InvalidOperationException>().WithMessage("*rank-deficient*");
        }
    }

    [Fact]
    public void DecimalIndependentTinyColumns_KeepTheirRepresentableCoefficients()
    {
        decimal[,] x = { { 1e-28m, 0 }, { 0, 1e-28m }, { 1e-28m, 1e-28m } };
        decimal[] y = [2e-28m, 3e-28m, 5e-28m];
        var fit = OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept: false);
        fit.Coefficients[0].Should().BeApproximately(2m, 1e-25m);
        fit.Coefficients[1].Should().BeApproximately(3m, 1e-25m);
    }

    // Noisy straight line y = 1.7 + 2.3·x + e with a deterministic, non-trivial e.
    private static (double[,] X, double[] Y) NoisyLine(int m, double scale)
    {
        var x = new double[m, 1];
        var y = new double[m];
        for (var i = 0; i < m; i++)
        {
            var xi = i + 1.0;
            var noise = 0.005 * Math.Sin(1.3 * (i + 1));
            x[i, 0] = xi * scale;
            y[i] = (1.7 + (2.3 * xi) + noise) * scale;
        }

        return (x, y);
    }

    [Theory]
    [InlineData(-40)]
    [InlineData(-50)]
    public void SmallScale_ResidualSumOfSquares_ScalesWithSquareOfDataScale(int log2Scale)
    {
        // OLS is scale-equivariant: multiplying x and y by s (a power of two, so the
        // scaled inputs are exact) multiplies RSS by s² and leaves the slope's standard
        // error unchanged. The unit-scale fit is the independent expected value.
        var scale = Math.ScaleB(1.0, log2Scale);
        var (x1, y1) = NoisyLine(20, 1.0);
        var (xs, ys) = NoisyLine(20, scale);

        var unit = OrdinaryLeastSquares<double>.Fit(x1, y1);
        var scaled = OrdinaryLeastSquares<double>.Fit(xs, ys);

        (scaled.ResidualSumOfSquares / (scale * scale))
            .Should().BeApproximately(unit.ResidualSumOfSquares, 1e-9 * unit.ResidualSumOfSquares);
        scaled.StandardErrors[1]
            .Should().BeApproximately(unit.StandardErrors[1], 1e-9 * unit.StandardErrors[1]);
    }

    [Fact]
    public void SmallScale_ResidualSumOfSquares_EqualsSumOfSquaredResiduals()
    {
        // At s = 1e-13 the residuals are ~1e-16; RSS must still be Σ rᵢ² of the
        // returned residual vector, and the design has 48 residual degrees of freedom,
        // so a zero RSS (and a zero standard error) is never the correct answer.
        var random = new System.Random(20260927);
        const int m = 50;
        const double s = 1e-13;
        var x = new double[m, 1];
        var y = new double[m];
        for (var i = 0; i < m; i++)
        {
            var xi = 0.5 + random.NextDouble();
            x[i, 0] = xi;
            y[i] = s * ((3.0 * xi) + (0.01 * ((2.0 * random.NextDouble()) - 1.0)));
        }

        var result = OrdinaryLeastSquares<double>.Fit(x, y);

        var sumSquares = 0.0;
        foreach (var r in result.Residuals)
        {
            sumSquares += r * r;
        }

        result.DegreesOfFreedom.Should().Be(48);
        sumSquares.Should().BeGreaterThan(0.0);
        result.ResidualSumOfSquares.Should().BeApproximately(sumSquares, 1e-10 * sumSquares);
        result.StandardErrors[1].Should().BeGreaterThan(0.0);
    }

    [Fact]
    public void FullMantissaConsistentData_RecoversSlopeToOneUlp()
    {
        // y = 2·x is exact in double for every x, so the least-squares slope of the
        // double data is exactly 2 and the refined estimate must land within one ulp.
        var random = new System.Random(4000);
        const int m = 4000;
        var x = new double[m, 1];
        var y = new double[m];
        for (var i = 0; i < m; i++)
        {
            var xi = 0.5 + random.NextDouble();
            x[i, 0] = xi;
            y[i] = 2.0 * xi;
        }

        var result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false);

        Math.Abs(result.Coefficients[0] - 2.0).Should().BeLessThanOrEqualTo(Math.BitIncrement(2.0) - 2.0);
        // The true residuals are exactly zero; perturbed inputs leave ~1e-25.
        result.ResidualSumOfSquares.Should().BeLessThan(1e-40);
    }

    /// <summary>
    /// Builds an inconsistent, ill-conditioned integer system whose least-squares
    /// solution is exactly <c>(3, -2)</c>: columns <c>a</c> and <c>a + d</c> with
    /// <c>a ≈ 1e8</c> and <c>d ∈ {-3..3}</c> (κ₂ ≈ 1e8), plus residual vectors
    /// <c>u × d</c> on disjoint row triples, each orthogonal to both columns. Every
    /// entry is an integer below 2^53, so the double data is the exact problem.
    /// </summary>
    internal static (double[,] X, double[] Y) InconsistentIntegerSystem(int seed, long magnitude)
    {
        var random = new System.Random(seed);
        const int m = 30;
        var a = new long[m];
        var d = new long[m];
        for (var i = 0; i < m; i++)
        {
            a[i] = magnitude + random.NextInt64(magnitude * 9);
            d[i] = random.Next(-3, 4);
        }

        var residual = new long[m];
        for (var t = 0; t + 2 < m; t += 3)
        {
            var (u0, u1, u2) = (a[t], a[t + 1], a[t + 2]);
            var (d0, d1, d2) = (d[t], d[t + 1], d[t + 2]);
            residual[t] = (u1 * d2) - (u2 * d1);
            residual[t + 1] = (u2 * d0) - (u0 * d2);
            residual[t + 2] = (u0 * d1) - (u1 * d0);
        }

        var x = new double[m, 2];
        var y = new double[m];
        for (var i = 0; i < m; i++)
        {
            x[i, 0] = a[i];
            x[i, 1] = a[i] + d[i];
            y[i] = (3 * a[i]) - (2 * (a[i] + d[i])) + residual[i];
        }

        return (x, y);
    }

    [Fact]
    public void InconsistentIllConditionedIntegerSystem_RecoversExactLeastSquaresSolution()
    {
        // Björck's augmented-system refinement makes κ₂(A) govern the attainable accuracy
        // regardless of the residual size; x-only refinement stalls near κ₂²·u·‖r‖ here.
        var (x, y) = InconsistentIntegerSystem(seed: 7, magnitude: 100_000_000);

        var result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false);

        result.Coefficients[0].Should().BeApproximately(3.0, 1e-12 * 3.0);
        result.Coefficients[1].Should().BeApproximately(-2.0, 1e-12 * 2.0);
    }

    [Fact]
    public void SingleTinyColumn_IsNotReportedRankDeficient()
    {
        // A single nonzero column has condition number 1 at every scale. The exact
        // no-intercept slope is Σxy/Σx² = 55.6/55 per unit of the regressor.
        const double s = 1e-16;
        var x = new double[,] { { 1 * s }, { 2 * s }, { 3 * s }, { 4 * s }, { 5 * s } };
        var y = new[] { 1.1, 1.9, 3.2, 3.9, 5.1 };

        OlsResult<double>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false))
            .Should().NotThrow();

        (result!.Coefficients[0] * s).Should().BeApproximately(55.6 / 55.0, 1e-12);
    }

    [Fact]
    public void Factor_TinyFullRankColumn_ReturnsTrueWithColumnNormDiagonal()
    {
        const double s = 1e-20;
        var a = new double[,] { { 1 * s }, { 2 * s }, { 3 * s }, { 4 * s }, { 5 * s } };
        var y = new double[5];
        var diagonal = new double[1];
        var permutation = new int[1];

        var factored = HouseholderQr<double>.Factor(a, y, diagonal, permutation);

        factored.Should().BeTrue();
        Math.Abs(diagonal[0]).Should().BeApproximately(Math.Sqrt(55.0) * s, 1e-15 * Math.Sqrt(55.0) * s);
    }

    [Fact]
    public void Factor_NumericallyDependentLargeScaleColumns_ReturnsFalse()
    {
        // Column 2 is exactly 3 × column 1; at a 1e10 scale the rounding residue left in
        // the second pivot column is ~1e-6 in absolute terms, far above any absolute
        // floor, but it is ~1e-16 relative to the column's own norm.
        var a = new double[6, 2];
        for (var i = 0; i < 6; i++)
        {
            var v = 1e10 * (1.0 + (0.37 * i) + (0.011 * i * i));
            a[i, 0] = v;
            a[i, 1] = 3.0 * v;
        }

        var y = new double[6];
        var factored = HouseholderQr<double>.Factor(a, y, new double[2], new int[2]);

        factored.Should().BeFalse();
    }

    [Theory]
    [InlineData(1e-160)]
    [InlineData(1e-200)]
    [InlineData(1e-250)]
    public void SingleColumnAtExtremeSmallScale_FitsLikeUnitScale(double s)
    {
        var x = new double[,] { { 1 * s }, { 2 * s }, { 3 * s }, { 4 * s }, { 5 * s } };
        var y = new[] { 1.1, 1.9, 3.2, 3.9, 5.1 };

        OlsResult<double>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false))
            .Should().NotThrow();

        (result!.Coefficients[0] * s).Should().BeApproximately(55.6 / 55.0, 1e-12);
    }
}
