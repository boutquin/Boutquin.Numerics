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

using System.Numerics;

using Boutquin.Numerics.LinearAlgebra.Internal;
using Boutquin.Numerics.Solvers;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Range behavior of <see cref="OrdinaryLeastSquares{T}"/> and its Householder QR away from
/// unit scale. Every expected value comes from exact integer arithmetic on the returned
/// numbers, from a closed form, or from the same fit at unit scale (OLS is scale-equivariant),
/// never from the implementation under test.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class Arena414074R1OlsQrTests
{
    // ── exact oracles ────────────────────────────────────────────────────────────

    /// <summary>The exact integer <c>value / 2^-1074</c>.</summary>
    private static BigInteger SubnormalUnits(double value)
    {
        var bits = BitConverter.DoubleToInt64Bits(value);
        var biased = (int)((bits >> 52) & 0x7FF);
        var fraction = bits & 0xF_FFFF_FFFF_FFFFL;
        var units = biased == 0
            ? new BigInteger(fraction)
            : new BigInteger(fraction | (1L << 52)) << (biased - 1);
        return bits < 0 ? -units : units;
    }

    /// <summary>The exact integer <c>value / 1e-28</c> (decimal values always have scale at most 28).</summary>
    private static BigInteger DecimalUnits(decimal value)
    {
        var bits = decimal.GetBits(value);
        var scale = (bits[3] >> 16) & 0xFF;
        var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | (uint)bits[0];
        var units = magnitude * BigInteger.Pow(10, 28 - scale);
        return (bits[3] & int.MinValue) != 0 ? -units : units;
    }

    /// <summary>
    /// Asserts <paramref name="reported"/> is the exact sum of the squares of
    /// <paramref name="residuals"/> to within one representable step (one subnormal unit) or
    /// 2^-49 relative, whichever is larger.
    /// </summary>
    private static void AssertExactSumOfSquares(double reported, double[] residuals)
    {
        var exact = residuals.Aggregate(BigInteger.Zero, (sum, r) => sum + BigInteger.Pow(SubnormalUnits(r), 2));
        var returned = SubnormalUnits(reported) << 1074;
        var tolerance = BigInteger.Max(BigInteger.One << 1074, exact >> 49);
        var difference = BigInteger.Abs(returned - exact);
        (difference <= tolerance).Should().BeTrue(
            because: $"RSS {reported:R} must equal the exact sum of the squared returned residuals "
                + $"(difference {difference} against tolerance {tolerance}, in units of 2^-2148)");
    }

    private static void AssertExactSumOfSquares(decimal reported, decimal[] residuals)
    {
        var exact = residuals.Aggregate(BigInteger.Zero, (sum, r) => sum + BigInteger.Pow(DecimalUnits(r), 2));
        var returned = DecimalUnits(reported) * BigInteger.Pow(10, 28);
        var tolerance = BigInteger.Pow(10, 28) + (exact / BigInteger.Pow(10, 20));
        var difference = BigInteger.Abs(returned - exact);
        (difference <= tolerance).Should().BeTrue(
            because: $"RSS {reported} must equal the exact sum of the squared returned residuals "
                + $"(difference {difference} against tolerance {tolerance}, in units of 1e-56)");
    }

    private static double[] Alternating(int count, double magnitude) =>
        Enumerable.Range(0, count).Select(i => i % 2 == 0 ? magnitude : -magnitude).ToArray();

    private static decimal[] Alternating(int count, decimal magnitude) =>
        Enumerable.Range(0, count).Select(i => i % 2 == 0 ? magnitude : -magnitude).ToArray();

    // ── F-L1-09: returned RSS must not flush a representable sum ────────────────

    [Fact]
    public void Ols_ResidualSumOfSquares_SubnormalSumOfFlushedSquaresIsRecovered()
    {
        // Sixteen residuals of +-1e-162: each square (1e-324) rounds to zero, yet their exact
        // sum 1.6e-323 is representable (three subnormal units).
        var result = OrdinaryLeastSquares<double>.Fit(new double[16, 0], Alternating(16, 1e-162));

        AssertExactSumOfSquares(result.ResidualSumOfSquares, result.Residuals);
        result.ResidualSumOfSquares.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Ols_LegacyFacade_ResidualSumOfSquares_SubnormalSumOfFlushedSquaresIsRecovered()
    {
        var result = OrdinaryLeastSquares.Fit(new double[16, 0], Alternating(16, 1e-162));

        AssertExactSumOfSquares(result.ResidualSumOfSquares, result.Residuals);
        result.ResidualSumOfSquares.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Ols_Decimal_ResidualSumOfSquares_SumBelowFixedScaleSquaresIsRecovered()
    {
        // 100 residuals of +-3e-15: each square (9e-30) rounds to zero at decimal's 28 places,
        // while the exact sum 9e-28 is nine representable units.
        var result = OrdinaryLeastSquares<decimal>.Fit(new decimal[100, 0], Alternating(100, 3e-15m));

        AssertExactSumOfSquares(result.ResidualSumOfSquares, result.Residuals);
        result.ResidualSumOfSquares.Should().BeGreaterThan(0m);
    }

    // ── F-L1-10: sigma must not be square-rooted from a partly flushed variance ─

    [Fact]
    public void Ols_ResidualStandardDeviation_PositiveSubnormalVarianceKeepsAllDigits()
    {
        // +-1e-160 on 16 observations: sigma^2 = 16e-320 / 15 is subnormal but positive, so
        // the old rescue (zero or non-finite only) square-rooted a variance with ~5 digits.
        var result = OrdinaryLeastSquares<double>.Fit(new double[16, 0], Alternating(16, 1e-160));

        var a = Math.Abs(result.Residuals[0]);
        var expected = a * Math.Sqrt(16.0 / 15.0);
        result.ResidualStandardDeviation.Should().BeApproximately(expected, expected * 1e-14);
    }

    [Fact]
    public void Ols_Decimal_ResidualStandardDeviation_FixedScaleVarianceKeepsAllDigits()
    {
        // sigma^2 = 2.4e-23 / 15 sits on decimal's 1e-28 grid with four digits left.
        var result = OrdinaryLeastSquares<decimal>.Fit(new decimal[16, 0], Alternating(16, 1.234567890123e-12m));

        var a = Math.Abs(result.Residuals[0]);
        var expected = (double)a * Math.Sqrt(16.0 / 15.0);
        ((double)result.ResidualStandardDeviation).Should().BeApproximately(expected, expected * 1e-13);
    }

    // ── F-L1-11: standard errors must not be square roots of a flushed/overflowed variance ─

    [Fact]
    public void Ols_StandardErrors_RepresentableSeIsReturnedWhenVarianceOverflows()
    {
        // sigma = 1.03e200 and SE = sigma / 4 = 2.58e199 are representable; the variance
        // 6.7e398 is not.
        var result = OrdinaryLeastSquares<double>.Fit(new double[16, 0], Alternating(16, 1e200));

        var expected = 1e200 / Math.Sqrt(15.0);
        result.StandardErrors[0].Should().BeApproximately(expected, expected * 1e-14);
    }

    [Fact]
    public void Ols_StandardErrors_RepresentableSeIsReturnedWhenVarianceUnderflows()
    {
        var result = OrdinaryLeastSquares<double>.Fit(new double[16, 0], Alternating(16, 1e-162));

        var expected = 1e-162 / Math.Sqrt(15.0);
        result.StandardErrors[0].Should().BeApproximately(expected, expected * 1e-14);
    }

    [Fact]
    public void Ols_StandardErrors_TinyRegressorScaleEquivariance()
    {
        // Scaling y by s scales beta, sigma and every standard error by s; XtX is unchanged.
        double[,] x = { { -1 }, { 0 }, { 1 } };
        var unit = OrdinaryLeastSquares<double>.Fit(x, [1.0, -2.0, 1.0]).StandardErrors;
        var tiny = OrdinaryLeastSquares<double>.Fit(x, [1e-200, -2e-200, 1e-200]);

        tiny.HasInferentialContent.Should().BeTrue();
        for (var i = 0; i < unit.Length; i++)
        {
            var expected = unit[i] * 1e-200;
            tiny.StandardErrors[i].Should().BeApproximately(expected, expected * 1e-13, because: $"SE[{i}]");
        }
    }

    [Fact]
    public void Ols_StandardErrors_OverflowingInverseEntriesSaturateToInfinityNotNaN()
    {
        // sigma / ||x - mean|| = 2.4e160 / 1.4e-150 is beyond double, so the scaled inverse factor
        // has infinite entries while every coefficient is finite. The covariance reports +Infinity
        // there, and each standard error must agree with it: +Infinity, never NaN (an infinite
        // row maximum divided into itself) and never a finite value.
        double[,] x = { { -1e-150 }, { 0 }, { 1e-150 } };
        var result = OrdinaryLeastSquares<double>.Fit(x, [1e160, -2e160, 1e160]);

        result.Coefficients.Should().OnlyContain(c => double.IsFinite(c), because: "the premise: every coefficient is finite");
        var covariance = result.CovarianceMatrix;
        for (var i = 0; i < 2; i++)
        {
            double.IsPositiveInfinity(covariance[i, i]).Should().BeTrue(because: $"the premise: covariance[{i},{i}] overflows");
            double.IsPositiveInfinity(result.StandardErrors[i]).Should().BeTrue(
                because: $"standard error {i} must saturate with its covariance entry, found {result.StandardErrors[i]}");
        }
    }

    [Fact]
    public void Ols_Decimal_StandardErrors_FixedScaleVarianceKeepsRepresentableSe()
    {
        var result = OrdinaryLeastSquares<decimal>.Fit(new decimal[100, 0], Alternating(100, 3e-15m));

        // SE = sigma / sqrt(100) with sigma = a * sqrt(100 / 99). A decimal near 3e-16 has
        // about 13 digits left on its 1e-28 grid, so the bar is one grid step, not 1e-13 relative.
        var expected = 3e-15 * Math.Sqrt(100.0 / 99.0) / 10.0;
        ((double)result.StandardErrors[0]).Should().BeApproximately(expected, 1e-28);
    }

    // ── F-L1-12: decimal residual evaluation must not need an unrepresentable beta*x ─

    [Fact]
    public void Ols_Decimal_ResidualEvaluation_RepresentableResidualNeverNeedsAnUnrepresentableProduct()
    {
        // Without the noise below the fit is exact, beta = (1e19, -1e19): each product
        // beta_j * x_ij is about 1e29, past decimal's 7.9e28, while the residual y - X.beta is
        // tiny. The +-1e13 noise leaves residuals of the order 1e13 (squares below decimal's
        // range) that the exact oracle below pins to a few units, so a residual shifted back by
        // a wrong power of ten is caught.
        decimal[,] x =
        {
            { 1e10m, 1e10m },
            { 1e10m, 1e10m + 1e4m },
            { 1e10m, 1e10m + 2e4m },
            { 1e10m, 1e10m + 3e4m },
        };
        decimal[] y = [4e13m, -1e23m - 7e13m, -2e23m + 2e13m, -3e23m - 5e13m];

        OlsResult<decimal>? result = null;
        var fit = () => result = OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept: false);
        fit.Should().NotThrow();

        var beta = result!.Coefficients;
        var residuals = result.Residuals;
        for (var i = 0; i < 4; i++)
        {
            // Exact y - X.beta of the returned coefficients, in units of 1e-56.
            var exact = (DecimalUnits(y[i]) * BigInteger.Pow(10, 28))
                - (DecimalUnits(beta[0]) * DecimalUnits(x[i, 0]))
                - (DecimalUnits(beta[1]) * DecimalUnits(x[i, 1]));
            var returned = DecimalUnits(residuals[i]) * BigInteger.Pow(10, 28);
            var tolerance = new BigInteger(100) * BigInteger.Pow(10, 56);
            (BigInteger.Abs(returned - exact) <= tolerance).Should().BeTrue(
                because: $"residual[{i}] = {residuals[i]} must match the exact y - X.beta (within 100)");
        }
    }

    // ── F-L1-18: the QR row retry must not normalise small entries to decimal zero ─

    [Fact]
    public void HouseholderQr_Decimal_UpperRowRetry_RepresentableQuotientSurvivesUnitEntries()
    {
        // R = [[2, -1], [0, 1]], c = (7e28, 7e28): x1 = 7e28, and row 0 forms
        // 7e28 + 7e28 (decimal overflow) although (7e28 + 7e28) / 2 = 7e28 is representable.
        decimal[,] a = { { 0m, -1m }, { 0m, 0m } };
        decimal[] diagonal = [2m, 1m];
        decimal[]? solved = null;
        var solve = () => solved = HouseholderQr<decimal>.SolveUpperTriangular(a, diagonal, [7e28m, 7e28m], 2, [0, 1]);

        solve.Should().NotThrow();
        solved.Should().Equal(7e28m, 7e28m);
    }

    [Fact]
    public void HouseholderQr_Decimal_TransposeRowRetry_RepresentableQuotientSurvivesUnitEntries()
    {
        // Rt h = g with R = [[1, 1], [0, 2]], g = (7e28, -7e28): h0 = 7e28 and row 1 forms
        // -7e28 - 7e28 although (-7e28 - 7e28) / 2 = -7e28 is representable.
        decimal[,] a = { { 0m, 1m }, { 0m, 0m } };
        decimal[] diagonal = [1m, 2m];
        decimal[]? solved = null;
        var solve = () => solved = HouseholderQr<decimal>.SolveUpperTriangularTranspose(a, diagonal, [7e28m, -7e28m], 2, [0, 1]);

        solve.Should().NotThrow();
        solved.Should().Equal(7e28m, -7e28m);
    }

    [Fact]
    public void Ols_Decimal_RepresentableCoefficientsAboveHalfTheRangeAreFitted()
    {
        // The public reach of the row retry: X = [[2, -1], [0, 1]], y = (7e28, 7e28) has the
        // exact solution (7e28, 7e28); the residual evaluation also forms 1.4e29.
        decimal[,] x = { { 2m, -1m }, { 0m, 1m } };
        OlsResult<decimal>? result = null;
        var fit = () => result = OrdinaryLeastSquares<decimal>.Fit(x, [7e28m, 7e28m], includeIntercept: false);

        fit.Should().NotThrow();
        result!.Coefficients.Should().Equal(7e28m, 7e28m);
    }
}
