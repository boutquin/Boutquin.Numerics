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

using Boutquin.Numerics.Solvers;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Range behavior of <see cref="OrdinaryLeastSquares{T}"/> at <c>decimal</c> when the response
/// approaches the type's range: a representable least-squares solution is returned, and an
/// unrepresentable one is reported with the documented <see cref="InvalidOperationException"/>,
/// never a raw <see cref="OverflowException"/>. Every expected value comes from a construction
/// (the solution is chosen first and the data derived from it exactly) or from exact integer
/// arithmetic on the returned numbers, never from the implementation under test.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class Arena414074R2OlsQrTests
{
    // ── exact oracles ────────────────────────────────────────────────────────────

    /// <summary>The exact integer <c>value / 1e-28</c> (decimal values always have scale at most 28).</summary>
    private static BigInteger DecimalUnits(decimal value)
    {
        var bits = decimal.GetBits(value);
        var scale = (bits[3] >> 16) & 0xFF;
        var magnitude = (new BigInteger((uint)bits[2]) << 64) | (new BigInteger((uint)bits[1]) << 32) | (uint)bits[0];
        var units = magnitude * BigInteger.Pow(10, 28 - scale);
        return (bits[3] & int.MinValue) != 0 ? -units : units;
    }

    private static OlsResult<decimal> FitWithoutException(decimal[,] x, decimal[] y, bool includeIntercept, string because)
    {
        OlsResult<decimal>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept))
            .Should().NotThrow(because);
        return result!;
    }

    /// <summary>
    /// Asserts every returned coefficient equals the constructed solution to 1e-25 of the largest
    /// coefficient plus one unit (decimal resolves whole units at this magnitude).
    /// </summary>
    private static void AssertCoefficients(OlsResult<decimal> result, decimal[] expected)
    {
        result.Coefficients.Should().HaveCount(expected.Length);
        var largest = expected.Max(Math.Abs);
        var tolerance = (largest * 1e-25m) + 1m;
        for (var k = 0; k < expected.Length; k++)
        {
            Math.Abs(result.Coefficients[k] - expected[k]).Should().BeLessThanOrEqualTo(
                tolerance,
                because: $"coefficient {k} must equal the constructed solution {expected[k]} (returned {result.Coefficients[k]})");
        }
    }

    /// <summary>
    /// Asserts every returned residual equals the exact <c>y - X.beta</c> of the RETURNED
    /// coefficients (BigInteger arithmetic) to within one decimal unit per term.
    /// </summary>
    private static void AssertResidualsAreExact(decimal[,] x, decimal[] y, bool includeIntercept, OlsResult<decimal> result)
    {
        var scale = BigInteger.Pow(10, 28);
        var offset = includeIntercept ? 1 : 0;
        var p = x.GetLength(1);
        for (var i = 0; i < y.Length; i++)
        {
            var exact = DecimalUnits(y[i]) * scale;
            if (includeIntercept)
            {
                exact -= DecimalUnits(result.Coefficients[0]) * scale;
            }

            for (var j = 0; j < p; j++)
            {
                exact -= DecimalUnits(result.Coefficients[j + offset]) * DecimalUnits(x[i, j]);
            }

            var returned = DecimalUnits(result.Residuals[i]) * scale;
            var tolerance = new BigInteger(p + 1) * BigInteger.Pow(10, 56);
            (BigInteger.Abs(returned - exact) <= tolerance).Should().BeTrue(
                because: $"residual[{i}] = {result.Residuals[i]} must match the exact y - X.beta of the returned coefficients");
        }
    }

    // ── a response whose norm exceeds decimal.MaxValue, with a representable solution ──

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_OrthogonalPairIsFitted()
    {
        // ‖y‖₂ = 8.5e28 > decimal.MaxValue (7.9e28): the first entry of Qt.y equals it, yet the
        // exact solution (6e28, 0) of this orthogonal design is representable.
        decimal[,] x = { { 1m, 1m }, { 1m, -1m } };
        decimal[] y = [6e28m, 6e28m];

        var result = FitWithoutException(x, y, false, "the least-squares solution (6e28, 0) is representable");

        AssertCoefficients(result, [6e28m, 0m]);
        AssertResidualsAreExact(x, y, false, result);
    }

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_SignFlippedResponseIsFitted()
    {
        decimal[,] x = { { 1m, 1m }, { 1m, -1m } };
        decimal[] y = [6e28m, -6e28m];

        var result = FitWithoutException(x, y, false, "the least-squares solution (0, 6e28) is representable");

        AssertCoefficients(result, [0m, 6e28m]);
        AssertResidualsAreExact(x, y, false, result);
    }

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_InterceptDesignIsFitted()
    {
        // y = 4e28 + 3e28 * x on x = (1, -1, 1, -1): y = (7e28, 1e28, 7e28, 1e28), ‖y‖₂ = 1.0e29.
        decimal[,] x = { { 1m }, { -1m }, { 1m }, { -1m } };
        decimal[] y = [7e28m, 1e28m, 7e28m, 1e28m];

        var result = FitWithoutException(x, y, true, "the least-squares solution (4e28, 3e28) is representable");

        AssertCoefficients(result, [4e28m, 3e28m]);
        AssertResidualsAreExact(x, y, true, result);
    }

    // ── response-derived statistics of such a fit ────────────────────────────────

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_RSquaredOfAMixedSignResponseIsReturned()
    {
        // y = (7e28, 7e28, 7e28, -7e28): the mean is 3.5e28, so the centred deviation of the last
        // observation is -1.05e29, beyond decimal.MaxValue, although the fit is exact
        // (intercept 7e28, slope -1.4e28 on x = (0, 0, 0, 10)) and R-squared is 1.
        decimal[,] x = { { 0m }, { 0m }, { 0m }, { 10m } };
        decimal[] y = [7e28m, 7e28m, 7e28m, -7e28m];

        var result = FitWithoutException(x, y, true, "the fit and its R-squared are representable");

        AssertCoefficients(result, [7e28m, -1.4e28m]);
        Math.Abs(result.RSquared - 1m).Should().BeLessThanOrEqualTo(1e-20m, "an exact fit explains the whole variance");
    }

    // ── a result that is NOT representable is reported with the documented exception ──

    [Fact]
    public void Ols_Decimal_UnrepresentableSolution_IsReportedAsInvalidOperation()
    {
        // x = 7e28 / 0.5 = 1.4e29 is beyond decimal.MaxValue.
        decimal[,] x = { { 0.5m } };
        decimal[] y = [7e28m];

        FluentActions.Invoking(() => OrdinaryLeastSquares<decimal>.Fit(x, y, false))
            .Should().Throw<InvalidOperationException>()
            .WithInnerException<OverflowException>();
    }

    [Fact]
    public void Ols_Decimal_UnrepresentableResidualSumOfSquares_IsReportedAsInvalidOperation()
    {
        // The coefficient is 0 and the residuals are +-7e28 (representable), but their sum of
        // squares, 9.8e57, is not: no result of this fit can be returned.
        decimal[,] x = { { 1m }, { 1m } };
        decimal[] y = [7e28m, -7e28m];

        FluentActions.Invoking(() => OrdinaryLeastSquares<decimal>.Fit(x, y, false))
            .Should().Throw<InvalidOperationException>()
            .WithInnerException<OverflowException>();
    }

    // ── a design column whose 2-norm exceeds the range, with a representable solution ──
    // R's diagonal is a column norm, so such a column cannot be factored as given; its least-squares
    // solution is nevertheless representable whenever the coefficients are. These replace the former
    // refusal pin for X = [[7e28], [7e28]]: its exact solution is 1, so refusing it was wrong behavior.

    [Fact]
    public void Ols_Decimal_DesignColumnNormBeyondRange_RepresentableSolutionIsReturned()
    {
        // The column norm 7e28 * sqrt(2) = 9.9e28 exceeds decimal.MaxValue; y = X . 1 exactly.
        decimal[,] x = { { 7e28m }, { 7e28m } };
        decimal[] y = [7e28m, 7e28m];

        var result = FitWithoutException(x, y, false, "the least-squares solution 1 is representable");

        AssertCoefficients(result, [1m]);
        AssertResidualsAreExact(x, y, false, result);
    }

    [Fact]
    public void Ols_Decimal_DesignColumnNormBeyondRange_HalfCoefficientIsReturned()
    {
        // ||x||2 = 5e28 * 2 = 1e29 > decimal.MaxValue; y = 0.5 . x exactly.
        decimal[,] x = { { 5e28m }, { 5e28m }, { 5e28m }, { 5e28m } };
        decimal[] y = [2.5e28m, 2.5e28m, 2.5e28m, 2.5e28m];

        var result = FitWithoutException(x, y, false, "the least-squares solution 0.5 is representable");

        result.Coefficients.Should().Equal([0.5m], "y = 0.5 . x holds exactly");
        result.Residuals.Should().OnlyContain(r => r == 0m, "the fit is exact");
    }

    [Fact]
    public void Ols_Decimal_DesignColumnNormBeyondRange_StandardErrorsAreReturned()
    {
        // Orthogonal columns c0 = (1, -1, 1, -1) and c1 = 5e28 . 1 (norm 1e29, beyond the range; pivoting
        // moves it first). y = (12, 4, 10, 2) . 1e13: beta = (4e13, 7e13 / 5e28) = (4e13, 1.4e-15), residuals
        // (1, 1, -1, -1) . 1e13, sigma = sqrt(4e26 / 2) = sqrt(2) . 1e13, SE0 = sigma / 2,
        // SE1 = sigma / (2 . 5e28), Cov00 = sigma^2 / 4 = 5e25.
        decimal[,] x = { { 1m, 5e28m }, { -1m, 5e28m }, { 1m, 5e28m }, { -1m, 5e28m } };
        decimal[] y = [12e13m, 4e13m, 10e13m, 2e13m];

        var result = FitWithoutException(x, y, false, "every returned quantity is representable");

        Math.Abs(result.Coefficients[0] - 4e13m).Should().BeLessThanOrEqualTo(1e-12m);
        Math.Abs(result.Coefficients[1] - 1.4e-15m).Should().BeLessThanOrEqualTo(1e-27m);
        var sigma = Math.Sqrt(2.0) * 1e13;
        RelativeError((double)result.ResidualStandardDeviation, sigma).Should().BeLessThanOrEqualTo(1e-12);
        RelativeError((double)result.StandardErrors[0], sigma / 2).Should().BeLessThanOrEqualTo(1e-12);
        RelativeError((double)result.StandardErrors[1], sigma / 1e29).Should().BeLessThanOrEqualTo(1e-10);
        RelativeError((double)result.CovarianceMatrix[0, 0], 5e25).Should().BeLessThanOrEqualTo(1e-12);
    }

    [Fact]
    public void Ols_Double_DesignColumnNormBeyondRange_RepresentableSolutionIsReturned()
    {
        // ||x||2 = 1.5e308 * 2 = 3e308 > double.MaxValue; y = 1e-300 . x, so beta = 1e-300.
        double[,] x = { { 1.5e308 }, { 1.5e308 }, { 1.5e308 }, { 1.5e308 } };
        double[] y = [1.5e8, 1.5e8, 1.5e8, 1.5e8];

        OlsResult<double>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<double>.Fit(x, y, false))
            .Should().NotThrow("the least-squares solution 1e-300 is representable and the column has full rank");

        RelativeError(result!.Coefficients[0], 1e-300).Should().BeLessThanOrEqualTo(1e-14);
    }

    [Fact]
    public void Ols_Double_DesignColumnNormBeyondRange_StandardErrorsAndCovarianceAreUnscaled()
    {
        // Orthogonal columns c0 = 2^1023 . 1 (norm 2^1024, beyond the range) and c1 = 2^600 . (1, -1, 1, -1),
        // y = (12, 4, 10, 2) . 2^900. Exactly: beta = (7 . 2^-123, 4 . 2^300), residuals (1, 1, -1, -1) . 2^900,
        // sigma = sqrt(2) . 2^900, SE0 = sigma / (2 . 2^1023), SE1 = sigma / (2 . 2^600),
        // Cov00 = 2^(2 . 900 + 1 - 2 - 2046), Cov11 = 2^(2 . 900 + 1 - 2 - 1200), Cov01 = 0.
        var a = Math.ScaleB(1.0, 1023);
        var b = Math.ScaleB(1.0, 600);
        double[,] x = { { a, b }, { a, -b }, { a, b }, { a, -b } };
        double[] y = [Math.ScaleB(12.0, 900), Math.ScaleB(4.0, 900), Math.ScaleB(10.0, 900), Math.ScaleB(2.0, 900)];

        OlsResult<double>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<double>.Fit(x, y, false))
            .Should().NotThrow("every returned quantity is representable");

        RelativeError(result!.Coefficients[0], Math.ScaleB(7.0, -123)).Should().BeLessThanOrEqualTo(1e-14);
        RelativeError(result.Coefficients[1], Math.ScaleB(4.0, 300)).Should().BeLessThanOrEqualTo(1e-14);
        var halfRoot2 = Math.Sqrt(2.0) / 2.0;
        RelativeError(result.StandardErrors[0], Math.ScaleB(halfRoot2, 900 - 1023)).Should().BeLessThanOrEqualTo(1e-14);
        RelativeError(result.StandardErrors[1], Math.ScaleB(halfRoot2, 900 - 600)).Should().BeLessThanOrEqualTo(1e-14);
        RelativeError(result.CovarianceMatrix[0, 0], Math.ScaleB(1.0, 1800 + 1 - 2 - 2046)).Should().BeLessThanOrEqualTo(1e-14);
        RelativeError(result.CovarianceMatrix[1, 1], Math.ScaleB(1.0, 1800 + 1 - 2 - 1200)).Should().BeLessThanOrEqualTo(1e-14);
        Math.Abs(result.CovarianceMatrix[0, 1]).Should().BeLessThanOrEqualTo(
            1e-14 * Math.Sqrt(result.CovarianceMatrix[0, 0]) * Math.Sqrt(result.CovarianceMatrix[1, 1]));
    }

    [Fact]
    public void Ols_Double_DesignColumnNormBeyondRange_IllConditionedNoisyFitIsRefined()
    {
        // c0 = 2^1023 . 1 (norm 2^1024, beyond the range) and c1 = 1 + 2^-20 . (0, 1, -1, 2): nearly
        // collinear, so the QR solution alone is off and the refinement must correct it. y = X . beta + 2^10 . u
        // with beta = (3 . 2^-1000, 5) and u = (-3, 0, 2, 1) orthogonal to both columns, every entry exact
        // in double: the least-squares solution is beta and the residual is 2^10 . u exactly.
        var a = Math.ScaleB(1.0, 1023);
        var delta = Math.ScaleB(1.0, -20);
        double[] n = [0.0, 1.0, -1.0, 2.0];
        double[] u = [-3.0, 0.0, 2.0, 1.0];
        var beta0 = Math.ScaleB(3.0, -1000);
        var x = new double[4, 2];
        var y = new double[4];
        for (var i = 0; i < 4; i++)
        {
            x[i, 0] = a;
            x[i, 1] = 1.0 + (delta * n[i]);
            y[i] = Math.ScaleB(3.0, 23) + (5.0 * x[i, 1]) + Math.ScaleB(u[i], 10);
        }

        OlsResult<double>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<double>.Fit(x, y, false))
            .Should().NotThrow("the least-squares solution is representable and the design has full rank");

        RelativeError(result!.Coefficients[0], beta0).Should().BeLessThanOrEqualTo(1e-12);
        RelativeError(result.Coefficients[1], 5.0).Should().BeLessThanOrEqualTo(1e-12);
        RelativeError(result.ResidualSumOfSquares, Math.ScaleB(14.0, 20)).Should().BeLessThanOrEqualTo(1e-9);
    }

    // ── the refusal side of the pair: a solution that is NOT representable is still refused ──

    [Fact]
    public void Ols_Decimal_DesignColumnNormBeyondRange_UnrepresentableSolutionIsReportedAsInvalidOperation()
    {
        // Square system: 7e28 a = 7e28 and 7e28 a + 0.5 b = 0, so b = -1.4e29, beyond decimal.MaxValue.
        decimal[,] x = { { 7e28m, 0.5m }, { 7e28m, 0m } };
        decimal[] y = [0m, 7e28m];

        FluentActions.Invoking(() => OrdinaryLeastSquares<decimal>.Fit(x, y, false))
            .Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Ols_Double_DesignColumnNormBeyondRange_UnrepresentableSolutionIsReportedAsInvalidOperation()
    {
        // 1.5e308 a = 1.5e308 and 1.5e308 a + 0.5 b = 0, so b = -3e308, beyond double.MaxValue.
        double[,] x = { { 1.5e308, 0.5 }, { 1.5e308, 0.0 } };
        double[] y = [0.0, 1.5e308];

        FluentActions.Invoking(() => OrdinaryLeastSquares<double>.Fit(x, y, false))
            .Should().Throw<InvalidOperationException>();
    }

    private static double RelativeError(double actual, double expected)
        => double.IsFinite(actual) ? Math.Abs((actual / expected) - 1.0) : double.PositiveInfinity;
}
