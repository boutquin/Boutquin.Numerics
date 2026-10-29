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

using Boutquin.Numerics.Solvers;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

/// <summary>
/// Range behavior of the Householder QR path behind <see cref="OrdinaryLeastSquares{T}"/> when an
/// intermediate of the factorization, the inverse factor or the refinement exceeds the working range
/// although every returned quantity is representable. Expected values come from the exact algebra of
/// each constructed design (stated beside each test) or from exact power-of-two scale equivariance
/// against the unit-scale fit, never from the scaled path under test.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class HouseholderQrExtremeScaleTests
{
    private static void AssertRelative(double actual, double expected, double tolerance, string because)
    {
        double.IsFinite(actual).Should().BeTrue(because);
        Math.Abs((actual / expected) - 1.0).Should().BeLessThanOrEqualTo(tolerance, because);
    }

    private static void AssertRelative(decimal actual, decimal expected, decimal tolerance, string because)
        => Math.Abs((actual / expected) - 1m).Should().BeLessThanOrEqualTo(tolerance, because);

    private static OlsResult<T> FitWithoutException<T>(T[,] x, T[] y, bool includeIntercept, string because)
        where T : System.Numerics.IFloatingPoint<T>
    {
        OlsResult<T>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<T>.Fit(x, y, includeIntercept))
            .Should().NotThrow(because);
        return result!;
    }

    // ── R^-1 substitution: r_ij * z_j formed before the division by r_ii ──────────

    [Fact]
    public void Ols_Double_RInverseNumeratorBeyondRange_StandardErrorsAreFinite()
    {
        // X = [[1e100, 1e100], [0, 1e90], [0, 0]], y = (0, 0, 1e300): beta = 0 and the only residual is
        // 1e300, so sigma = 1e300 (dof 1). XtX = [[1e200, 1e200], [1e200, 1e200 + 1e180]] has determinant
        // 1e380, so diag((XtX)^-1) = ((1e200 + 1e180) / 1e380, 1e200 / 1e380) = (1e-180 (1 + 1e-20), 1e-180)
        // and both standard errors are 1e300 * 1e-90 = 1e210.
        double[,] x = { { 1e100, 1e100 }, { 0, 1e90 }, { 0, 0 } };
        double[] y = [0, 0, 1e300];

        var result = FitWithoutException(x, y, false, "the design is full rank and every standard error is representable");

        result.StandardErrors.Should().HaveCount(2);
        AssertRelative(result.StandardErrors[0], 1e210, 1e-12, "SE_0 = sigma * sqrt((XtX)^-1_00) = 1e210");
        AssertRelative(result.StandardErrors[1], 1e210, 1e-12, "SE_1 = sigma * sqrt((XtX)^-1_11) = 1e210");
    }

    [Fact]
    public void Ols_Decimal_RInverseNumeratorBeyondRange_FullRankFitIsNotRejected()
    {
        // X = [[1e15, 1e15], [0, 1], [0, 0]], y = (0, 0, 1e14): beta = 0, sigma = 1e14 (dof 1).
        // XtX = [[1e30, 1e30], [1e30, 1e30 + 1]] has determinant 1e30, so
        // sigma^2 (XtX)^-1 = 1e28 * [[1 + 1e-30, -1], [-1, 1]] and both standard errors are 1e14.
        decimal[,] x = { { 1e15m, 1e15m }, { 0m, 1m }, { 0m, 0m } };
        decimal[] y = [0m, 0m, 1e14m];

        var result = FitWithoutException(x, y, false, "the design is full rank (cond 2e15) and its covariance is representable");

        AssertRelative(result.StandardErrors[0], 1e14m, 1e-12m, "SE_0 = 1e14");
        AssertRelative(result.StandardErrors[1], 1e14m, 1e-12m, "SE_1 = 1e14");
        AssertRelative(result.CovarianceMatrix[0, 1], -1e28m, 1e-12m, "Cov_01 = -1e28");
        AssertRelative(result.CovarianceMatrix[1, 1], 1e28m, 1e-12m, "Cov_11 = 1e28");
    }

    // ── Qt.y: a response whose 2-norm exceeds the range, with a representable solution ─────

    [Fact]
    public void Ols_Double_ResponseNormBeyondRange_SingleColumnIsFitted()
    {
        // X = (1, 1)t, y = (1.5e308, 1.5e308): beta = 1.5e308 exactly, ||y||2 = 2.1e308 > double.MaxValue.
        double[,] x = { { 1 }, { 1 } };
        double[] y = [1.5e308, 1.5e308];

        var result = FitWithoutException(x, y, false, "the least-squares solution 1.5e308 is representable");

        AssertRelative(result.Coefficients[0], 1.5e308, 1e-15, "beta = mean(y) = 1.5e308");
    }

    [Fact]
    public void Ols_Double_ResponseNormBeyondRange_OrthogonalPairIsFitted()
    {
        // X = [[1, 1], [1, -1]], y = (1.5e308, 1.5e308): beta = (1.5e308, 0) exactly.
        double[,] x = { { 1, 1 }, { 1, -1 } };
        double[] y = [1.5e308, 1.5e308];

        var result = FitWithoutException(x, y, false, "the least-squares solution (1.5e308, 0) is representable");

        AssertRelative(result.Coefficients[0], 1.5e308, 1e-15, "beta_0 = (y_0 + y_1) / 2 = 1.5e308");
        Math.Abs(result.Coefficients[1]).Should().BeLessThanOrEqualTo(1.5e308 * 1e-15, "beta_1 = (y_0 - y_1) / 2 = 0");
    }

    [Fact]
    public void Ols_Double_ResponseNormBeyondRange_InterceptDesignIsFitted()
    {
        // y = 1e308 + 0.5e308 * x on x = (1, -1, 1, -1): y = (1.5e308, 0.5e308, 1.5e308, 0.5e308).
        double[,] x = { { 1 }, { -1 }, { 1 }, { -1 } };
        double[] y = [1.5e308, 0.5e308, 1.5e308, 0.5e308];

        var result = FitWithoutException(x, y, true, "the least-squares solution (1e308, 0.5e308) is representable");

        AssertRelative(result.Coefficients[0], 1e308, 1e-15, "intercept = 1e308");
        AssertRelative(result.Coefficients[1], 0.5e308, 1e-15, "slope = 0.5e308");
    }

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_UnitColumnIsFitted()
    {
        // X = (1, 1)t, y = (6e28, 6e28): beta = 6e28, ||y||2 = 8.5e28 > decimal.MaxValue.
        decimal[,] x = { { 1m }, { 1m } };
        decimal[] y = [6e28m, 6e28m];

        var result = FitWithoutException(x, y, false, "the least-squares solution 6e28 is representable");

        Math.Abs(result.Coefficients[0] - 6e28m).Should().BeLessThanOrEqualTo(10m, "beta = mean(y) = 6e28");
    }

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_LargeColumnIsFitted()
    {
        // X = (4e28, 4e28)t, y = (6e28, 6e28): beta = 1.5.
        decimal[,] x = { { 4e28m }, { 4e28m } };
        decimal[] y = [6e28m, 6e28m];

        var result = FitWithoutException(x, y, false, "the least-squares solution 1.5 is representable");

        Math.Abs(result.Coefficients[0] - 1.5m).Should().BeLessThanOrEqualTo(1e-20m, "beta = 6e28 / 4e28 = 1.5");
    }

    [Fact]
    public void Ols_Double_UnrepresentableSolution_IsReportedAsInvalidOperation()
    {
        // beta = 1.5e308 / 0.5 = 3e308 is beyond double.MaxValue: no result can be returned.
        double[,] x = { { 0.5 }, { 0.5 } };
        double[] y = [1.5e308, 1.5e308];

        FluentActions.Invoking(() => OrdinaryLeastSquares<double>.Fit(x, y, false))
            .Should().Throw<InvalidOperationException>();
    }

    // ── refinement: g = -Xt.r restored at absolute scale ─────────────────────────────

    [Fact]
    public void Ols_Double_RefinementGradientBeyondRange_ExactFitIsReturned()
    {
        // X = (1e307, 1e307)t, y = (1.5e307, 1.5e307): beta = 1.5 and the exact residual is 0, but a
        // rounding-level residual of the first iterate times the 1e307 column exceeds double.MaxValue.
        double[,] x = { { 1e307 }, { 1e307 } };
        double[] y = [1.5e307, 1.5e307];

        var result = FitWithoutException(x, y, false, "the least-squares solution 1.5 is representable");

        AssertRelative(result.Coefficients[0], 1.5, 1e-15, "beta = 1.5e307 / 1e307 = 1.5");
    }

    private static readonly double[,] s_unitX =
    {
        { 1, 0.5 }, { 2, -1.25 }, { 3, 2.0 }, { -1, 0.75 }, { 0.5, -3.0 }, { 4, 1.5 },
    };

    private static readonly double[] s_unitY = [0.3, -1.7, 2.2, 0.9, -2.6, 1.1];

    [Theory]
    [InlineData(500, 900, false)]
    [InlineData(500, 900, true)]
    [InlineData(300, 800, false)]
    [InlineData(300, 800, true)]
    public void Ols_Double_PowerOfTwoScaledFit_IsScaleEquivariant(int designExponent, int responseExponent, bool includeIntercept)
    {
        // Multiplying X by 2^a and y by 2^b is exact, and the least-squares problem is equivariant under
        // it: every slope scales by 2^(b-a), the intercept and sigma by 2^b, and each standard error with
        // its coefficient. The unit-scale fit is the reference; only the ratio is asserted.
        var m = s_unitY.Length;
        var p = s_unitX.GetLength(1);
        var xs = new double[m, p];
        var ys = new double[m];
        for (var i = 0; i < m; i++)
        {
            ys[i] = Math.ScaleB(s_unitY[i], responseExponent);
            for (var j = 0; j < p; j++)
            {
                xs[i, j] = Math.ScaleB(s_unitX[i, j], designExponent);
            }
        }

        var unit = OrdinaryLeastSquares<double>.Fit(s_unitX, s_unitY, includeIntercept);
        var scaled = FitWithoutException(xs, ys, includeIntercept, "the scaled problem has a representable solution");

        var offset = includeIntercept ? 1 : 0;
        for (var k = 0; k < unit.Coefficients.Length; k++)
        {
            var exponent = k < offset ? responseExponent : responseExponent - designExponent;
            AssertRelative(scaled.Coefficients[k], Math.ScaleB(unit.Coefficients[k], exponent), 1e-13,
                $"coefficient {k} scales by 2^{exponent}");
            AssertRelative(scaled.StandardErrors[k], Math.ScaleB(unit.StandardErrors[k], exponent), 1e-12,
                $"standard error {k} scales by 2^{exponent}");
        }

        AssertRelative(scaled.ResidualStandardDeviation, Math.ScaleB(unit.ResidualStandardDeviation, responseExponent), 1e-12,
            "sigma scales by 2^b");
    }
}
