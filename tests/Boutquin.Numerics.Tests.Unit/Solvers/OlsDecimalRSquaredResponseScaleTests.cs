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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// R-squared of a <c>decimal</c> OLS fit whose response norm exceeds the type's range, so the
/// factorization works on an exactly rescaled response. A non-exact fit is required: the residual
/// scale enters R-squared only through the residuals, which an exact fit makes zero. Expected values
/// are exact rational arithmetic on the constructed data, never the implementation's own output.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class OlsDecimalRSquaredResponseScaleTests
{
    // x = 0..3, y = 7e28 + d with d = (1, 5, 2, 9) * 1e12. Centred on the common offset, with
    // x-mean 1.5 and d-mean 4.25 (in units of 1e12):
    //   Sxx = 5, Sxy = 10.5, slope = 2.1, explained SS = Sxy^2 / Sxx = 22.05,
    //   TSS = 1 + 25 + 4 + 81 - 4 * 4.25^2 = 38.75, RSS = 38.75 - 22.05 = 16.7,
    // so R-squared = 22.05 / 38.75 = 0.569032258064516129... and RSS = 16.7 * (1e12)^2 = 1.67e25.
    private const decimal ExactRSquared = 22.05m / 38.75m;
    private const decimal ExactResidualSumOfSquares = 1.67e25m;

    private static OlsResult<decimal> FitOffsetResponse()
    {
        decimal[,] x = { { 0m }, { 1m }, { 2m }, { 3m } };
        decimal[] y = [7e28m + 1e12m, 7e28m + 5e12m, 7e28m + 2e12m, 7e28m + 9e12m];
        return OrdinaryLeastSquares<decimal>.Fit(x, y, includeIntercept: true);
    }

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_NonExactFitRSquaredMatchesExactRational()
    {
        var result = FitOffsetResponse();

        Math.Abs(result.RSquared - ExactRSquared).Should().BeLessThanOrEqualTo(
            1e-20m,
            because: $"R-squared of the constructed fit is 22.05 / 38.75 (returned {result.RSquared})");
    }

    [Fact]
    public void Ols_Decimal_ResponseNormBeyondRange_NonExactFitResidualSumOfSquaresMatchesExactRational()
    {
        var result = FitOffsetResponse();

        // The residual and total scales must agree: R-squared = 1 - RSS / TSS ties this value to the
        // R-squared asserted above through the independently derived TSS of 38.75e24.
        Math.Abs(result.ResidualSumOfSquares - ExactResidualSumOfSquares).Should().BeLessThanOrEqualTo(
            1e10m,
            because: $"the residual sum of squares of the constructed fit is 1.67e25 (returned {result.ResidualSumOfSquares})");
        (1m - (result.ResidualSumOfSquares / 3.875e25m)).Should().BeApproximately(
            result.RSquared,
            1e-15m,
            "R-squared must equal 1 - RSS / TSS for the returned residual sum of squares");
    }
}
