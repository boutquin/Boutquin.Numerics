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

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Python cross-checks for <see cref="OrdinaryLeastSquares"/> against
/// ``statsmodels.api.OLS``. Closes the reference-verification loop on the
/// OLS estimator shipped via ``feature-nist-strd-expansion.md`` (which
/// covered NIST StRD only). See
/// <c>tests/Verification/generate_ols_vectors.py</c> for the reference
/// generator.
/// </summary>
/// <remarks>
/// Three regimes (well-conditioned, polynomial degree 5, no-intercept)
/// exercise the common-case quick-path, the moderately-conditioned pivoting
/// path, and the uncentred-R² branch respectively. Tolerances match the
/// spec's §2.1 — 1e-10 relative on coefficients (NIST Longley bar for
/// well-conditioned problems), 1e-8 relative on derived quantities.
/// </remarks>
public sealed class OrdinaryLeastSquaresVerificationTests : CrossLanguageVerificationBase
{
    private const double CoefficientTolerance = 1e-10;
    private const double DerivedTolerance = 1e-8;

    [Theory]
    [InlineData("well_conditioned")]
    [InlineData("polynomial_degree_5")]
    [InlineData("no_intercept")]
    public void Fit_MatchesStatsmodels(string regimeName)
    {
        using var doc = LoadVector("ols");
        var regime = doc.RootElement.GetProperty("regimes").GetProperty(regimeName);

        var x = GetDouble2D(regime.GetProperty("x"));
        var y = GetDoubleArray(regime.GetProperty("y"));
        var includeIntercept = regime.GetProperty("include_intercept").GetBoolean();

        var expectedCoefficients = GetDoubleArray(regime.GetProperty("coefficients"));
        var expectedStandardErrors = GetDoubleArray(regime.GetProperty("standard_errors"));
        var expectedRss = regime.GetProperty("residual_sum_of_squares").GetDouble();
        var expectedSigma = regime.GetProperty("residual_standard_deviation").GetDouble();
        var expectedDof = regime.GetProperty("degrees_of_freedom").GetInt32();
        var expectedR2 = regime.GetProperty("r_squared").GetDouble();

        var result = OrdinaryLeastSquares.Fit(x, y, includeIntercept);

        // --- Coefficients --------------------------------------------------
        // statsmodels reports coefficients in the design-matrix order, which
        // after ``sm.add_constant(..., has_constant="add")`` places the
        // intercept at index 0. The Numerics convention is identical, so
        // indices line up without reshuffling.
        Assert.Equal(expectedCoefficients.Length, result.Coefficients.Length);
        for (var i = 0; i < expectedCoefficients.Length; i++)
        {
            AssertRelative(
                result.Coefficients[i],
                expectedCoefficients[i],
                CoefficientTolerance,
                $"{regimeName}/coefficient[{i}]");
        }

        // --- Standard errors ----------------------------------------------
        Assert.Equal(expectedStandardErrors.Length, result.StandardErrors.Length);
        for (var i = 0; i < expectedStandardErrors.Length; i++)
        {
            AssertRelative(
                result.StandardErrors[i],
                expectedStandardErrors[i],
                DerivedTolerance,
                $"{regimeName}/standard_error[{i}]");
        }

        // --- Scalar derived quantities ------------------------------------
        AssertRelative(result.ResidualSumOfSquares, expectedRss, DerivedTolerance, $"{regimeName}/rss");
        AssertRelative(result.ResidualStandardDeviation, expectedSigma, DerivedTolerance, $"{regimeName}/sigma");
        Assert.Equal(expectedDof, result.DegreesOfFreedom);
        AssertRelative(result.RSquared, expectedR2, DerivedTolerance, $"{regimeName}/r_squared");
    }

    /// <summary>
    /// Asserts <paramref name="actual"/> agrees with <paramref name="expected"/>
    /// within <paramref name="tolerance"/> in the ``max(|expected|, 1)``
    /// relative sense — exactly the convention statsmodels uses internally
    /// when expressing per-coefficient agreement against published NIST
    /// certified values. Falls back to absolute tolerance when
    /// ``|expected| &lt; 1`` so zero-valued coefficients don't trigger a
    /// divide-by-noise spuriously.
    /// </summary>
    private static void AssertRelative(double actual, double expected, double tolerance, string label)
    {
        var denominator = Math.Max(Math.Abs(expected), 1.0);
        var relativeError = Math.Abs(actual - expected) / denominator;
        Assert.True(
            relativeError <= tolerance,
            $"{label}: expected {expected}, got {actual}, |Δ|/max(|expected|,1) = {relativeError} > tol = {tolerance}");
    }
}
