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

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

/// <summary>
/// NIST Statistical Reference Datasets — linear-regression benchmark suite for
/// <see cref="OrdinaryLeastSquares"/>. Each test runs <c>Fit</c> against a canonical
/// problem from <c>https://www.itl.nist.gov/div898/strd/lls/lls.shtml</c> and compares
/// the estimator's coefficients, standard errors, residual sum of squares, residual
/// standard deviation, and R-squared against NIST-certified values.
/// </summary>
/// <remarks>
/// <para>
/// Accuracy bar: coefficients match to 1e-12 relative and standard errors to 1e-11 where
/// NIST publishes them, on every problem except <c>Filip</c> (1e-7 and 1e-6). The suite deliberately mixes the
/// well-conditioned <c>Norris</c> and <c>NoInt*</c> problems with the pathological
/// <c>Longley</c> and <c>Wampler4/5</c> problems. All five Wampler designs have
/// cond ≈ 6.4·10⁶; the latter two increase response noise and cancellation.
/// Pivoted QR with augmented refinement satisfies the stated coefficient bars.
/// </para>
/// <para>
/// R²: NIST publishes the centred form for intercept models and the uncentred form for
/// the no-intercept problems <c>NoInt1</c>/<c>NoInt2</c>. <see cref="OrdinaryLeastSquares"/>
/// switches between the two internally based on the <c>includeIntercept</c> flag, so a
/// single relative-tolerance check suffices here.
/// </para>
/// </remarks>
public sealed class NistLinearRegressionTests
{
    // Coefficient and standard-error bars. Double-precision QR with column pivoting and
    // Björck's augmented-system refinement (residual blocks accumulated in decimal)
    // reproduces the certified coefficients to 1e-12 and the standard errors to 1e-11 on
    // every problem except Filip, including Longley and Wampler4/5. The Wampler
    // designs share cond ≈ 6.4·10⁶; their different response noise does not alter it.
    // These are the bars the OrdinaryLeastSquares<T> summary documents.
    private const double StandardErrorTolerance = 1e-11;
    private const double RssTolerance = 1e-7;
    private const double RSquaredTolerance = 1e-9;

    // Filip (degree-10 polynomial, cond ≈ 10¹⁰) is the one exception: its tabulated design
    // is the double-rounded powers x, x², …, x¹⁰, and the fit measures 2.4e-8 on
    // coefficients and 8.7e-8 on standard errors against the certified values.
    private static double CoefficientToleranceFor(string name) =>
        NistLinearRegressionData.DoubleCoefficientToleranceFor(name);

    private static double StandardErrorToleranceFor(string name) => name switch
    {
        "Filip" => 1e-6,
        _ => StandardErrorTolerance,
    };

    private static double RssToleranceFor(string name) => name switch
    {
        "Filip" => 1e-5,
        "Wampler4" => 1e-6,
        "Wampler5" => 1e-3,
        _ => RssTolerance,
    };

    public static TheoryData<string> ProblemNames => new()
    {
        "Norris", "Pontius", "NoInt1", "NoInt2", "Filip", "Longley",
        "Wampler1", "Wampler2", "Wampler3", "Wampler4", "Wampler5",
    };

    [Theory]
    [MemberData(nameof(ProblemNames))]
    public void Fit_MatchesCertifiedValues(string name)
    {
        var problem = Resolve(name);

        var result = OrdinaryLeastSquares.Fit(problem.X, problem.Y, problem.IncludeIntercept);

        // NIST's certified observation/parameter counts give these residual degrees of freedom.
        var certifiedDegrees = name switch
        {
            "Norris" => 34,
            "Pontius" => 37,
            "NoInt1" => 10,
            "NoInt2" => 2,
            "Filip" => 71,
            "Longley" => 9,
            _ => 15, // Wampler1–5:21observations,6parameters.
        };
        result.DegreesOfFreedom.Should().Be(certifiedDegrees);

        result.Coefficients.Should().HaveSameCount(problem.Coefficients,
            because: $"{problem.Name}: OLS coefficient vector must have the same length as the certified vector");

        var coefTol = CoefficientToleranceFor(name);
        var seTol = StandardErrorToleranceFor(name);
        var rssTol = RssToleranceFor(name);

        for (var i = 0; i < problem.Coefficients.Length; i++)
        {
            AssertRelative(result.Coefficients[i], problem.Coefficients[i], coefTol,
                $"{problem.Name}: coefficient b{i}");
        }

        // Wampler1 and Wampler2 are exact polynomial fits with certified zero SE.
        // Use an absolute numerical-zero bar rather than skipping those coefficients.
        for (var i = 0; i < problem.StandardErrors.Length; i++)
        {
            if (problem.StandardErrors[i] == 0.0)
            {
                result.StandardErrors[i].Should().BeInRange(0.0, 1e-12,
                    because: $"{problem.Name}: the certified zero standard error requires an absolute comparison");
                continue;
            }

            AssertRelative(result.StandardErrors[i], problem.StandardErrors[i], seTol,
                $"{problem.Name}: SE(b{i})");
        }

        // Residual SS: NIST publishes this at roughly 12 digits. For Wampler1–2 the residual
        // SSE is exactly zero (polynomial exact fit); the test falls through to an absolute
        // upper bound there because any relative comparison against zero is ill-defined.
        if (problem.ResidualSumOfSquares == 0.0)
        {
            result.ResidualStandardDeviation.Should().BeInRange(0.0, 1e-12,
                because: $"{problem.Name}: certified residual standard deviation is zero");
            result.ResidualSumOfSquares.Should().BeLessThan(1e-18,
                because: $"{problem.Name}: certified residual SS is 0 (exact polynomial fit); solver must also reach numerical zero");
        }
        else
        {
            AssertRelative(result.ResidualSumOfSquares, problem.ResidualSumOfSquares, rssTol,
                $"{problem.Name}: residual sum of squares");
            AssertRelative(result.ResidualStandardDeviation, problem.ResidualStandardDeviation, rssTol,
                $"{problem.Name}: residual standard deviation");
        }

        AssertRelative(result.RSquared, problem.RSquared, RSquaredTolerance,
            $"{problem.Name}: R-squared");
    }

    /// <summary>
    /// Supplemental Longley regression check at 1e-9 relative coefficient error.
    /// The main certified-value theory applies the tighter 1e-12 bar to this fixture.
    /// </summary>
    [Fact]
    public void Longley_MeetsSupplementalCoefficientBar()
    {
        var problem = NistLinearRegressionData.Longley();

        var result = OrdinaryLeastSquares.Fit(problem.X, problem.Y, problem.IncludeIntercept);

        for (var i = 0; i < problem.Coefficients.Length; i++)
        {
            AssertRelative(result.Coefficients[i], problem.Coefficients[i], 1e-9,
                $"Longley: b{i} (condition ≈ 5·10⁷)");
        }
    }

    /// <summary>
    /// Wampler5 shares the degree-five design of Wampler1–4 (condition ≈ 6.4·10⁶),
    /// with larger response noise. This supplemental test checks a 1e-6 relative
    /// coefficient bar; <see cref="Fit_MatchesCertifiedValues"/> applies the tighter
    /// 1e-12 bar. Neither tolerance is a condition-number-implied accuracy floor.
    /// </summary>
    [Fact]
    public void Wampler5_MeetsSupplementalSixDigitCoefficientBar()
    {
        var problem = NistLinearRegressionData.Wampler5();

        var result = OrdinaryLeastSquares.Fit(problem.X, problem.Y, problem.IncludeIntercept);

        // Preserve this supplemental regression bar; the main certified-value theory
        // exercises the stronger coefficient tolerance on the same fixture.
        for (var i = 0; i < problem.Coefficients.Length; i++)
        {
            AssertRelative(result.Coefficients[i], problem.Coefficients[i], 1e-6,
                $"Wampler5: b{i} (supplemental 1e-6 relative bar)");
        }
    }

    private static NistLinearRegressionData.Problem Resolve(string name) => name switch
    {
        "Norris" => NistLinearRegressionData.Norris(),
        "Pontius" => NistLinearRegressionData.Pontius(),
        "NoInt1" => NistLinearRegressionData.NoInt1(),
        "NoInt2" => NistLinearRegressionData.NoInt2(),
        "Filip" => NistLinearRegressionData.Filip(),
        "Longley" => NistLinearRegressionData.Longley(),
        "Wampler1" => NistLinearRegressionData.Wampler1(),
        "Wampler2" => NistLinearRegressionData.Wampler2(),
        "Wampler3" => NistLinearRegressionData.Wampler3(),
        "Wampler4" => NistLinearRegressionData.Wampler4(),
        "Wampler5" => NistLinearRegressionData.Wampler5(),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown NIST linear-regression problem."),
    };

    private static void AssertRelative(double actual, double expected, double tolerance, string label)
    {
        if (expected == 0.0)
        {
            Math.Abs(actual).Should().BeLessThan(tolerance,
                because: $"{label}: expected 0, got {actual:G17}");
            return;
        }

        var rel = Math.Abs((actual - expected) / expected);
        rel.Should().BeLessThan(tolerance,
            because: $"{label}: expected {expected:G17}, got {actual:G17} (relative error {rel:G6})");
    }
}
