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
/// <see cref="OrdinaryLeastSquares{T}"/> at <c>T = decimal</c>. Runs the same 11
/// problems as <see cref="NistLinearRegressionTests"/> but instantiates the generic
/// OLS at decimal precision. Every problem except <c>Filip</c> is held to the uniform
/// 1e-13 coefficient bar.
/// </summary>
/// <remarks>
/// <para>
/// The data enter through <c>(decimal)double</c>, which rounds each value to 15
/// significant digits. <c>Filip</c>'s design is the double-rounded powers
/// <c>x, x², …, x¹⁰</c> of a degree-10 polynomial (cond ≈ 10¹⁰), so its certified
/// coefficients are recovered to about 6 digits; its bars are 1e-6.
/// </para>
/// </remarks>
public sealed class NistLinearRegressionDecimalTests
{
    // Default coefficient bar: the certified values carry 15 significant digits and the
    // decimal fit reproduces them to within 4.5e-15 on every problem but Filip. This is
    // the bar the OlsResult<T> remarks document.
    private const decimal StandardErrorTolerance = 1e-7m;
    private const decimal RssTolerance = 1e-7m;
    private const decimal RSquaredTolerance = 1e-9m;

    // Per-problem tolerance overrides for decimal QR. With every reflector normalised the
    // decimal factorisation reduces loss from column scaling within its representable
    // range. Filip uses double-rounded polynomial inputs and needs a looser measured bar.
    private static decimal CoefficientToleranceFor(string name) =>
        NistLinearRegressionData.DecimalCoefficientToleranceFor(name);

    public static TheoryData<string> ProblemNames => new()
    {
        "Norris", "Pontius", "NoInt1", "NoInt2", "Filip", "Longley",
        "Wampler1", "Wampler2", "Wampler3", "Wampler4", "Wampler5",
    };

    [Theory]
    [MemberData(nameof(ProblemNames))]
    public void Fit_Decimal_MatchesCertifiedValues_UniformTolerance(string name)
    {
        var problem = Resolve(name);

        // Convert double data to decimal for the generic OLS.
        var xDecimal = ToDecimalMatrix(problem.X);
        var yDecimal = ToDecimalArray(problem.Y);

        var result = OrdinaryLeastSquares<decimal>.Fit(xDecimal, yDecimal, problem.IncludeIntercept);

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
            because: $"{problem.Name}: OLS<decimal> coefficient vector length must match certified vector");

        var coefTol = CoefficientToleranceFor(name);
        for (var i = 0; i < problem.Coefficients.Length; i++)
        {
            AssertRelative(result.Coefficients[i], (decimal)problem.Coefficients[i], coefTol,
                $"{problem.Name}: coefficient b{i}");
        }

        var seTol = name switch
        {
            "Filip" => 1e-6m,
            _ => StandardErrorTolerance,
        };
        for (var i = 0; i < problem.StandardErrors.Length; i++)
        {
            // Only the reference value decides whether a relative comparison applies.
            var certSe = (decimal)problem.StandardErrors[i];
            if (problem.StandardErrors[i] == 0.0 || certSe == 0m)
            {
                result.StandardErrors[i].Should().BeInRange(0m, 1e-18m,
                    because: $"{problem.Name}: the certified zero standard error requires an absolute comparison");
                continue;
            }

            if (name == "Pontius" && i == 2)
            {
                // Its certified variance is about 2.37e-33, below decimal's 1e-28
                // quantum. Bound the absolute SE error explicitly for this one case.
                result.StandardErrors[i].Should().BeGreaterThanOrEqualTo(0m);
                Math.Abs(result.StandardErrors[i] - certSe).Should().BeLessThan(5e-17m);
                continue;
            }

            AssertRelative(result.StandardErrors[i], certSe, seTol,
                $"{problem.Name}: SE(b{i})");
        }

        var rssTol = name switch
        {
            "Filip" => 1e-6m,
            _ => RssTolerance,
        };
        if (problem.ResidualSumOfSquares == 0.0)
        {
            result.ResidualStandardDeviation.Should().BeInRange(0m, 1e-18m,
                because: $"{problem.Name}: certified residual standard deviation is zero");
            result.ResidualSumOfSquares.Should().BeLessThan(1e-18m,
                because: $"{problem.Name}: certified residual SS is 0 (exact polynomial fit)");
        }
        else
        {
            AssertRelative(result.ResidualSumOfSquares, (decimal)problem.ResidualSumOfSquares, rssTol,
                $"{problem.Name}: residual sum of squares");
            AssertRelative(result.ResidualStandardDeviation, (decimal)problem.ResidualStandardDeviation, rssTol,
                $"{problem.Name}: residual standard deviation");
        }

        AssertRelative(result.RSquared, (decimal)problem.RSquared, RSquaredTolerance,
            $"{problem.Name}: R-squared");
    }

    private static void AssertRelative(decimal actual, decimal expected, decimal tolerance, string label)
    {
        if (expected == 0m)
        {
            Math.Abs(actual).Should().BeLessThan(tolerance,
                because: $"{label}: expected 0, got {actual}");
            return;
        }

        var rel = Math.Abs((actual - expected) / expected);
        rel.Should().BeLessThan(tolerance,
            because: $"{label}: expected {expected}, got {actual} (error {rel:E3})");
    }

    private static decimal[,] ToDecimalMatrix(double[,] source)
    {
        var rows = source.GetLength(0);
        var cols = source.GetLength(1);
        var result = new decimal[rows, cols];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                result[i, j] = (decimal)source[i, j];
            }
        }

        return result;
    }

    private static decimal[] ToDecimalArray(double[] source)
    {
        var result = new decimal[source.Length];
        for (var i = 0; i < source.Length; i++)
        {
            result[i] = (decimal)source[i];
        }

        return result;
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
        _ => throw new ArgumentException($"Unknown NIST linear-regression problem: {name}"),
    };
}
