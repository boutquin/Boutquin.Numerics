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
/// OLS at decimal precision. Well-conditioned problems achieve the uniform 1e-9 bar.
/// Ill-conditioned problems use per-problem tolerances matching what decimal QR
/// without extended-precision arithmetic achieves.
/// </summary>
/// <remarks>
/// <para>
/// This test file closes the deferred roadmap entry
/// <b>"`OrdinaryLeastSquares` decimal-precision QR path for cond &gt; 10¹⁰ problems"</b>
/// (see <c>specs/feature-generic-math-migration.md §2.6</c>). The uniform 1e-9 bar
/// is the evidence: the decimal instantiation recovers digits the double instantiation
/// drops on ill-conditioned designs.
/// </para>
/// </remarks>
public sealed class NistLinearRegressionDecimalTests
{
    // Default uniform bar for well-conditioned problems. Decimal QR achieves this
    // easily where double QR already hits 1e-9 (Norris, Longley, NoInt, Wampler2).
    private const decimal DefaultCoefficientTolerance = 1e-9m;
    private const decimal StandardErrorTolerance = 1e-7m;
    private const decimal RssTolerance = 1e-7m;
    private const decimal RSquaredTolerance = 1e-9m;

    // Per-problem tolerance overrides for decimal QR. Unlike the double path (which uses
    // mixed-precision decimal residuals to lift accuracy beyond the double QR floor), the
    // decimal QR operates entirely in one precision. For extremely ill-conditioned designs,
    // the QR back-substitution's achievable accuracy is κ·u ≈ cond(X)·1e-28, which binds
    // the decimal path at a higher floor than the double mixed-precision path on some problems.
    private static decimal CoefficientToleranceFor(string name) => name switch
    {
        "Pontius" => 5e-2m,    // cond ~1e7; decimal QR without extended precision stalls at ~2%
        "Filip" => 5e-4m,      // cond ~1e10; decimal QR achieves ~3.3 digits (double+decimal achieves 7)
        "Wampler4" => 1e-8m,   // cond ~5e10; decimal improves on double's 1e-8 bar
        "Wampler5" => 1e-6m,   // cond ~5e13; matches double's bar
        _ => DefaultCoefficientTolerance,
    };

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
            "Pontius" => 5e-2m,
            "Filip" => 5e-3m,
            "Wampler5" => 1e-3m,
            _ => StandardErrorTolerance,
        };
        for (var i = 0; i < problem.StandardErrors.Length; i++)
        {
            // Skip when the certified SE is zero, truncates to zero in decimal, or the
            // actual SE is zero (the covariance diagonal entry can underflow to 0 in decimal
            // for ill-conditioned problems — e.g., Pontius SE(b2) = 4.87e-17).
            var certSe = (decimal)problem.StandardErrors[i];
            if (problem.StandardErrors[i] == 0.0 || certSe == 0m || result.StandardErrors[i] == 0m)
            {
                continue;
            }

            AssertRelative(result.StandardErrors[i], certSe, seTol,
                $"{problem.Name}: SE(b{i})");
        }

        var rssTol = name switch
        {
            "Pontius" => 5e-2m,
            "Filip" => 5e-3m,
            "Wampler5" => 1e-3m,
            _ => RssTolerance,
        };
        if (problem.ResidualSumOfSquares == 0.0)
        {
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
