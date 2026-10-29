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

using System.Globalization;

using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The reduced QP forms its free weights from the inverse solves as a budget part plus a
/// budget-orthogonal correction (<c>a·ρ + (b − m·a)/λ</c> for mean-variance, <c>cScale·z − zc</c>
/// for minimum variance). When the right-hand side lies almost entirely along a near-null direction
/// of <c>Σ_FF</c> that the budget direction also spans, the orthogonal correction is below the
/// rounding of the solves and the difference cancels to noise. The solver must still return the
/// exact optimum (or refuse with the documented exception); it must never return a vector that
/// violates the budget or the free-variable stationarity of its own KKT system. Each problem is
/// well posed: its exact optimum does not move under any one-ulp perturbation of an input. Expected
/// values are the exact rational KKT optimum (active-set enumeration in <c>fractions.Fraction</c>
/// on the exact inputs), rounded once to the working type — never the implementation's output.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpBudgetOrthogonalCancellationTests
{
    private static readonly CultureInfo s_invariant = CultureInfo.InvariantCulture;

    public static TheoryData<string> DoubleMeanVarianceCases => new() { "dbl-corr-6-28", "dbl-corr-6-33", "dbl-corr-6-34" };

    [Theory]
    [MemberData(nameof(DoubleMeanVarianceCases))]
    public void QpMeanVariance_DoubleBudgetOrthogonalPartBelowRounding_ReturnsTheExactOptimum(string caseId)
    {
        var (covariance, means, riskAversion, lower, upper, optimum) = DoubleMeanVarianceCase(caseId);

        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(covariance, means, riskAversion, lower, upper);

        AssertBudget(weights);
        var scale = optimum.Max(Math.Abs);
        for (var i = 0; i < optimum.Length; i++)
        {
            weights[i].Should().BeApproximately(
                optimum[i], scale * 1e-9, "weight {0} of {1} must be the exact KKT optimum", i, caseId);
        }
    }

    [Fact]
    public void QpMinVariance_DoubleCrossTermAlongBudgetNullDirection_ReturnsTheExactOptimum()
    {
        // The third asset's covariance with the near-singular pair is (almost) parallel to the budget
        // direction, which is the pair's near-null direction, so zc = Σ⁻¹·crossTerms is ~1e16 times
        // the free weight it must reproduce once the pair's lower bound pins two assets.
        double[,] covariance =
        {
            { 7.944624722998775e-35, -7.944624722998765e-35, 4.616822896156709e-18 },
            { -7.944624722998765e-35, 7.944624722998775e-35, 4.61682289615669e-18 },
            { 4.616822896156709e-18, 4.61682289615669e-18, 1.2386458468319458e16 },
        };
        const double lower = 0.07628629197909641;
        const double upper = 577865503.5082259;

        var weights = ActiveSetQpSolver<double>.SolveMinVariance(covariance, lower, upper);

        // Exact optimum: assets 0 and 2 at the lower bound, asset 1 takes 1 − 2·lb.
        double[] optimum = [0.07628629197909641, 0.8474274160418072, 0.07628629197909641];
        AssertBudget(weights);
        for (var i = 0; i < 3; i++)
        {
            weights[i].Should().BeApproximately(optimum[i], 1e-12, "weight {0} must be the exact KKT optimum", i);
        }
    }

    [Fact]
    public void QpMeanVariance_DecimalBudgetOrthogonalPartBelowRounding_ReturnsTheExactOptimum()
    {
        // Assets 0, 2, 3 form a cluster whose near-null direction (eigenvalue 8e-28) is the budget
        // direction, and their means lie along it to within 1.2e-8 relative, so in the all-free solve
        // b's budget-orthogonal part is 4.9e-29 of b: below decimal's 28-digit rounding.
        const decimal c = -0.0000000399999999999999999996m;
        decimal[,] covariance =
        {
            { 0.00000008m, 0m, c, c },
            { 0m, 0.0000000009m, 0m, 0m },
            { c, 0m, 0.00000008m, c },
            { c, 0m, c, 0.00000008m },
        };
        decimal[] means = [89999999435826m, -90000000000000m, 90000000532871m, 89999999653984m];

        var weights = ActiveSetQpSolver<decimal>.SolveMeanVariance(covariance, means, 80000m, -6000000m, 60000000m);

        // Exact optimum: assets 0, 1, 3 at the lower bound, asset 2 takes 1 − 3·lb.
        decimal[] optimum = [-6000000m, -6000000m, 18000001m, -6000000m];
        weights.Sum().Should().Be(1m, "the documented postcondition is an optimal weight vector summing to 1");
        for (var i = 0; i < 4; i++)
        {
            weights[i].Should().BeApproximately(optimum[i], 1e-12m, "weight {0} must be the exact KKT optimum", i);
        }
    }

    [Fact]
    public void QpMinVariance_DecimalCrossTermAlongBudgetNullDirection_ReturnsTheExactOptimum()
    {
        decimal[,] covariance =
        {
            { 0.00008m, -0.00007999999999999992m, 0.000000435259000000000019891m },
            { -0.00007999999999999992m, 0.00008m, 0.000000435259000000000019891m },
            { 0.000000435259000000000019891m, 0.000000435259000000000019891m, 800000000m },
        };
        const decimal lower = 0.068333333333333333333333333m;

        var weights = ActiveSetQpSolver<decimal>.SolveMinVariance(covariance, lower, 100000000m);

        // Exact optimum: the high-variance asset 2 at the lower bound, the pair splits the rest equally
        // ((1 − lb)/2 = 931666666666666666666666667 / 2·10^27).
        decimal[] optimum = [0.4658333333333333333333333335m, 0.4658333333333333333333333335m, lower];
        Math.Abs(weights.Sum() - 1m).Should().BeLessThanOrEqualTo(
            1e-26m, "the documented postcondition is an optimal weight vector summing to 1");
        for (var i = 0; i < 3; i++)
        {
            weights[i].Should().BeApproximately(optimum[i], 1e-20m, "weight {0} must be the exact KKT optimum", i);
        }
    }

    private static void AssertBudget(double[] weights)
    {
        var magnitude = 1.0 + weights.Sum(Math.Abs);
        Math.Abs(weights.Sum() - 1.0).Should().BeLessThanOrEqualTo(
            magnitude * 1e-12, "the documented postcondition is an optimal weight vector summing to 1");
    }

    private static (double[,] Covariance, double[] Means, double RiskAversion, double Lower, double Upper, double[] Optimum)
        DoubleMeanVarianceCase(string caseId) => caseId switch
        {
            // The investigator's witness: HEAD returned (8.33e47, 0.5, 0.5), summing to 8.33e47.
            "dbl-corr-6-28" => (
                Matrix(
                    ["1.8637575559325146e-20", "0.0", "0.0"],
                    ["0.0", "7.216096801090916e-149", "-7.216096800779981e-149"],
                    ["0.0", "-7.216096800779981e-149", "7.216096801090916e-149"]),
                Vector("5.120972222429012e+149", "-5.1209722259932635e+149", "-5.120972224737676e+149"),
                6.594700997873288e121, -3.8991196973818446e117, 8.734655991597823e142,
                Vector("8.33293298554354e+47", "-3.8991196973818446e+117", "3.8991196973818446e+117")),
            "dbl-corr-6-33" => (
                Matrix(
                    ["8.997117877635822e-149", "-4.498558938791854e-149", "0.0", "-4.498558938791854e-149"],
                    ["-4.498558938791854e-149", "8.997117877635822e-149", "0.0", "-4.498558938791854e-149"],
                    ["0.0", "0.0", "6.7298290175244e-36", "0.0"],
                    ["-4.498558938791854e-149", "-4.498558938791854e-149", "0.0", "8.997117877635822e-149"]),
                Vector("5.66138339055944e+148", "5.661383397106605e+148", "-5.661383393228222e+148", "5.661383388567183e+148"),
                5.484908069333786e148, -4.968973750588688e144, 7.680967052491104e132,
                Vector("7.680967052491104e+132", "7.680967052491104e+132", "-3.0674619114718302e+35", "-1.5361934104982209e+133")),
            "dbl-corr-6-34" => (
                Matrix(
                    ["4.891281237147999e-150", "-4.891281236782672e-150", "0.0"],
                    ["-4.891281236782672e-150", "4.891281237147999e-150", "0.0"],
                    ["0.0", "0.0", "8.92438018390442e-98"]),
                Vector("-6.418137470001299e+148", "-6.418137470023383e+148", "6.418137470309765e+148"),
                3.947321690427148e33, -7.547777805633202e147, 2.7595529718888005e124,
                Vector("2.7595529718888005e+124", "-5.519105943777601e+124", "2.7595529718888005e+124")),
            _ => throw new ArgumentOutOfRangeException(nameof(caseId), caseId, "Unknown case."),
        };

    private static double[] Vector(params string[] values) =>
        values.Select(v => double.Parse(v, NumberStyles.Float, s_invariant)).ToArray();

    private static double[,] Matrix(params string[][] rows)
    {
        var n = rows.Length;
        var matrix = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                matrix[i, j] = double.Parse(rows[i][j], NumberStyles.Float, s_invariant);
            }
        }

        return matrix;
    }
}
