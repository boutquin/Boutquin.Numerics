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

using HeldOutOutcome = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutOutcome;
using HeldOutProblem = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutProblem;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// The conditioning refusal of design step 4 through <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/> (design
/// step 5: mean variance reuses the same machinery). A certified result is refused, naming stationarity, when
/// <c>48·n·u·κ_eq(B) ≥ 1</c> for the block <c>B</c> that decides it; the documented refusal class covers both entry
/// points. Each case is the mean-variance analogue of a min-variance pin in <see cref="ActiveSetQpCorpusTests"/> or
/// <see cref="ActiveSetQpDocumentedContractTests"/>, at <see cref="double"/> and <see cref="decimal"/>, with means
/// orthogonal to the near-null direction so that the exact optimum lies strictly inside the box and away from the
/// equal-weight point (<c>B = F*</c>, the whole free block). Both directions are pinned: past the margin the solver
/// throws the documented <see cref="InvalidOperationException"/>; inside it the solver returns the optimum.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpMeanVarianceConditioningRefusalTests
{
    // μ = (1/2, 1/4, 3/8, 1/8): exact in both types, s·μ = 0 for s = (1, −1, −1, 1), Σμ = 5/4. With λ = 1 the
    // exact optimum is w = μ − ν·1 with ν = (5/4 − 1)/4 = 1/16, so w = (7/16, 3/16, 5/16, 1/16) whatever ε is,
    // because Σ acts as the identity on the complement of s, which holds μ and the budget row.
    private static readonly string[] s_means = ["0.5", "0.25", "0.375", "0.125"];
    private static readonly double[] s_optimum = [0.4375, 0.1875, 0.3125, 0.0625];

    private const string RefusalMessage = "*stationarity cannot be certified*48·n·u·κ ≥ 1*";

    /// <summary>
    /// <c>Σ = I − (1−ε)/4·ssᵀ</c> with <c>s = (1, −1, −1, 1)</c> has the budget-orthogonal near-null direction
    /// <c>s</c> of eigenvalue <c>ε</c> and <c>κ₁ = (3/2 − ε/2)/ε</c>; the equilibration leaves its unit-scale
    /// diagonal unchanged. Double: <c>ε = k·2⁻⁵⁰</c>, <c>48·n·u·κ_eq</c> = 18, 9 and 1.03 at <c>k</c> = 2, 4 and 35.
    /// Decimal: <c>ε</c> = 1e-26 and 2.8e-26, <c>48·n·u·κ_eq</c> = 2.88 and 1.03. Every case is refusal-permitted
    /// and its optimum is interior, so the conditioning guard must refuse it naming stationarity.
    /// </summary>
    [Theory]
    [InlineData("double", "2")]
    [InlineData("double", "4")]
    [InlineData("double", "35")]
    [InlineData("decimal", "1E-26")]
    [InlineData("decimal", "2.8E-26")]
    public void SolveMeanVariance_NearSingularBudgetOrthogonalBlockPastTheMargin_IsRefusedNamingStationarity(string type, string epsilon)
    {
        var problem = NearSingularBudgetOrthogonalBlock(type, epsilon);
        var classified = ActiveSetQpCorpusTests.SolveHeldOut(problem);
        classified.Qualified.Should().BeFalse("48·n·u·κ_eq ≥ 1 makes the block refusal-permitted: " + classified.Detail);
        classified.Outcome.Should().Be(HeldOutOutcome.Refused, "the conditioning guard must refuse the deciding block: " + classified.Detail);
        classified.Detail.Should().Contain("stationarity");

        var solve = Solve(problem);
        solve.Should().Throw<InvalidOperationException>().WithMessage(RefusalMessage);
    }

    /// <summary>
    /// The same shape inside the margin (<c>48·n·u·κ_eq</c> = 0.56 and 0.14 at double <c>k</c> = 64 and 256, 0.58 and
    /// 0.14 at decimal <c>ε</c> = 5e-26 and 2e-25): qualified, so the solver must return the exact optimum within the
    /// AC-3.1 bound rather than refuse, which shows the refusal above is the margin and not the shape.
    /// </summary>
    [Theory]
    [InlineData("double", "64")]
    [InlineData("double", "256")]
    [InlineData("decimal", "5E-26")]
    [InlineData("decimal", "2E-25")]
    public void SolveMeanVariance_QualifiedNearSingularBudgetOrthogonalBlock_IsSolvedNotRefused(string type, string epsilon)
    {
        var problem = NearSingularBudgetOrthogonalBlock(type, epsilon);
        var classified = ActiveSetQpCorpusTests.SolveHeldOut(problem);
        classified.Qualified.Should().BeTrue("48·n·u·κ_eq < 1 qualifies the block: " + classified.Detail);
        classified.Outcome.Should().Be(HeldOutOutcome.Ok, classified.Detail);

        var weights = Solve(problem)();
        for (var i = 0; i < 4; i++)
        {
            weights[i].Should().BeApproximately(s_optimum[i], 1e-9, "weight " + i + " of the interior optimum (7, 3, 5, 1)/16");
        }
    }

    /// <summary>
    /// Near-duplicate assets with equal means: <c>Σ = [[1, 1−ε], [1−ε, 1]]</c>, <c>μ = (1/20, 1/20)</c>, <c>λ = 2</c>,
    /// box <c>[0, 1]</c>, interior optimum <c>(½, ½)</c>, <c>κ₁ = (2−ε)/ε</c>. Double: <c>ε = 1e-14</c> gives
    /// <c>48·2·2⁻⁵³·κ₁ ≈ 2.1</c>, refused; <c>ε = 1e-13</c> gives 0.21, answered. Decimal: <c>ε = 1e-26</c> gives
    /// <c>48·2·10⁻²⁸·κ₁ ≈ 1.92</c>, refused; <c>ε = 1e-24</c> gives 0.019, answered.
    /// </summary>
    [Theory]
    [InlineData("double", "0.99999999999999", "0.9999999999999")]
    [InlineData("decimal", "0.99999999999999999999999999", "0.999999999999999999999999")]
    public void SolveMeanVariance_NearDuplicateAssetsPastTheConditioningMargin_IsRefusedNamingStationarity_AndTheMarginIsTheBoundary(
        string type, string refusedCorrelation, string answeredCorrelation)
    {
        var refused = NearDuplicateAssets(type, refusedCorrelation);
        var answered = NearDuplicateAssets(type, answeredCorrelation);

        var refusal = Solve(refused);
        refusal.Should().Throw<InvalidOperationException>().WithMessage(RefusalMessage);
        ActiveSetQpCorpusTests.SolveHeldOut(refused).Qualified.Should().BeFalse();

        var weights = Solve(answered)();
        weights[0].Should().BeApproximately(0.5, 1e-12);
        weights[1].Should().BeApproximately(0.5, 1e-12);
        ActiveSetQpCorpusTests.SolveHeldOut(answered).Qualified.Should().BeTrue();
    }

    private static HeldOutProblem NearSingularBudgetOrthogonalBlock(string type, string epsilon)
    {
        string diagonal;
        string off;
        if (type == "double")
        {
            var quarterEps = Math.ScaleB(int.Parse(epsilon, CultureInfo.InvariantCulture), -52);
            diagonal = (0.75 + quarterEps).ToString("R", CultureInfo.InvariantCulture);
            off = (0.25 - quarterEps).ToString("R", CultureInfo.InvariantCulture);
        }
        else
        {
            var quarterEps = decimal.Parse(epsilon, NumberStyles.Float, CultureInfo.InvariantCulture) / 4m;
            diagonal = (0.75m + quarterEps).ToString(CultureInfo.InvariantCulture);
            off = (0.25m - quarterEps).ToString(CultureInfo.InvariantCulture);
        }

        int[] s = [1, -1, -1, 1];
        var cov = new string[4][];
        for (var i = 0; i < 4; i++)
        {
            cov[i] = new string[4];
            for (var j = 0; j < 4; j++)
            {
                cov[i][j] = i == j ? diagonal : (s[i] * s[j] > 0 ? "-" : "") + off;
            }
        }

        return new HeldOutProblem("mv-near-singular-" + epsilon, type, cov, "-1", "2", s_means, "1");
    }

    private static HeldOutProblem NearDuplicateAssets(string type, string correlation) =>
        new("mv-near-duplicate-" + correlation, type, [["1", correlation], [correlation, "1"]], "0", "1", ["0.05", "0.05"], "2");

    // The public SolveMeanVariance of the problem's working type, its weights as double for the assertions.
    private static Func<double[]> Solve(HeldOutProblem p)
    {
        var n = p.Cov.Length;
        if (p.Type == "double")
        {
            double D(string x) => double.Parse(x, NumberStyles.Float, CultureInfo.InvariantCulture);
            var cov = new double[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    cov[i, j] = D(p.Cov[i][j]);
                }
            }

            var mu = p.Mu!.Select(D).ToArray();
            return () => ActiveSetQpSolver<double>.SolveMeanVariance(cov, mu, D(p.Lambda!), D(p.Lb), D(p.Ub));
        }

        decimal M(string x) => decimal.Parse(x, NumberStyles.Float, CultureInfo.InvariantCulture);
        var covM = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                covM[i, j] = M(p.Cov[i][j]);
            }
        }

        var muM = p.Mu!.Select(M).ToArray();
        return () => ActiveSetQpSolver<decimal>.SolveMeanVariance(covM, muM, M(p.Lambda!), M(p.Lb), M(p.Ub)).Select(x => (double)x).ToArray();
    }
}
