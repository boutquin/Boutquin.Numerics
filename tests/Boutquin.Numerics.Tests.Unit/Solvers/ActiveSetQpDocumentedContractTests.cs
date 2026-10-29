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
/// Pins the documented <see cref="ActiveSetQpSolver{T}"/> contract in <c>docs/solvers.md</c> and the
/// <c>[Unreleased]</c> section of <c>CHANGELOG.md</c> to the behaviour the solver executes: every input
/// rejection the page lists is thrown with the listed type, the documented sample comment holds on the inputs
/// it is stated for, and the retired mechanisms of the previous core are no longer described.
/// </summary>
public sealed class ActiveSetQpDocumentedContractTests
{
    private static readonly double[,] s_identity2 = { { 1, 0 }, { 0, 1 } };

    /// <summary>Every input rejection the solver executes: the call, the exact exception type, the page keyword.</summary>
    public static TheoryData<string, Action, Type, string, string?> Rejections => new()
    {
        { "null covariance", () => ActiveSetQpSolver<double>.SolveMinVariance(null!, 0.0, 1.0), typeof(ArgumentNullException), "ArgumentNullException", null },
        { "zero dimension", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[0, 0], 0.0, 1.0), typeof(ArgumentException), "zero dimension", null },
        { "non-square", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[2, 3], 0.0, 1.0), typeof(ArgumentException), "not square", null },
        { "asymmetric", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 0.5 }, { 0.1, 1 } }, 0.0, 1.0), typeof(ArgumentException), "not symmetric", null },
        { "non-finite covariance entry", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 0 }, { 0, double.NaN } }, 0.0, 1.0), typeof(ArgumentException), "non-finite entry", null },
        { "means length mismatch", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1 }, 1.0, 0.0, 1.0), typeof(ArgumentException), "does not match", null },
        { "non-finite mean", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, double.PositiveInfinity }, 1.0, 0.0, 1.0), typeof(ArgumentException), "mean is non-finite", null },
        { "minWeight above maxWeight", () => ActiveSetQpSolver<double>.SolveMinVariance(s_identity2, 0.6, 0.4), typeof(ArgumentException), "greater than `maxWeight`", null },
        { "lambda 0 box with n*max below 1", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, 0.2 }, 0.0, 0.0, 0.4), typeof(ArgumentException), "`riskAversion = 0`", null },
        { "lambda 0 box with n*min above 1", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, 0.2 }, 0.0, 0.6, 1.0), typeof(ArgumentException), "`riskAversion = 0`", null },
        { "negative riskAversion", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, 0.2 }, -1.0, 0.0, 1.0), typeof(ArgumentOutOfRangeException), "ArgumentOutOfRangeException", null },
        { "non-finite riskAversion", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, 0.2 }, double.NaN, 0.0, 1.0), typeof(ArgumentOutOfRangeException), "ArgumentOutOfRangeException", null },
        { "non-finite minWeight", () => ActiveSetQpSolver<double>.SolveMinVariance(s_identity2, double.NegativeInfinity, 1.0), typeof(ArgumentOutOfRangeException), "non-finite", null },
        { "non-finite maxWeight", () => ActiveSetQpSolver<double>.SolveMinVariance(s_identity2, 0.0, double.PositiveInfinity), typeof(ArgumentOutOfRangeException), "non-finite", null },
        { "min-variance non-positive-definite", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 2 }, { 2, 1 } }, 0.0, 1.0), typeof(InvalidOperationException), "not positive definite", null },
        { "mean-variance non-positive-definite", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 2 }, { 2, 1 } }, new double[] { 0.1, 0.2 }, 1.0, 0.0, 1.0), typeof(InvalidOperationException), "not positive definite", null },
        { "lambda 0 non-square covariance", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, 0, 0 }, { 0, 1, 0 } }, new double[] { 0.1, 0.2 }, 0.0, 0.0, 1.0), typeof(ArgumentException), "not square", "covariance" },
        { "lambda 0 non-square covariance (decimal)", () => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[,] { { 1, 0, 0 }, { 0, 1, 0 } }, new decimal[] { 0.1m, 0.2m }, 0m, 0m, 1m), typeof(ArgumentException), "not square", "covariance" },
        { "lambda 0 NaN mean", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { double.NaN, 0.2 }, 0.0, 0.0, 1.0), typeof(ArgumentException), "mean is non-finite", "means" },
        { "lambda 0 infinite mean", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, double.PositiveInfinity }, 0.0, 0.0, 1.0), typeof(ArgumentException), "mean is non-finite", "means" },
        { "Half", () => ActiveSetQpSolver<Half>.SolveMinVariance(new Half[,] { { Half.One, Half.Zero }, { Half.Zero, Half.One } }, Half.Zero, Half.One), typeof(NotSupportedException), "NotSupportedException", null },
        { "non-square 1x2 (min-variance)", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[1, 2] { { 1, 0 } }, 0.0, 1.0), typeof(ArgumentException), "not square", "covariance" },
        { "non-square 1x2 (min-variance, decimal)", () => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[1, 2] { { 1, 0 } }, 0m, 1m), typeof(ArgumentException), "not square", "covariance" },
        { "non-square 3x2 (min-variance)", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[3, 2] { { 1, 0 }, { 0, 1 }, { 0, 0 } }, 0.0, 1.0), typeof(ArgumentException), "not square", "covariance" },
        { "non-square 3x2 (min-variance, decimal)", () => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[3, 2] { { 1, 0 }, { 0, 1 }, { 0, 0 } }, 0m, 1m), typeof(ArgumentException), "not square", "covariance" },
        { "non-square 1x2 (mean-variance)", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[1, 2] { { 1, 0 } }, new double[] { 0.1 }, 1.0, 0.0, 1.0), typeof(ArgumentException), "not square", "covariance" },
        { "non-square 3x2 (mean-variance, decimal)", () => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[3, 2] { { 1, 0 }, { 0, 1 }, { 0, 0 } }, new decimal[] { 0.1m, 0.2m, 0.3m }, 1m, 0m, 1m), typeof(ArgumentException), "not square", "covariance" },
        { "infinite off-diagonal covariance entry (min-variance)", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, double.PositiveInfinity }, { double.PositiveInfinity, 1 } }, 0.0, 1.0), typeof(ArgumentException), "non-finite entry", "covariance" },
        { "infinite off-diagonal covariance entry (mean-variance)", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, double.PositiveInfinity }, { double.PositiveInfinity, 1 } }, new double[] { 0.1, 0.2 }, 1.0, 0.0, 1.0), typeof(ArgumentException), "non-finite entry", "covariance" },
        { "lambda 0 infinite off-diagonal covariance entry", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, double.NegativeInfinity }, { double.NegativeInfinity, 1 } }, new double[] { 0.1, 0.2 }, 0.0, 0.0, 1.0), typeof(ArgumentException), "non-finite entry", "covariance" },
        { "lambda 0 LP capacity alone not finite", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1 } }, new double[] { 2 }, 0.0, -1e308, 1e308), typeof(InvalidOperationException), "budget or capacity is not finite", null },
        { "lambda 0 LP budget alone not finite", () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 2, 1 }, 0.0, -1e308, 1.0), typeof(InvalidOperationException), "budget or capacity is not finite", null },
    };

    /// <summary>
    /// The entry guards themselves refuse every non-square shape and every non-finite entry, with their own messages: a
    /// later check (the covariance scaling's own square check, an index out of range, or the positive-definiteness test)
    /// must not be what refuses, and a shape that returns before those checks (one row) must not be answered.
    /// </summary>
    public static TheoryData<string, Action, string> EntryGuardMessages => new()
    {
        { "1x2 min-variance double", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[1, 2] { { 1, 0 } }, 0.0, 1.0), "Covariance matrix must be square." },
        { "2x1 min-variance double", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[2, 1] { { 1 }, { 1 } }, 0.0, 1.0), "Covariance matrix must be square." },
        { "2x3 min-variance double", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[2, 3] { { 1, 0, 0 }, { 0, 1, 0 } }, 0.0, 1.0), "Covariance matrix must be square." },
        { "3x2 min-variance double", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[3, 2] { { 1, 0 }, { 0, 1 }, { 0, 0 } }, 0.0, 1.0), "Covariance matrix must be square." },
        { "1x2 min-variance decimal", () => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[1, 2] { { 1, 0 } }, 0m, 1m), "Covariance matrix must be square." },
        { "2x3 min-variance decimal", () => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[2, 3] { { 1, 0, 0 }, { 0, 1, 0 } }, 0m, 1m), "Covariance matrix must be square." },
        { "3x2 min-variance decimal", () => ActiveSetQpSolver<decimal>.SolveMinVariance(new decimal[3, 2] { { 1, 0 }, { 0, 1 }, { 0, 0 } }, 0m, 1m), "Covariance matrix must be square." },
        { "1x2 mean-variance double", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[1, 2] { { 1, 0 } }, new double[] { 0.1 }, 1.0, 0.0, 1.0), "Covariance matrix must be square." },
        { "2x3 mean-variance double", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[2, 3] { { 1, 0, 0 }, { 0, 1, 0 } }, new double[] { 0.1, 0.2 }, 1.0, 0.0, 1.0), "Covariance matrix must be square." },
        { "3x2 mean-variance decimal", () => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[3, 2] { { 1, 0 }, { 0, 1 }, { 0, 0 } }, new decimal[] { 0.1m, 0.2m, 0.3m }, 1m, 0m, 1m), "Covariance matrix must be square." },
        { "2x3 mean-variance decimal", () => ActiveSetQpSolver<decimal>.SolveMeanVariance(new decimal[2, 3] { { 1, 0, 0 }, { 0, 1, 0 } }, new decimal[] { 0.1m, 0.2m }, 1m, 0m, 1m), "Covariance matrix must be square." },
        { "infinite [0, 1] min-variance", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, double.PositiveInfinity }, { double.PositiveInfinity, 1 } }, 0.0, 1.0), "Covariance entry at [0, 1] must be finite." },
        { "NaN [1, 2] min-variance", () => ActiveSetQpSolver<double>.SolveMinVariance(new double[,] { { 1, 0, 0 }, { 0, 1, double.NaN }, { 0, double.NaN, 1 } }, 0.0, 1.0), "Covariance entry at [1, 2] must be finite." },
        { "infinite [0, 1] mean-variance", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, double.PositiveInfinity }, { double.PositiveInfinity, 1 } }, new double[] { 0.1, 0.2 }, 1.0, 0.0, 1.0), "Covariance entry at [0, 1] must be finite." },
        { "infinite [0, 1] lambda 0", () => ActiveSetQpSolver<double>.SolveMeanVariance(new double[,] { { 1, double.PositiveInfinity }, { double.PositiveInfinity, 1 } }, new double[] { 0.1, 0.2 }, 0.0, 0.0, 1.0), "Covariance entry at [0, 1] must be finite." },
    };

    [Theory]
    [MemberData(nameof(EntryGuardMessages))]
    public void EntryGuard_RefusesTheInputItselfWithItsOwnMessage(string name, Action solve, string message)
    {
        var thrown = Record.Exception(solve);
        thrown.Should().NotBeNull(name);
        thrown!.GetType().Should().Be<ArgumentException>(name);
        ((ArgumentException)thrown).ParamName.Should().Be("covariance", name);
        thrown.Message.Should().Be(message + " (Parameter 'covariance')", name);
    }

    [Theory]
    [MemberData(nameof(Rejections))]
    public void Rejection_IsThrownWithTheTypeAndConditionThePageLists(string name, Action solve, Type expected, string pageKeyword, string? parameterName)
    {
        // The executed half: the solver throws exactly this type (no subclass or sibling).
        var thrown = Record.Exception(solve);
        thrown.Should().NotBeNull(name);
        thrown!.GetType().Should().Be(expected, name);

        // The guard that threw: at lambda = 0 the linear programme reaches no later check of the covariance or the means,
        // so the parameter the exception names identifies the validation that ran.
        if (parameterName is not null)
        {
            ((ArgumentException)thrown).ParamName.Should().Be(parameterName, name);
        }

        // The documented half: the page's exception list names that type and that condition.
        ExceptionsSection().Should().Contain(pageKeyword, name);
    }

    [Fact]
    public void ExceptionsList_NamesEveryTypeTheSolverThrows_AndNoSupersededRefusal()
    {
        var section = ExceptionsSection();

        foreach (var type in new[]
        {
            nameof(ArgumentNullException), nameof(ArgumentException), nameof(ArgumentOutOfRangeException),
            nameof(InvalidOperationException), nameof(NotSupportedException),
        })
        {
            section.Should().Contain("`" + type + "`");
        }

        // The rounded core hands an unusable working set to the exact route and never rolls a move back;
        // the only refusal on that path is an exactly singular working-set block in the exact route.
        section.Should().NotContain("roll back");
        section.Should().NotContain("rolled back");
        section.Should().Contain("exactly singular");

        // The lambda = 0 linear programme never inspects the covariance beyond validation.
        section.Should().Contain("`riskAversion > 0`");
    }

    [Fact]
    public void GuideAndSample_StateTheBoundRelaxation_AndQualifyTheWeightCap()
    {
        var page = QpSection();

        // The solver relaxes the offending bound to 1/N for a positive risk aversion and rejects it at lambda = 0.
        page.Should().Contain("1/N");
        page.Should().Contain("relax");

        // The sample's comment is true only when N * 0.4 >= 1; the page must say so.
        page.Should().NotContain("// weights.Sum() ≈ 1, each 0 ≤ w_i ≤ 0.4\n");
        page.Should().Contain("N ≥ 3");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(10)]
    public void DocumentedSample_HoldsForThreeOrMoreAssets(int n)
    {
        var covariance = new decimal[n, n];
        for (var i = 0; i < n; i++)
        {
            covariance[i, i] = 0.04m + 0.01m * i;
        }

        var weights = ActiveSetQpSolver.SolveMinVariance(covariance, minWeight: 0.0m, maxWeight: 0.4m);

        weights.Sum().Should().BeApproximately(1.0m, 1e-12m);
        weights.Should().OnlyContain(w => w >= 0.0m && w <= 0.4m);
    }

    [Fact]
    public void DocumentedSample_TwoAssets_RaisesTheUpperBoundToEqualWeight_AndLambdaZeroRejects()
    {
        var minVariance = ActiveSetQpSolver<double>.SolveMinVariance(s_identity2, 0.0, 0.4);
        minVariance[0].Should().BeApproximately(0.5, 1e-12);
        minVariance[1].Should().BeApproximately(0.5, 1e-12);

        var meanVariance = ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, 0.2 }, 3.0, 0.0, 0.4);
        meanVariance.Sum().Should().BeApproximately(1.0, 1e-12);
        meanVariance.Should().OnlyContain(w => w <= 0.5 + 1e-12);

        var lowerRelaxed = ActiveSetQpSolver<double>.SolveMinVariance(s_identity2, 0.6, 1.0);
        lowerRelaxed.Sum().Should().BeApproximately(1.0, 1e-12);
        lowerRelaxed.Should().OnlyContain(w => w >= 0.5 - 1e-12);

        var lambdaZero = () => ActiveSetQpSolver<double>.SolveMeanVariance(s_identity2, new double[] { 0.1, 0.2 }, 0.0, 0.0, 0.4);
        lambdaZero.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void LambdaZero_DoesNotRequireAPositiveDefiniteCovariance()
    {
        var covariance = new double[,] { { 1, 2 }, { 2, 1 } };

        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(covariance, new double[] { 0.1, 0.2 }, 0.0, 0.0, 0.7);

        weights.Sum().Should().BeApproximately(1.0, 1e-12);
        weights[1].Should().BeApproximately(0.7, 1e-12);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(10)]
    public void EqualWeightFrontier_IsSolvedWithoutRefusal(int n)
    {
        var covariance = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            covariance[i, i] = 1.0 + 0.1 * i;
        }

        var weights = ActiveSetQpSolver<double>.SolveMinVariance(covariance, 0.0, 1.0 / n);

        weights.Sum().Should().BeApproximately(1.0, 1e-12);
        weights.Should().OnlyContain(w => Math.Abs(w - 1.0 / n) <= 1e-12);
    }

    [Fact]
    public void Weights_AreInvariantToAPositiveRescalingOfTheCovariance()
    {
        var unit = new double[,] { { 4, 1, 0.5 }, { 1, 3, 0.2 }, { 0.5, 0.2, 2 } };
        var scaled = new double[3, 3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                scaled[i, j] = unit[i, j] * 1e-12;
            }
        }

        var reference = ActiveSetQpSolver<double>.SolveMinVariance(unit, 0.0, 0.5);
        var rescaled = ActiveSetQpSolver<double>.SolveMinVariance(scaled, 0.0, 0.5);

        for (var i = 0; i < 3; i++)
        {
            rescaled[i].Should().BeApproximately(reference[i], 1e-9);
        }
    }

    [Fact]
    public void Changelog_VersionTwoSection_DescribesOnlyTheShippedCore()
    {
        var unreleased = ChangelogSection("## [2.0.0]");

        // Mechanisms of the bound-fixing core that the primal-feasible active set replaced.
        unreleased.Should().NotContain("KKT-release tolerance");
        unreleased.Should().NotContain("max(|ν|, max|grad|)");
        unreleased.Should().NotContain("all-fixed exit");
        unreleased.Should().NotContain("all-fixed frontiers");
        unreleased.Should().NotContain("inverse-budget solves");

        // The rewrite entry remains and carries the behaviours that still hold.
        unreleased.Should().Contain("primal-feasible active set");
        unreleased.Should().Contain("equal-weight frontier");
        unreleased.Should().Contain("rescaling");
    }

    [Fact]
    public void DecimalSample_DoesNotPromiseNoDoubleCast_AndEveryPublicPageNamesTheBooleanOnlyPreChecks()
    {
        // At T = decimal the covariance's positive-definiteness proof and the conditioning guard's bound convert a
        // block to double; the sample comment must promise what is true (the returned weights), not "no cast".
        var solvers = QpSection();
        solvers.Should().NotContain("(no decimal → double cast)");
        solvers.Should().Contain("every returned weight is computed and certified in `decimal`");
        solvers.Should().Contain("Double-side pre-checks");

        var readme = QpReadmeBullets();
        readme.Should().Contain("pre-checks in `double`");
        readme.Should().Contain("every returned weight is computed in `T`");

        // The tier page lists the pattern as a permitted exception and the QP row agrees.
        var tiers = ReadText("docs", "tier-constraints.md");
        tiers.Should().Contain("**Exception 3");
        tiers.Should().NotContain("Two exceptions preserve");
        var row = tiers.Split('\n').Single(l => l.StartsWith("| `Solvers/ActiveSetQpSolver<T>`", StringComparison.Ordinal));
        row.Should().Contain("`double`").And.Contain("Exception 3");
    }

    [Fact]
    public void Changelog_VersionTwoEntry_StatesTheNewRefusalClass_AndNoLongerClaimsBehaviourChangesOnlyWhereTheOldResultWasWrong()
    {
        var unreleased = ChangelogSection("## [2.0.0]");

        unreleased.Should().NotContain("Behaviour changes only where");
        unreleased.Should().Contain("48·n·u·κ₁ ≥ 1");
        unreleased.Should().Contain("near-duplicate");
        unreleased.Should().Contain("8n + 8");
        unreleased.Should().Contain("12n + 12");
    }

    [Fact]
    public void Changelog_ReleaseSection2_0_0_DescribesTheShippedBudgetsAndRefusals_NotTheDeletedCore()
    {
        var release = ChangelogSection("## [2.0.0]");

        release.Should().NotContain("`2N`-iteration cap");
        release.Should().NotContain("nFree");
        release.Should().NotContain("fixedTotal");
        release.Should().Contain("`8N + 8`");
        release.Should().Contain("`12N + 12`");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(10)]
    [InlineData(150)]
    public void DocumentedBudgets_AreTheOnesTheSolverExecutes(int n)
    {
        ActiveSetQpSolver<double>.IterationBudget(n).Should().Be((8 * n) + 8);
        ActiveSetQpSolver<decimal>.IterationBudget(n).Should().Be((8 * n) + 8);
        ActiveSetQpSolver<double>.ExactRouteBudget(n).Should().Be((12 * n) + 12);
        ActiveSetQpSolver<decimal>.ExactRouteBudget(n).Should().Be((12 * n) + 12);
    }

    [Fact]
    public void DocumentedRefusalClass_NearDuplicateAssetsPastTheConditioningMargin_IsRefusedNamingStationarity_AndTheMarginIsTheBoundary()
    {
        // The CHANGELOG states the refusal class: a deciding block with 48·n·u·κ₁ ≥ 1. For the two-asset block
        // [[1, 1−ε], [1−ε, 1]], κ₁ ≈ 2/ε, so the margin 48·2·2⁻⁵³·κ₁ crosses 1 near ε ≈ 2e-14.
        var refused = new double[,] { { 1, 1 - 1e-14 }, { 1 - 1e-14, 1 } };
        var answered = new double[,] { { 1, 1 - 1e-13 }, { 1 - 1e-13, 1 } };

        var refusal = () => ActiveSetQpSolver<double>.SolveMinVariance(refused, 0.0, 1.0);
        refusal.Should().Throw<InvalidOperationException>().WithMessage("*stationarity cannot be certified*48·n·u·κ ≥ 1*");

        var weights = ActiveSetQpSolver<double>.SolveMinVariance(answered, 0.0, 1.0);
        weights[0].Should().BeApproximately(0.5, 1e-12);
        weights[1].Should().BeApproximately(0.5, 1e-12);
    }

    private static string QpReadmeBullets()
    {
        var text = ReadText("README.md");
        var start = text.IndexOf("#### Portfolio Optimization (Quadratic Programming)", StringComparison.Ordinal);
        start.Should().BeGreaterThanOrEqualTo(0);
        var end = text.IndexOf("\n### ", start, StringComparison.Ordinal);
        end.Should().BeGreaterThan(start);
        return text[start..end];
    }

    private static string ChangelogSection(string heading)
    {
        var text = ReadText("CHANGELOG.md");
        var start = text.IndexOf(heading, StringComparison.Ordinal);
        start.Should().BeGreaterThanOrEqualTo(0);
        var end = text.IndexOf("\n## [", start + 1, StringComparison.Ordinal);
        end.Should().BeGreaterThan(start);
        return text[start..end];
    }

    private static string ReadText(params string[] path)
        => File.ReadAllText(Path.Combine([RepositoryRoot(), .. path])).Replace("\r\n", "\n", StringComparison.Ordinal);

    private static string ExceptionsSection()
    {
        var page = QpSection();
        var start = page.IndexOf("**Exceptions:**", StringComparison.Ordinal);
        start.Should().BeGreaterThanOrEqualTo(0, "the QP section must carry an Exceptions list");
        return page[start..];
    }

    private static string QpSection()
    {
        var text = File.ReadAllText(Path.Combine(RepositoryRoot(), "docs", "solvers.md")).Replace("\r\n", "\n", StringComparison.Ordinal);
        var start = text.IndexOf("## Portfolio-optimization QP", StringComparison.Ordinal);
        start.Should().BeGreaterThanOrEqualTo(0);
        var end = text.IndexOf("\n### Validation", start, StringComparison.Ordinal);
        end.Should().BeGreaterThan(start);
        return text[start..end];
    }

    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Boutquin.Numerics.slnx")))
            {
                return dir.FullName;
            }
        }

        throw new FileNotFoundException("Repository root (Boutquin.Numerics.slnx) not found above " + AppContext.BaseDirectory);
    }
}
