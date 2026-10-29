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
/// Range behaviour of the Nelder–Mead simplex minimiser and the active-set QP solver:
/// a trial point or KKT quantity that is representable must not be lost to an
/// overflowing intermediate, and a genuinely unrepresentable one must surface as the
/// documented exception type rather than a raw <see cref="OverflowException"/>.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class Arena414074R1OptimizerTests
{
    private const double U = 1e308;

    // ── Nelder–Mead: representable trial points ────────────────────────────

    [Fact]
    public void NelderMead_ExpansionWhoseProductOverflows_IsEvaluatedAtTheRepresentableExactPoint()
    {
        // Exact expansion point: c + γ(r − c) = −0.5U + 2·U = 1.5U (representable);
        // the product γ·(r − c) = 2U is not.
        var seen = new List<double>();
        SimplexResult<double>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<double>.Minimize(
            x => { seen.Add(x[0]); var t = (x[0] / U) - 1.5; return t * t; },
            [-1.5 * U],
            new SimplexOptions<double>(InitialStep: U, XTolerance: 0, FTolerance: 4.5)));

        error.Should().BeNull("the exact expansion point 1.5e308 is representable");
        seen.Should().Contain(x => Math.Abs(x - (1.5 * U)) <= 1e293);
        result!.Converged.Should().BeTrue();
        result.Argmin[0].Should().BeApproximately(1.5 * U, 1e293);
        result.MinValue.Should().BeLessThan(1e-12);
    }

    [Fact]
    public void NelderMead_ReflectionWhoseProductOverflows_IsEvaluatedAtTheRepresentableExactPoint()
    {
        // best c ≈ −0.9U, worst −1.6U, α = 3: α·(c − worst) = 2.1U overflows while the exact
        // reflected point c + α(c − worst) = 1.2U is representable. It is rejected (worse than
        // the worst vertex), so the search continues with an inside contraction.
        static double Objective(double[] x)
        {
            var u = x[0] / U;
            return u >= -0.95 ? (u > 1 ? 3 : 0.4 * (u + 0.9)) : (-u - 0.9) + 0.5;
        }

        var seen = new List<double>();
        SimplexResult<double>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<double>.Minimize(
            x => { seen.Add(x[0]); return Objective(x); },
            [-1.6 * U],
            new SimplexOptions<double>(
                InitialStep: 0.7 * U, XTolerance: 0, FTolerance: 1.0, Reflection: 3, Expansion: 4)));

        error.Should().BeNull("the exact reflection point 1.2e308 is representable");
        seen.Should().Contain(x => Math.Abs(x - (1.2 * U)) <= 1e293);
        result!.Converged.Should().BeTrue();
    }

    [Fact]
    public void NelderMead_OutsideContractionWhoseDifferenceOverflows_IsEvaluatedAtTheRepresentableExactPoint()
    {
        // best c ≈ −0.9U, worst −1.6U, α = 3 ⇒ reflected 1.2U; (r − c) = 2.1U overflows
        // while c + ρ(r − c) = 0.15U (ρ = ½) is representable.
        static double Objective(double[] x)
        {
            var u = x[0] / U;
            return u >= -0.95 ? 0.4 * (u + 0.9) : (-u - 0.9) + 0.5;
        }

        var seen = new List<double>();
        SimplexResult<double>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<double>.Minimize(
            x => { seen.Add(x[0]); return Objective(x); },
            [-1.6 * U],
            new SimplexOptions<double>(
                InitialStep: 0.7 * U, XTolerance: 0, FTolerance: 0.5, Reflection: 3, Expansion: 4)));

        error.Should().BeNull("the exact outside-contraction point 0.15e308 is representable");
        seen.Should().Contain(x => Math.Abs(x - (0.15 * U)) <= 1e293);
        result!.Converged.Should().BeTrue();
    }

    // Two-iteration trajectory shared by the inside-contraction and shrink tests. Vertices
    // start at −1.5U and −0.5U; the first iteration expands (γ = 20) to 1.5U, leaving a
    // simplex {1.5U, −0.5U} whose 2U width is itself beyond MaxValue. The second iteration
    // therefore starts from a representable simplex whose vertex difference overflows.
    private static double EvolvedObjective(double[] x)
    {
        var u = x[0] / U;
        if (Math.Abs(u + 1.5) < 1e-6)
        {
            return 9;
        }

        if (Math.Abs(u + 0.5) < 1e-6)
        {
            return 4;
        }

        if (Math.Abs(u + 0.4) < 1e-6)
        {
            return 3.61;
        }

        if (Math.Abs(u - 1.5) < 1e-6)
        {
            return 0;
        }

        return Math.Abs(u - 0.7) < 1e-6 ? 0.64 : 10;
    }

    [Fact]
    public void NelderMead_InsideContractionWhoseDifferenceOverflows_IsEvaluatedAtTheRepresentableExactPoint()
    {
        // Second iteration: best c = 1.5U, worst −0.5U, reflected 1.7U is rejected (f = 10).
        // (worst − c) = −2U overflows while c + 0.4·(worst − c) = 0.7U is representable.
        var seen = new List<double>();
        SimplexResult<double>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<double>.Minimize(
            x => { seen.Add(x[0]); return EvolvedObjective(x); },
            [-1.5 * U],
            new SimplexOptions<double>(
                InitialStep: U, XTolerance: 0, FTolerance: 0.7, Reflection: 0.1, Expansion: 20, Contraction: 0.4)));

        error.Should().BeNull("the exact inside-contraction point 0.7e308 is representable");
        seen.Should().Contain(x => Math.Abs(x - (0.7 * U)) <= 1e293);
        result!.Converged.Should().BeTrue();
        result.Argmin[0].Should().BeApproximately(1.5 * U, 1e293);
    }

    [Fact]
    public void NelderMead_ShrinkWhoseDifferenceOverflows_IsEvaluatedAtTheRepresentableExactPoint()
    {
        // Same trajectory, but the inside contraction (0.6U, f = 10) is also rejected, so the
        // simplex shrinks: c + 0.4·(worst − c) = 0.7U with (worst − c) = −2U overflowing.
        var seen = new List<double>();
        SimplexResult<double>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<double>.Minimize(
            x => { seen.Add(x[0]); return EvolvedObjective(x); },
            [-1.5 * U],
            new SimplexOptions<double>(
                InitialStep: U, XTolerance: 0.9 * U, FTolerance: 0, Reflection: 0.1, Expansion: 20,
                Contraction: 0.45, Shrink: 0.4)));

        error.Should().BeNull("the exact shrunk vertex 0.7e308 is representable");
        seen.Should().Contain(x => Math.Abs(x - (0.6 * U)) <= 1e293, "the inside contraction was tried and rejected");
        seen.Count(x => Math.Abs(x - (0.7 * U)) <= 1e293).Should().Be(1);
        result!.Converged.Should().BeTrue();
        result.Argmin[0].Should().BeApproximately(1.5 * U, 1e293);
    }

    [Fact]
    public void NelderMead_UnrepresentableExpansion_StillFailsBeforeTheObjectiveSeesIt()
    {
        // Negative pairing: best 1.0U, reflected 1.5U, expansion 2.0U > MaxValue. No
        // fallback may invent a finite coordinate; the pinned ArithmeticException stays.
        var sawNonFinite = false;
        var error = Record.Exception(() => NelderMeadMinimizer<double>.Minimize(
            x =>
            {
                sawNonFinite |= !double.IsFinite(x[0]);
                return -x[0] / U;
            },
            [0.5 * U],
            new SimplexOptions<double>(InitialStep: 0.5 * U)));

        error.Should().BeAssignableTo<ArithmeticException>();
        sawNonFinite.Should().BeFalse();
    }

    // ── Nelder–Mead: decimal analogue ──────────────────────────────────────

    [Fact]
    public void NelderMead_DecimalExpansionWhoseProductOverflows_ConvergesOnTheRepresentableExactPoint()
    {
        // Exact expansion point 6e28 is representable; γ·(r − c) = 8e28 exceeds decimal.MaxValue.
        var seen = new List<decimal>();
        SimplexResult<decimal>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<decimal>.Minimize(
            x => { seen.Add(x[0]); var t = (x[0] / 1e28m) - 6m; return t * t; },
            [-6e28m],
            new SimplexOptions<decimal>(InitialStep: 4e28m, XTolerance: 0m, FTolerance: 70m)));

        error.Should().BeNull("the exact expansion point 6e28 is representable");
        seen.Should().Contain(6e28m);
        result!.Converged.Should().BeTrue();
        result.Argmin[0].Should().Be(6e28m);
        result.MinValue.Should().Be(0m);
    }

    [Fact]
    public void NelderMead_DecimalSimplexDiameterBeyondRange_IsTreatedAsLargerThanAnyTolerance()
    {
        // best c = −3.5e28, worst −5.5e28, α = 5.5 ⇒ reflected 7.5e28, outside contraction
        // (ρ = 0.9) accepted at 6.4e28. Both vertices are representable; their distance
        // 9.9e28 is not. IEEE types see an infinite diameter there (never within a
        // tolerance), so decimal must not throw either.
        static decimal Objective(decimal[] x)
        {
            var u = x[0] / 1e28m;
            return u >= -3.6m ? 0.1m * (u + 3.5m) : (-u - 3.5m) + 0.5m;
        }

        var seen = new List<decimal>();
        SimplexResult<decimal>? result = null;
        var error = Record.Exception(() => result = NelderMeadMinimizer<decimal>.Minimize(
            x => { seen.Add(x[0]); return Objective(x); },
            [-5.5e28m],
            new SimplexOptions<decimal>(
                InitialStep: 2e28m, XTolerance: 1e28m, FTolerance: 1.0m,
                Reflection: 5.5m, Expansion: 6m, Contraction: 0.9m)));

        error.Should().BeNull("an unrepresentable simplex diameter exceeds every tolerance");
        seen.Should().Contain(6.4e28m);
        result!.Converged.Should().BeTrue("the value spread 0.99 satisfies FTolerance 1.0");
        result.Argmin[0].Should().Be(-3.5e28m);
    }

    // ── Active-set QP: fixed-variable cross terms ──────────────────────────

    private static double[,] S3(double scale) => new[,]
    {
        { scale, 0, 0.6 * scale },
        { 0, scale, -0.6 * scale },
        { 0.6 * scale, -0.6 * scale, scale },
    };

    private static decimal[,] S3D(decimal scale) => new[,]
    {
        { scale, 0m, 0.6m * scale },
        { 0m, scale, -0.6m * scale },
        { 0.6m * scale, -0.6m * scale, scale },
    };

    private static double[,] L2(double scale) => new[,]
    {
        { scale, 1.049895 * scale },
        { 1.049895 * scale, 1.1025 * scale },
    };

    [Fact]
    public void QpMeanVariance_RiskAversionTimesCovarianceOverflowingAtAZeroWeight_ReturnsTheRepresentableSolution()
    {
        // λ·Σ₀₂ = 3·0.6e308 overflows; times the zero lower bound it formed ∞·0 = NaN.
        // The exact cross term is 0 and the (λ-independent, zero-mean) answer is [0, ½, ½].
        double[]? weights = null;
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            S3(1e308), new[] { 0.0, 0, 0 }, 3, 0, 1));

        error.Should().BeNull("every KKT quantity of this problem is representable");
        weights.Should().NotBeNull();
        weights![0].Should().BeApproximately(0, 1e-12);
        weights[1].Should().BeApproximately(0.5, 1e-12);
        weights[2].Should().BeApproximately(0.5, 1e-12);
    }

    [Fact]
    public void QpMeanVariance_DecimalRiskAversionTimesCovarianceOverflowingAtAZeroWeight_ReturnsTheRepresentableSolution()
    {
        decimal[]? weights = null;
        var error = Record.Exception(() => weights = ActiveSetQpSolver<decimal>.SolveMeanVariance(
            S3D(1e28m), new[] { 0m, 0m, 0m }, 14m, 0m, 1m));

        error.Should().BeNull("every KKT quantity of this problem is representable");
        weights.Should().NotBeNull();
        weights![0].Should().BeApproximately(0m, 1e-12m);
        weights[1].Should().BeApproximately(0.5m, 1e-12m);
        weights[2].Should().BeApproximately(0.5m, 1e-12m);
    }

    [Fact]
    public void QpMeanVariance_RiskAversionTimesCovarianceOverflowingAtANonzeroWeight_ReturnsTheRepresentableSolution()
    {
        // Same problem with the weights pinned at nonzero bounds: λΣ = 1.8e308 overflows
        // but λ·(Σ·w) stays below MaxValue, so the unit-scale solution must come back.
        var expected = ActiveSetQpSolver<double>.SolveMeanVariance(S3(1), new[] { 0.0, 0, 0 }, 3, 0.1, 0.4);
        double[]? weights = null;
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            S3(1e308), new[] { 0.0, 0, 0 }, 3, 0.1, 0.4));

        error.Should().BeNull("every KKT quantity of this problem is representable");
        weights.Should().NotBeNull();
        for (var i = 0; i < 3; i++)
        {
            weights![i].Should().BeApproximately(expected[i], 1e-9);
        }
    }

    [Fact]
    public void QpMinVariance_UnrepresentableCrossTerm_IsReportedAsTheDocumentedOverflowException()
    {
        // Σ·bound = 1.57e309 cannot be represented, so the solve is rejected — with the
        // documented InvalidOperationException, not a Cholesky argument error naming a
        // parameter the caller never passed.
        var act = () => ActiveSetQpSolver<double>.SolveMinVariance(L2(1.5e308), -10, 10);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void QpMeanVariance_UnrepresentableCrossTerm_IsReportedAsTheDocumentedOverflowException()
    {
        // λ·Σ·bound = 1.57e309 cannot be represented, so the solve is rejected with the
        // documented InvalidOperationException rather than a Cholesky argument error.
        var act = () => ActiveSetQpSolver<double>.SolveMeanVariance(
            L2(1.5e308), new[] { 0.0, 0.0 }, 1, -10, 10);

        act.Should().Throw<InvalidOperationException>();
    }

    // ── Active-set QP: decimal overflow contract ───────────────────────────

    [Fact]
    public void QpMinVariance_DecimalReleaseGradientOverflow_IsReportedAsTheDocumentedOverflowException()
    {
        var act = () => ActiveSetQpSolver<decimal>.SolveMinVariance(
            new[,] { { 5e27m, 5e27m * 1.049895m }, { 5e27m * 1.049895m, 5e27m * 1.1025m } }, -20m, 20m);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void QpMeanVariance_DecimalReleaseGradientOverflow_IsReportedAsTheDocumentedOverflowException()
    {
        var act = () => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            new[,] { { 7e28m, 0m }, { 0m, 7e28m } }, new[] { 0m, 0m }, 4m, 0m, 1m);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void QpMeanVariance_DecimalLpBudgetOverflow_IsReportedAsTheDocumentedOverflowException()
    {
        var act = () => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            new[,] { { 1m, 0m }, { 0m, 1m } }, new[] { 1m, 0m }, 0m, -7e28m, 7e28m);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void QpDecimalFacade_ReleaseGradientOverflow_IsReportedAsTheDocumentedOverflowException()
    {
        var act = () => ActiveSetQpSolver.SolveMeanVariance(
            new[,] { { 7e28m, 0m }, { 0m, 7e28m } }, new[] { 0m, 0m }, 4m, 0m, 1m);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void QpMinVariance_DecimalProblemInsideTheRange_StillSolves()
    {
        // Positive pairing for the overflow translation: representable work is untouched.
        var weights = ActiveSetQpSolver<decimal>.SolveMinVariance(
            new[,] { { 1m, 1.049895m }, { 1.049895m, 1.1025m } }, -20m, 20m);

        weights[0].Should().BeApproximately(19.4114391143911439m, 1e-12m);
        weights[1].Should().BeApproximately(-18.4114391143911439m, 1e-12m);
    }
}
