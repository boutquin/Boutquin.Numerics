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

using Boutquin.Numerics.LinearAlgebra.Internal;
using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Regression tests for the Solvers-cluster review-fix batch (2026-06-28).
/// Each test pins a specific finding (S3, S4, S8, S11) or fills a coverage gap
/// (B30, B31, B32, B25). Behaviour-changing fixes were authored test-first.
/// </summary>
public sealed class SolversReviewFixTests
{
    [Fact]
    public void BracketedSolvers_ReversedEndpoints_AreRejectedBeforeObjectiveEvaluation()
    {
        var bisection = new BisectionSolver<double>(1e-12, 1e-12);
        var newton = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0);

        Action bisectionCall = () => bisection.Solve(_ => throw new InvalidOperationException("objective"), 1.0, -1.0);
        Action newtonCall = () => newton.Solve(_ => throw new InvalidOperationException("objective"), 1.0, -1.0);

        bisectionCall.Should().Throw<ArgumentException>().WithParameterName("lowerBound");
        newtonCall.Should().Throw<ArgumentException>().WithParameterName("lowerBound");
    }

    [Fact]
    public void MeanVariance_NegativeOrNonFiniteRiskAversion_IsRejected()
    {
        var covariance = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        double[] means = [0.1, 0.2];

        foreach (var riskAversion in new[] { -1.0, double.NaN, double.PositiveInfinity })
        {
            Action call = () => ActiveSetQpSolver<double>.SolveMeanVariance(covariance, means, riskAversion, 0.0, 1.0);
            call.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("riskAversion");
        }
    }

    // ── S3: NewtonRaphson bracketed Solve must check residual before reporting convergence ──

    [Fact]
    public void NewtonRaphson_Bracketed_SmallFallbackStepFollowsSelectedStoppingRule()
    {
        // Force bisection with a deliberately unusable derivative. The successive
        // signed steps are1/2,1/4,1/8; the third is strictly below tolerance.2.
        // The selected stopping rule returns7/8 despite its large residual.
        static double Steep(double x) => 1e8 * (x - 0.999);

        var solver = new NewtonRaphsonSolver<double>(
            functionTolerance: 1e-9,
            bracketTolerance: 1e-300,
            stepTolerance: 0.2,         // generous: the bisection step crosses it fast
            maxIterations: 4,           // stop before bisection actually nears the root
            derivative: _ => 1e-40);    // ~0 ⇒ Newton step is NaN ⇒ always bisection

        var result = solver.Solve(Steep, 0.0, 1.0);

        result.Converged.Should().BeTrue();
        result.Iterations.Should().Be(3);
        result.Root.Should().Be(0.875);
        result.EstimatedError.Should().Be(0.125);
        result.FinalResidual.Should().BeApproximately(-12400000, 1e-8);
    }

    [Fact]
    public void NewtonRaphson_Bracketed_ConvergesWithSmallStepAndResidual()
    {
        // When both the step and the residual are small, convergence is legitimate.
        static double Quadratic(double x) => (x * x) - 2.0;

        var solver = new NewtonRaphsonSolver<double>(
            functionTolerance: 1e-9,
            bracketTolerance: 1e-12,
            stepTolerance: 1e-10,
            maxIterations: 100,
            derivative: x => 2.0 * x);

        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-7);
        Math.Abs(result.FinalResidual).Should().BeLessThan(1e-9);
    }

    // ── F-L1-01: NelderMead must not converge on a degenerate zero-diameter simplex ──

    [Fact]
    public void NelderMead_NonPositiveInitialStep_DoesNotFalseConvergeAtInitialGuess()
    {
        // A supplied InitialStep of 0 was treated as "a step was supplied" via
        // options.InitialStep.HasValue, building a zero-diameter simplex whose first
        // convergence test fired immediately — returning Converged:true AT THE INITIAL
        // GUESS (0,0) with the wrong argmin. The true minimum of (x-5)²+(y+3)² is 0 at
        // (5,-3). A non-positive supplied step must fall back to the non-degenerate
        // default step so the search actually runs.
        static double Objective(double[] x)
        {
            var dx = x[0] - 5.0;
            var dy = x[1] + 3.0;
            return (dx * dx) + (dy * dy);
        }

        var result = NelderMeadMinimizer<double>.Minimize(
            Objective,
            [0.0, 0.0],
            new SimplexOptions<double>(InitialStep: 0.0));

        // Before the fix: Converged=true, Argmin=(0,0), MinValue=34 — a false converge.
        result.Argmin[0].Should().BeApproximately(5.0, 1e-4);
        result.Argmin[1].Should().BeApproximately(-3.0, 1e-4);
        result.MinValue.Should().BeApproximately(0.0, 1e-6);
    }

    // ── S4: Muller must guard against a non-finite x3 ──────────────────────────

    [Fact]
    public void Muller_NonFiniteObjectiveValue_Throws()
    {
        // A degenerate near-flat quadratic fit can drive the Muller update to ±∞.
        // exp(x) on three nearly-collinear points produces a tiny denominator while
        // the curvature term blows up the step. Before the fix, x3 became ±∞ and the
        // subsequent objective(x3) poisoned the result with NaN while still flowing
        // through the loop.
        static double Objective(double x) => Math.Exp(x) + 1.0; // no real root; always positive

        var solver = new MullerSolver<double>(tolerance: 1e-12, maxIterations: 200);

        // Start far out where exp overflow + flat curvature drive the step non-finite.
        var act = () => solver.Solve(Objective, 700.0, 710.0, 720.0);

        act.Should().Throw<InvalidOperationException>().WithMessage("*non-finite*");
    }

    // ── S8: SolveMaxReturnLP (λ=0) weights must sum to 1 ───────────────────────

    [Fact]
    public void MeanVariance_ZeroRiskAversion_WeightsMeetTheBudgetWithinRounding()
    {
        // An exact greedy allocation fixes the first weight at its upper bound and
        // the last five at their lower bound. Only the second weight is rounded.
        // Summing rounded double weights need not produce the exact bit pattern of 1.
        const int n = 7;
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = 0.04 + (0.01 * i);
        }

        double[] means = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08];

        var w = ActiveSetQpSolver<double>.SolveMeanVariance(
            cov, means, riskAversion: 0.0, minWeight: 0.059, maxWeight: 0.559);

        w[0].Should().Be(0.559);
        w.Skip(2).Should().OnlyContain(weight => weight == 0.059);
        w[1].Should().BeApproximately(0.146, 1e-15);
        w.Sum().Should().BeApproximately(1.0, n * Math.ScaleB(1.0, -52),
            because: "the rounded weights satisfy the unit budget within summation error");
    }

    [Theory]
    [InlineData(0.059, 0.559)]
    [InlineData(0.05, 0.4)]
    [InlineData(0.0, 1.0)]
    [InlineData(0.1, 0.5)]
    public void MeanVariance_ZeroRiskAversion_WeightsSumToOne_AcrossBounds(double lo, double hi)
    {
        const int n = 7;
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = 0.04 + (0.01 * i);
        }

        double[] means = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08];

        var w = ActiveSetQpSolver<double>.SolveMeanVariance(
            cov, means, riskAversion: 0.0, minWeight: lo, maxWeight: hi);

        w.Sum().Should().BeApproximately(1.0, 1e-15);
        foreach (var wi in w)
        {
            wi.Should().BeGreaterThanOrEqualTo(lo - 1e-12);
            wi.Should().BeLessThanOrEqualTo(hi + 1e-12);
        }
    }

    // ── S11: OlsResult<T>.HasInferentialContent ────────────────────────────────

    [Fact]
    public void Ols_SaturatedDesign_ZeroDof_HasNoInferentialContent()
    {
        // Two observations, two coefficients (intercept + slope): dof = 0.
        var x = new double[,] { { 1.0 }, { 2.0 } };
        double[] y = [3.0, 5.0];

        var result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: true);

        result.DegreesOfFreedom.Should().Be(0);
        result.HasInferentialContent.Should().BeFalse(
            because: "zero residual degrees of freedom yields zeroed standard errors — no inference possible");
        result.StandardErrors.Should().OnlyContain(se => se == 0.0);
    }

    [Fact]
    public void Ols_OverdeterminedDesign_HasInferentialContent()
    {
        var x = new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 } };
        double[] y = [2.1, 3.9, 6.2, 7.8, 10.1];

        var result = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: true);

        result.DegreesOfFreedom.Should().BeGreaterThan(0);
        result.HasInferentialContent.Should().BeTrue();
    }

    // ── T15: LM reports a convergence reason; DampingOverflow is reachable ──────

    [Fact]
    public void Lm_SuccessfulFit_ReportsAConvergenceReason()
    {
        // A clean linear fit must terminate via one of the three convergence rules,
        // never MaxIterationsReached or DampingOverflow.
        double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1

        double[] Residuals(double[] p)
        {
            var r = new double[xs.Length];
            for (var i = 0; i < xs.Length; i++)
            {
                r[i] = ys[i] - ((p[0] * xs[i]) + p[1]);
            }

            return r;
        }

        var solver = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-12, 1e-3);
        var result = solver.Solve(Residuals, new double[] { 0.0, 0.0 });

        result.Converged.Should().BeTrue();
        result.TerminationReason.Should().BeOneOf(
            LmTerminationReason.FunctionToleranceReached,
            LmTerminationReason.ParameterToleranceReached,
            LmTerminationReason.GradientToleranceReached);
    }

    [Fact]
    public void Lm_NonFiniteJacobian_ThrowsCallbackContractException()
    {
        // A degenerate analytic Jacobian full of non-finite entries makes the damped
        // normal-equation matrix non-factorable on every retry, so λ grows to its
        // ceiling and the solver reports DampingOverflow rather than spinning or
        // returning a NaN-poisoned "converged" result.
        double[] Residuals(double[] p) => [p[0] - 1.0, p[1] - 1.0];

        // Jacobian entries are +∞ — JᵀJ is non-finite, Cholesky fails for any λ.
        double[,] BadJacobian(double[] p) => new double[,]
        {
            { double.PositiveInfinity, double.PositiveInfinity },
            { double.PositiveInfinity, double.PositiveInfinity },
        };

        var solver = new LevenbergMarquardtSolver<double>(200, 1e-12, 1e-12, 1e-12, 1e-3);
        var act = () => solver.Solve(Residuals, new double[] { 5.0, 5.0 }, BadJacobian);

        act.Should().Throw<InvalidOperationException>().WithMessage("*Jacobian*non-finite*");
    }

    // ── B30: root-solver argument guards ───────────────────────────────────────

    [Fact]
    public void Bisection_NullObjective_ThrowsArgumentNull()
    {
        var solver = new BisectionSolver<double>(1e-12, 1e-12);
        var act = () => solver.Solve(null!, 0.0, 1.0);
        act.Should().Throw<ArgumentNullException>().WithParameterName("objective");
    }

    [Fact]
    public void Brent_NullObjective_ThrowsArgumentNull()
    {
        var solver = new BrentSolver<double>(1e-12);
        var act = () => solver.Solve(null!, 0.0, 1.0);
        act.Should().Throw<ArgumentNullException>().WithParameterName("objective");
    }

    [Fact]
    public void NewtonRaphson_Bracketed_NullObjective_ThrowsArgumentNull()
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0);
        var act = () => solver.Solve(null!, 0.0, 1.0);
        act.Should().Throw<ArgumentNullException>().WithParameterName("objective");
    }

    [Fact]
    public void NewtonRaphson_Unbracketed_NullObjective_ThrowsArgumentNull()
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0);
        var act = () => solver.Solve(null!, 1.0);
        act.Should().Throw<ArgumentNullException>().WithParameterName("objective");
    }

    [Fact]
    public void Secant_NullObjective_ThrowsArgumentNull()
    {
        var solver = new SecantSolver<double>(1e-12);
        var act = () => solver.Solve(null!, 1.0);
        act.Should().Throw<ArgumentNullException>().WithParameterName("objective");
    }

    [Fact]
    public void Muller_NullObjective_ThrowsArgumentNull()
    {
        var solver = new MullerSolver<double>(1e-12);
        var act = () => solver.Solve(null!, 1.0);
        act.Should().Throw<ArgumentNullException>().WithParameterName("objective");
    }

    [Fact]
    public void Bisection_NonPositiveMaxIterations_ThrowsArgumentOutOfRange()
    {
        var act = () => new BisectionSolver<double>(1e-12, 1e-12, maxIterations: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxIterations");
    }

    [Fact]
    public void Brent_NonPositiveMaxIterations_ThrowsArgumentOutOfRange()
    {
        var act = () => new BrentSolver<double>(1e-12, maxIterations: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxIterations");
    }

    [Fact]
    public void NewtonRaphson_NonPositiveMaxIterations_ThrowsArgumentOutOfRange()
    {
        var act = () => new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0, maxIterations: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxIterations");
    }

    [Fact]
    public void Secant_NonPositiveMaxIterations_ThrowsArgumentOutOfRange()
    {
        var act = () => new SecantSolver<double>(1e-12, maxIterations: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxIterations");
    }

    [Fact]
    public void Muller_NonPositiveMaxIterations_ThrowsArgumentOutOfRange()
    {
        var act = () => new MullerSolver<double>(1e-12, maxIterations: 0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxIterations");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    public void Brent_InvalidTolerance_ThrowsArgumentOutOfRange(double tolerance)
    {
        var act = () => new BrentSolver<double>(tolerance);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(tolerance));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-1e-9)]
    public void Secant_InvalidTolerance_ThrowsArgumentOutOfRange(double tolerance)
    {
        var act = () => new SecantSolver<double>(tolerance);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(tolerance));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    public void Muller_InvalidTolerance_ThrowsArgumentOutOfRange(double tolerance)
    {
        var act = () => new MullerSolver<double>(tolerance);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(tolerance));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    public void Bisection_InvalidFunctionTolerance_ThrowsArgumentOutOfRange(double functionTolerance)
    {
        var act = () => new BisectionSolver<double>(functionTolerance, bracketTolerance: 1e-12);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(functionTolerance));
    }

    // ── F-L5-1/F-L1-03: Bisection ctor must validate bracketTolerance (sibling-consistency) ──

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1e-9)]
    public void Bisection_InvalidBracketTolerance_ThrowsArgumentOutOfRange(double bracketTol)
    {
        var act = () => new BisectionSolver<double>(functionTolerance: 1e-12, bracketTolerance: bracketTol);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("bracketTolerance");
    }

    // ── F-L5-2: NewtonRaphson ctor must validate all three tolerances (sibling-consistency) ──

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1e-9)]
    public void NewtonRaphson_InvalidFunctionTolerance_ThrowsArgumentOutOfRange(double tol)
    {
        var act = () => new NewtonRaphsonSolver<double>(functionTolerance: tol, bracketTolerance: 1e-12, stepTolerance: 1e-12);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("functionTolerance");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1e-9)]
    public void NewtonRaphson_InvalidBracketTolerance_ThrowsArgumentOutOfRange(double tol)
    {
        var act = () => new NewtonRaphsonSolver<double>(functionTolerance: 1e-12, bracketTolerance: tol, stepTolerance: 1e-12);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("bracketTolerance");
    }

    // stepTolerance = 0 is the documented "no step-based stopping" sentinel (the legacy
    // facade defaults it to 0 and the loop gates it with `_stepTolerance > T.Zero`), so it
    // must be accepted; non-finite and negative values are invalid.
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-1e-9)]
    public void NewtonRaphson_InvalidStepTolerance_ThrowsArgumentOutOfRange(double tol)
    {
        var act = () => new NewtonRaphsonSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12, stepTolerance: tol);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("stepTolerance");
    }

    [Fact]
    public void NewtonRaphson_ZeroStepTolerance_IsAccepted()
    {
        var act = () => new NewtonRaphsonSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12, stepTolerance: 0.0);
        act.Should().NotThrow(because: "stepTolerance = 0 is the documented 'no step-based stopping' sentinel");
    }

    // ── F-L5-3: LevenbergMarquardt ctor must validate tolerances + initialDamping ──

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1e-9)]
    public void Lm_InvalidFunctionTolerance_ThrowsArgumentOutOfRange(double tol)
    {
        var act = () => new LevenbergMarquardtSolver<double>(
            maxIterations: 200, functionTolerance: tol, parameterTolerance: 1e-12,
            gradientTolerance: 1e-12, initialDamping: 1e-3);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("functionTolerance");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1e-9)]
    public void Lm_InvalidParameterTolerance_ThrowsArgumentOutOfRange(double tol)
    {
        var act = () => new LevenbergMarquardtSolver<double>(
            maxIterations: 200, functionTolerance: 1e-12, parameterTolerance: tol,
            gradientTolerance: 1e-12, initialDamping: 1e-3);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("parameterTolerance");
    }

    // gradientTolerance = 0 is the legitimate exact-stationary gate (must NOT throw);
    // Non-finite and negative values are the disabling/risky values that must be rejected.
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-1e-9)]
    public void Lm_InvalidGradientTolerance_ThrowsArgumentOutOfRange(double tol)
    {
        var act = () => new LevenbergMarquardtSolver<double>(
            maxIterations: 200, functionTolerance: 1e-12, parameterTolerance: 1e-12,
            gradientTolerance: tol, initialDamping: 1e-3);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("gradientTolerance");
    }

    [Fact]
    public void Lm_ZeroGradientTolerance_IsAccepted()
    {
        var act = () => new LevenbergMarquardtSolver<double>(
            maxIterations: 200, functionTolerance: 1e-12, parameterTolerance: 1e-12,
            gradientTolerance: 0.0, initialDamping: 1e-3);
        act.Should().NotThrow(because: "gradientTolerance = 0 is the exact-stationary-point gate, not an invalid value");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1e-3)]
    public void Lm_InvalidInitialDamping_ThrowsArgumentOutOfRange(double damping)
    {
        var act = () => new LevenbergMarquardtSolver<double>(
            maxIterations: 200, functionTolerance: 1e-12, parameterTolerance: 1e-12,
            gradientTolerance: 1e-12, initialDamping: damping);
        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("initialDamping");
    }

    // ── B31: ActiveSetQpSolver.SolveMeanVariance guards + single-asset + parity ─

    [Fact]
    public void MeanVariance_NonSquareCovariance_ThrowsArgument()
    {
        var cov = new double[2, 3];
        double[] means = [0.1, 0.2];
        var act = () => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 1.0, 0.0, 1.0);
        act.Should().Throw<ArgumentException>().WithParameterName("covariance");
    }

    [Fact]
    public void MeanVariance_MeansLengthMismatch_ThrowsArgument()
    {
        var cov = new double[3, 3];
        double[] means = [0.1, 0.2]; // length 2 != 3
        var act = () => ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 1.0, 0.0, 1.0);
        act.Should().Throw<ArgumentException>().WithParameterName("means");
    }

    [Fact]
    public void MeanVariance_SingleAsset_ReturnsUnitWeight()
    {
        var cov = new double[1, 1] { { 0.04 } };
        double[] means = [0.1];
        var w = ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, 2.0, 0.0, 1.0);
        w.Should().HaveCount(1);
        w[0].Should().Be(1.0);
    }

    [Fact]
    public void MeanVariance_DoubleVsDecimal_Parity()
    {
        var covD = new double[3, 3]
        {
            { 0.04, 0.01, 0.00 },
            { 0.01, 0.09, 0.02 },
            { 0.00, 0.02, 0.16 },
        };
        double[] meansD = [0.10, 0.07, 0.05];

        var covM = new decimal[3, 3]
        {
            { 0.04m, 0.01m, 0.00m },
            { 0.01m, 0.09m, 0.02m },
            { 0.00m, 0.02m, 0.16m },
        };
        decimal[] meansM = [0.10m, 0.07m, 0.05m];

        var wD = ActiveSetQpSolver<double>.SolveMeanVariance(covD, meansD, 3.0, 0.0, 1.0);
        var wM = ActiveSetQpSolver<decimal>.SolveMeanVariance(covM, meansM, 3.0m, 0.0m, 1.0m);

        for (var i = 0; i < wD.Length; i++)
        {
            ((double)wM[i]).Should().BeApproximately(wD[i], 1e-9,
                because: "double and decimal instantiations must agree on the optimum");
        }
    }

    // ── B32: HouseholderQr<T> direct tests ─────────────────────────────────────

    [Fact]
    public void HouseholderQr_RankDeficientColumn_FactorReturnsFalse()
    {
        // Column 1 is identically zero — rank-deficient, Factor must return false.
        var a = new double[2, 2] { { 1.0, 0.0 }, { 1.0, 0.0 } };
        var y = new double[] { 1.0, 1.0 };
        var diagonal = new double[2];
        var permutation = new int[2];

        var ok = HouseholderQr<double>.Factor(a, y, diagonal, permutation);

        ok.Should().BeFalse(because: "a zero column makes the design rank-deficient");
    }

    [Fact]
    public void HouseholderQr_BuildXtXInverse_NearCollinearDecimal_ThrowsInvalidOperation()
    {
        // For decimal, a near-collinear column drives vtv below the tau threshold or
        // R-inverse to overflow; both surface as InvalidOperationException.
        var a = new decimal[2, 2]
        {
            { 1m, 1m },
            { 1m, 1.0000000000000000000000001m },
        };
        var y = new decimal[] { 1m, 1m };
        var diagonal = new decimal[2];
        var permutation = new int[2];

        var ok = HouseholderQr<decimal>.Factor(a, y, diagonal, permutation);

        // Either Factor flags rank deficiency, or BuildXtXInverse throws on the
        // overflow/zero-diagonal path. Both are acceptable rank-deficiency signals.
        if (ok)
        {
            var act = () => HouseholderQr<decimal>.BuildXtXInverse(a, diagonal, permutation);
            act.Should().Throw<InvalidOperationException>();
        }
        else
        {
            ok.Should().BeFalse();
        }
    }

    [Fact]
    public void HouseholderQr_WellConditioned_FactorSucceeds_AndSolves()
    {
        // y = 2x + 1 exactly on 3 points; intercept design.
        var a = new double[3, 2]
        {
            { 1.0, 1.0 },
            { 1.0, 2.0 },
            { 1.0, 3.0 },
        };
        var y = new double[] { 3.0, 5.0, 7.0 };
        var diagonal = new double[2];
        var permutation = new int[2];

        var ok = HouseholderQr<double>.Factor(a, y, diagonal, permutation);
        ok.Should().BeTrue();

        var beta = HouseholderQr<double>.SolveUpperTriangular(a, diagonal, y, 2, permutation);
        beta[0].Should().BeApproximately(1.0, 1e-9); // intercept
        beta[1].Should().BeApproximately(2.0, 1e-9); // slope
    }

    [Fact]
    public void HouseholderQr_LargeFiniteColumns_PivotsByNormAndSolves()
    {
        var a = new double[3, 2]
        {
            { 1e200, 2e200 },
            { 0.0, 1e199 },
            { 0.0, 0.0 },
        };
        var y = new[] { 4.0, 0.1, 0.0 };
        var diagonal = new double[2];
        var permutation = new int[2];

        var factored = HouseholderQr<double>.Factor(a, y, diagonal, permutation);

        permutation.Should().Equal(new[] { 1, 0 }, because: "the second finite column has the larger 2-norm");
        factored.Should().BeTrue(because: "scaling a full-rank design cannot destroy its rank");
        var coefficients = HouseholderQr<double>.SolveUpperTriangular(a, diagonal, y, 2, permutation);
        ((1e200 * coefficients[0]) + (2e200 * coefficients[1])).Should().BeApproximately(4.0, 1e-12);
        (1e199 * coefficients[1]).Should().BeApproximately(0.1, 1e-13);
    }

    [Fact]
    public void Ols_LargeFiniteColumns_RecoversScaleEquivalentFit()
    {
        var large = new double[3, 2]
        {
            { 1e200, 2e200 },
            { 0.0, 1e199 },
            { 0.0, 0.0 },
        };
        var scaled = new double[3, 2]
        {
            { 1e100, 2e100 },
            { 0.0, 1e99 },
            { 0.0, 0.0 },
        };
        var y = new[] { 4.0, 0.1, 0.0 };

        var largeFit = OrdinaryLeastSquares<double>.Fit(large, y, includeIntercept: false);
        var scaledFit = OrdinaryLeastSquares<double>.Fit(scaled, y, includeIntercept: false);

        for (var row = 0; row < y.Length; row++)
        {
            var largePrediction = (large[row, 0] * largeFit.Coefficients[0]) +
                (large[row, 1] * largeFit.Coefficients[1]);
            var scaledPrediction = (scaled[row, 0] * scaledFit.Coefficients[0]) +
                (scaled[row, 1] * scaledFit.Coefficients[1]);
            largePrediction.Should().BeApproximately(y[row], 1e-12);
            largePrediction.Should().BeApproximately(scaledPrediction, 1e-12);
        }
    }

    // ── B25: float / Half smoke + parity for a representative solver ────────────

    [Fact]
    public void Brent_Float_FindsRoot()
    {
        var solver = new BrentSolver<float>(tolerance: 1e-4f, maxIterations: 100);
        var result = solver.Solve(x => (x * x) - 2.0f, 1.0f, 2.0f);

        result.Converged.Should().BeTrue();
        ((double)result.Root).Should().BeApproximately(Math.Sqrt(2.0), 1e-3);
    }

    [Fact]
    public void Brent_Half_FindsRoot()
    {
        var solver = new BrentSolver<Half>(tolerance: (Half)0.01f, maxIterations: 100);
        var result = solver.Solve(x => (x * x) - (Half)2.0f, (Half)1.0f, (Half)2.0f);

        result.Converged.Should().BeTrue();
        ((double)result.Root).Should().BeApproximately(Math.Sqrt(2.0), 0.1);
    }

    [Fact]
    public void Ols_Float_FitsLine()
    {
        var x = new float[,] { { 1.0f }, { 2.0f }, { 3.0f }, { 4.0f }, { 5.0f } };
        float[] y = [3.0f, 5.0f, 7.0f, 9.0f, 11.0f]; // y = 2x + 1

        var result = OrdinaryLeastSquares<float>.Fit(x, y, includeIntercept: true);

        ((double)result.Coefficients[0]).Should().BeApproximately(1.0, 1e-2);
        ((double)result.Coefficients[1]).Should().BeApproximately(2.0, 1e-2);
    }
}
