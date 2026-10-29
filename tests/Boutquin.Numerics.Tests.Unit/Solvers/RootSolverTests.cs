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

public sealed class RootSolverTests
{
    private static double Quadratic(double x) => x * x - 2.0;
    private static double QuadraticDerivative(double x) => 2.0 * x;
    private static double Cubic(double x) => x * x * x - x - 1.0;

    private static Func<double, double> OisPvFunction(double rate, double tau)
    {
        return df => rate * tau * df + df - 1.0;
    }

    // ── Bracketed solver tests ──────────────────────────────────

    [Theory]
    [InlineData(typeof(BisectionSolver))]
    [InlineData(typeof(BrentSolver))]
    [InlineData(typeof(NewtonRaphsonSolver))]
    public void BracketedSolver_ShouldFindRoot_OfQuadratic(Type solverType)
    {
        var solver = CreateBracketedSolver(solverType);
        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
        result.Iterations.Should().BeGreaterThan(0);
    }

    [Theory]
    [InlineData(typeof(BisectionSolver))]
    [InlineData(typeof(BrentSolver))]
    [InlineData(typeof(NewtonRaphsonSolver))]
    public void BracketedSolver_ShouldFindRoot_OfCubic(Type solverType)
    {
        var solver = CreateBracketedSolver(solverType);
        var result = solver.Solve(Cubic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.3247179572, 1e-8);
    }

    [Theory]
    [InlineData(typeof(BisectionSolver))]
    [InlineData(typeof(BrentSolver))]
    [InlineData(typeof(NewtonRaphsonSolver))]
    public void BracketedSolver_ShouldFindDiscountFactor_ForOisPv(Type solverType)
    {
        var solver = CreateBracketedSolver(solverType);
        double rate = 0.05;
        double tau = 2.0;
        double expectedDf = 1.0 / (1.0 + rate * tau);

        var result = solver.Solve(OisPvFunction(rate, tau), 0.5, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(expectedDf, 1e-10);
    }

    [Theory]
    [InlineData(typeof(BisectionSolver))]
    [InlineData(typeof(BrentSolver))]
    [InlineData(typeof(NewtonRaphsonSolver))]
    public void BracketedSolver_ShouldThrow_WhenBracketDoesNotContainRoot(Type solverType)
    {
        var solver = CreateBracketedSolver(solverType);

        Action act = () => solver.Solve(Quadratic, 2.0, 3.0);
        act.Should().Throw<InvalidOperationException>();
    }

    [Theory]
    [InlineData(typeof(BisectionSolver), double.NaN, 2.0, "lowerBound")]
    [InlineData(typeof(BrentSolver), double.NaN, 2.0, "lowerBound")]
    [InlineData(typeof(NewtonRaphsonSolver), double.NaN, 2.0, "lowerBound")]
    [InlineData(typeof(BisectionSolver), -1.0, double.NaN, "upperBound")]
    [InlineData(typeof(BrentSolver), -1.0, double.NaN, "upperBound")]
    [InlineData(typeof(NewtonRaphsonSolver), -1.0, double.NaN, "upperBound")]
    [InlineData(typeof(BisectionSolver), double.NegativeInfinity, 2.0, "lowerBound")]
    [InlineData(typeof(BrentSolver), double.NegativeInfinity, 2.0, "lowerBound")]
    [InlineData(typeof(NewtonRaphsonSolver), double.NegativeInfinity, 2.0, "lowerBound")]
    [InlineData(typeof(BisectionSolver), -1.0, double.PositiveInfinity, "upperBound")]
    [InlineData(typeof(BrentSolver), -1.0, double.PositiveInfinity, "upperBound")]
    [InlineData(typeof(NewtonRaphsonSolver), -1.0, double.PositiveInfinity, "upperBound")]
    public void BracketedSolver_RejectsNonFiniteEndpoints(
        Type solverType, double lowerBound, double upperBound, string parameterName)
    {
        var solver = CreateBracketedSolver(solverType);
        Action solve = () => solver.Solve(_ => 0.0, lowerBound, upperBound);

        solve.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(parameterName);
    }

    [Theory]
    [InlineData(typeof(BisectionSolver))]
    [InlineData(typeof(BrentSolver))]
    [InlineData(typeof(NewtonRaphsonSolver))]
    public void BracketedSolver_RejectsReversedEndpoints(Type solverType)
    {
        var solver = CreateBracketedSolver(solverType);
        Action solve = () => solver.Solve(x => x - 1.0, 2.0, 0.0);

        solve.Should().Throw<ArgumentException>().WithParameterName("lowerBound");
    }

    // ── Unbracketed solver tests ────────────────────────────────

    [Theory]
    [InlineData(typeof(NewtonRaphsonSolver))]
    [InlineData(typeof(SecantSolver))]
    [InlineData(typeof(MullerSolver))]
    public void UnbracketedSolver_ShouldFindRoot_OfQuadratic(Type solverType)
    {
        var solver = CreateUnbracketedSolver(solverType);
        var result = solver.Solve(Quadratic, 1.5);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-8);
    }

    [Theory]
    [InlineData(typeof(NewtonRaphsonSolver))]
    [InlineData(typeof(SecantSolver))]
    [InlineData(typeof(MullerSolver))]
    public void UnbracketedSolver_ShouldFindRoot_OfCubic(Type solverType)
    {
        var solver = CreateUnbracketedSolver(solverType);
        var result = solver.Solve(Cubic, 1.5);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.3247179572, 1e-6);
    }

    // ── Specific solver tests ───────────────────────────────────

    [Fact]
    public void BrentSolver_ShouldConvergeFasterThanBisection()
    {
        BisectionSolver bisection = new();
        BrentSolver brent = new();

        var bisectionResult = bisection.Solve(Quadratic, 1.0, 2.0);
        var brentResult = brent.Solve(Quadratic, 1.0, 2.0);

        brentResult.Iterations.Should().BeLessThan(bisectionResult.Iterations);
    }

    [Fact]
    public void SecantSolver_TwoPoint_ShouldConvergeFasterThanBisection()
    {
        BisectionSolver bisection = new();
        SecantSolver secant = new();

        var bisectionResult = bisection.Solve(Quadratic, 1.0, 2.0);
        var secantResult = secant.Solve(Quadratic, 1.0, 2.0);

        secantResult.Iterations.Should().BeLessThan(bisectionResult.Iterations);
    }

    [Fact]
    public void SecantSolver_SingleGuess_MaxFiniteExactRoot_Converges()
    {
        var solver = new SecantSolver<double>(1e-12);
        RootSolverResult<double>? result = null;

        Action solve = () => result = solver.Solve(
            x => x == double.MaxValue ? 0.0 : throw new InvalidOperationException("Exact root needs no neighbor"),
            double.MaxValue);

        solve.Should().NotThrow("a finite exact root must not produce a non-finite generated seed");
        result!.Converged.Should().BeTrue();
        result.Root.Should().Be(double.MaxValue);
        result.Iterations.Should().Be(0);
    }

    [Fact]
    public void SecantSolver_SingleGuess_HalfMax_GeneratesDistinctFiniteNeighbor()
    {
        var solver = new SecantSolver<Half>((Half)0.0001);
        var points = new List<Half>();

        _ = solver.Solve(x => { points.Add(x); return (Half)1; }, Half.MaxValue);

        points.Distinct().Count().Should().BeGreaterThanOrEqualTo(2);
        points.Should().OnlyContain(x => Half.IsFinite(x));
    }

    [Fact]
    public void NewtonRaphson_WithAnalyticDerivative_ConvergesQuickly()
    {
        NewtonRaphsonSolver solver = new(derivative: QuadraticDerivative);
        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-12);
        result.Iterations.Should().BeLessThanOrEqualTo(10);
    }

    [Fact]
    public void NewtonRaphson_BracketedRootReachedOnFinalIteration_ReportsConvergence()
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0,
            maxIterations: 1, derivative: _ => 1.0);

        var result = solver.Solve(x => x - 1.0, 0.0, 4.0);

        result.Root.Should().Be(1.0);
        result.FinalResidual.Should().Be(0.0);
        result.Converged.Should().BeTrue();
        result.Iterations.Should().Be(1);
    }

    [Fact]
    public void NewtonRaphson_Unbracketed_WithDerivative_ConvergesQuickly()
    {
        NewtonRaphsonSolver solver = new(derivative: QuadraticDerivative);
        var result = solver.Solve(Quadratic, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-12);
        result.Iterations.Should().BeLessThanOrEqualTo(5);
    }

    [Fact]
    public void NewtonRaphson_Unbracketed_MaxFiniteGuess_UsesFiniteDifferenceWithinDomain()
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0);

        RootSolverResult<double>? result = null;
        Action solve = () => result = solver.Solve(x => x - 1e307, double.MaxValue);

        solve.Should().NotThrow("finite guesses should only evaluate finite perturbations");
        result!.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1e307, 1e292);
    }

    [Fact]
    public void NewtonRaphson_Unbracketed_SmallStepWithLargeResidualDoesNotConverge()
    {
        static double Objective(double x) => 1.0 + 0.1 * Math.Tanh(1e14 * x) - x;
        static double Derivative(double x)
        {
            var cosh = Math.Cosh(1e14 * x);
            return 1e13 / (cosh * cosh) - 1.0;
        }

        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 1e-12,
            maxIterations: 1, derivative: Derivative);

        var result = solver.Solve(Objective, 0.0);

        result.Converged.Should().BeFalse("the residual remains far above function tolerance");
        Math.Abs(result.FinalResidual).Should().BeGreaterThan(0.5);
    }

    [Fact]
    public void Solver_ShouldHandleRootAtBoundary()
    {
        BrentSolver solver = new();
        var result = solver.Solve(x => x - 1.0, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void BrentSolver_RootReachedOnFinalIteration_ReportsConvergence()
    {
        var solver = new BrentSolver<double>(1e-12, maxIterations: 1);

        var result = solver.Solve(x => x - 1.0, 0.0, 3.0);

        result.Root.Should().Be(1.0);
        result.FinalResidual.Should().Be(0.0);
        result.Converged.Should().BeTrue();
        result.Iterations.Should().Be(1);
    }

    [Fact]
    public void BracketedSolver_ResultContains_NonNegativeEstimatedError()
    {
        BisectionSolver solver = new();
        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.EstimatedError.Should().BeGreaterThanOrEqualTo(0.0);
        double.IsFinite(result.EstimatedError).Should().BeTrue();
    }

    [Fact]
    public void BracketedSolver_LargeFunctionValues_DoNotOverflow()
    {
        static double LargeFunc(double x) => x * 1e200 - 1e200;

        BisectionSolver bisection = new();
        var result = bisection.Solve(LargeFunc, 0.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void BisectionSolver_WideFiniteBracket_DoesNotOverflowMidpoint()
    {
        var solver = new BisectionSolver<double>(1e-12, 1e-12);
        RootSolverResult<double>? result = null;

        Action solve = () => result = solver.Solve(x => x, -1e308, 1e308);

        solve.Should().NotThrow("the finite bracket contains a root at its midpoint");
        result!.Converged.Should().BeTrue();
        result.Root.Should().Be(0.0);
        double.IsFinite(result.EstimatedError).Should().BeTrue();
    }

    [Fact]
    public void NewtonRaphson_Bracketed_WideFiniteBracket_DoesNotOverflowMidpoint()
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0,
            derivative: _ => 1.0);
        RootSolverResult<double>? result = null;

        Action solve = () => result = solver.Solve(x => x, -1e308, 1e308);

        solve.Should().NotThrow("the finite bracket contains a root at its midpoint");
        result!.Converged.Should().BeTrue();
        result.Root.Should().Be(0.0);
        double.IsFinite(result.EstimatedError).Should().BeTrue();
    }

    [Fact]
    public void BrentSolver_WideFiniteBracket_ConvergesWithinFiniteBracket()
    {
        var solver = new BrentSolver<double>(1e-12);
        var lower = -1.1764505877282705e308;
        var upper = 1.1291763642938697e308;
        var evaluatedOutsideBracket = false;
        RootSolverResult<double>? result = null;

        Action solve = () => result = solver.Solve(x =>
        {
            evaluatedOutsideBracket |= !double.IsFinite(x) || x < lower || x > upper;
            return x / 1e308;
        }, lower, upper);

        solve.Should().NotThrow("a finite sign-changing bracket must remain representable");
        evaluatedOutsideBracket.Should().BeFalse();
        result!.Converged.Should().BeTrue();
        Math.Abs(result.Root).Should().BeLessThan(1e296, "the residual tolerance is applied to x / 1e308");
        double.IsFinite(result.EstimatedError).Should().BeTrue();
    }

    [Fact]
    public void SecantSolver_WideFiniteInitialPoints_ConvergesAtFiniteRoot()
    {
        var solver = new SecantSolver<double>(1e-12);
        var evaluatedNonFinite = false;
        RootSolverResult<double>? result = null;

        Action solve = () => result = solver.Solve(x =>
        {
            evaluatedNonFinite |= !double.IsFinite(x);
            return x / 1e308;
        }, -1e308, 1e308);

        solve.Should().NotThrow("finite initial points with a finite secant root must remain representable");
        evaluatedNonFinite.Should().BeFalse();
        result!.Converged.Should().BeTrue();
        result.Root.Should().Be(0.0);
    }

    [Fact]
    public void SecantSolver_NonFiniteExtrapolation_DoesNotEvaluateObjective()
    {
        var solver = new SecantSolver<double>(1e-12);
        var evaluatedNonFinite = false;
        RootSolverResult<double>? result = null;
        Action solve = () => result = solver.Solve(x =>
        {
            evaluatedNonFinite |= !double.IsFinite(x);
            return 1.0 + x / 1e308 * 0.01;
        }, -1e308, 1e308);

        solve.Should().NotThrow("an unrepresentable extrapolation is not a valid next iterate");
        evaluatedNonFinite.Should().BeFalse();
        result!.Converged.Should().BeFalse();
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void SecantSolver_RejectsNonFiniteInitialPointsBeforeObjective(double point)
    {
        var solver = new SecantSolver<double>(1e-12);
        var invoked = false;
        double Objective(double _) { invoked = true; return 0.0; }

        Action onePoint = () => solver.Solve(Objective, point);
        Action firstPoint = () => solver.Solve(Objective, point, 1.0);
        Action secondPoint = () => solver.Solve(Objective, 0.0, point);

        onePoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("initialGuess");
        firstPoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("p0");
        secondPoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("p1");
        invoked.Should().BeFalse();
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void MullerSolver_RejectsNonFiniteInitialPointsBeforeObjective(double point)
    {
        var solver = new MullerSolver<double>(1e-12);
        var invoked = false;
        double Objective(double _) { invoked = true; return 0.0; }

        Action onePoint = () => solver.Solve(Objective, point);
        Action firstPoint = () => solver.Solve(Objective, point, 0.0, 1.0);
        Action middlePoint = () => solver.Solve(Objective, -1.0, point, 1.0);
        Action lastPoint = () => solver.Solve(Objective, -1.0, 0.0, point);

        onePoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("initialGuess");
        firstPoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x0");
        middlePoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x1");
        lastPoint.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("x2");
        invoked.Should().BeFalse();
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NewtonRaphson_Unbracketed_RejectsNonFiniteInitialGuessBeforeObjective(double point)
    {
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0);
        var invoked = false;
        Action solve = () => solver.Solve(_ => { invoked = true; return 0.0; }, point);

        solve.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("initialGuess");
        invoked.Should().BeFalse();
    }

    // ── Forced-bisection safeguard tests ─────────────────────────

    [Theory]
    [InlineData("step")]
    [InlineData("lopsided")]
    [InlineData("power")]
    public void BrentSolver_ForcedBisection_BoundsWorstCase(string shape)
    {
        // Shapes on which interpolation stalls, so the forced-bisection branch carries the
        // convergence. The root sits off the bracket midpoint on purpose: a root at the
        // midpoint lets the first bisection land on it exactly and exercises nothing.
        const double root = 0.3;
        const double tolerance = 1e-12;
        Func<double, double> f = shape switch
        {
            "step" => x => x < root ? -1.0 : 1.0,
            "lopsided" => x => x < root ? x - root : 1e9 * (x - root),
            "power" => x => Math.Sign(x - root) * Math.Pow(Math.Abs(x - root), 0.01),
            _ => throw new ArgumentOutOfRangeException(nameof(shape), shape, "Unknown test shape."),
        };

        var solver = new BrentSolver(tolerance, maxIterations: 500);
        var result = solver.Solve(f, 0.0, 1.0);

        // The bracket halves at least every two iterations, so the solver needs at most twice
        // bisection's count, plus one for the final convergence check and one for rounding.
        var bisectionCount = (int)Math.Ceiling(Math.Log2(1.0 / tolerance));
        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(root, 1e-9);
        result.Iterations.Should().BeLessThanOrEqualTo(2 * bisectionCount + 2);
    }

    [Fact]
    public void BrentSolver_ForcedBisection_MeetsQuadraticEvaluationBudget()
    {
        // This concrete smooth case has a tested budget; it does not prove an asymptotic rate.
        var solver = new BrentSolver();
        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
        // Brent on sqrt(2) typically converges in ~5-10 iterations; the safeguard
        // may add a few forced bisection steps.
        result.Iterations.Should().BeLessThanOrEqualTo(20);
    }

    // ── Helpers ─────────────────────────────────────────────────

    private static IBracketedRootSolver CreateBracketedSolver(Type solverType)
    {
        return solverType.Name switch
        {
            nameof(BisectionSolver) => new BisectionSolver(),
            nameof(BrentSolver) => new BrentSolver(),
            nameof(NewtonRaphsonSolver) => new NewtonRaphsonSolver(),
            _ => throw new ArgumentException($"Unknown bracketed solver type: {solverType.Name}")
        };
    }

    private static IUnbracketedRootSolver CreateUnbracketedSolver(Type solverType)
    {
        return solverType.Name switch
        {
            nameof(NewtonRaphsonSolver) => new NewtonRaphsonSolver(),
            nameof(SecantSolver) => new SecantSolver(),
            nameof(MullerSolver) => new MullerSolver(),
            _ => throw new ArgumentException($"Unknown unbracketed solver type: {solverType.Name}")
        };
    }
}
