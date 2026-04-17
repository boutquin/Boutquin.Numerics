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
    public void NewtonRaphson_WithAnalyticDerivative_ConvergesQuickly()
    {
        NewtonRaphsonSolver solver = new(derivative: QuadraticDerivative);
        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-12);
        result.Iterations.Should().BeLessThanOrEqualTo(10);
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
    public void Solver_ShouldHandleRootAtBoundary()
    {
        BrentSolver solver = new();
        var result = solver.Solve(x => x - 1.0, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(1.0, 1e-12);
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

    // ── Halving interval guarantee tests ─────────────────────────

    [Fact]
    public void BrentSolver_HalvingGuarantee_ConvergesOnSlowFunction()
    {
        // A function that is nearly flat near the root, where standard Brent
        // may take many iterations without halving the bracket.
        static double SlowFunc(double x) => Math.Atan(1000 * (x - 0.5));

        var solver = new BrentSolver(tolerance: 1e-12, maxIterations: 200);
        var result = solver.Solve(SlowFunc, 0.0, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(0.5, 1e-10);

        // With halving guarantee, should converge in at most ceil(log2(1/1e-12)) ≈ 40 iterations.
        result.Iterations.Should().BeLessThanOrEqualTo(60);
    }

    [Fact]
    public void BrentSolver_HalvingGuarantee_StillConvergesSuperlinearly()
    {
        // On well-behaved functions, halving check should not slow convergence.
        var solver = new BrentSolver();
        var result = solver.Solve(Quadratic, 1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
        // Brent on sqrt(2) typically converges in ~5-10 iterations; halving check
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
