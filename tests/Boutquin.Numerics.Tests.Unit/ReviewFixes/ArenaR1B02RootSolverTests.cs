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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class ArenaR1B02RootSolverTests
{
    // ── Secant: a stalled step is not a root ─────────────────────────────

    [Theory]
    [InlineData(-2.0)]
    [InlineData(0.5)]
    [InlineData(5.0)]
    public void Secant_RootFreeFunctionMayMeetSelectedSmallStepStop(double initialGuess)
    {
        // cosh(x) >= 1 has no real root. NR rtsec can nevertheless satisfy its
        // small-step stopping rule; retain the residual as the root-accuracy diagnostic.
        var solver = new SecantSolver<double>(1e-12, maxIterations: 200);

        var result = solver.Solve(Math.Cosh, initialGuess);

        result.Converged.Should().BeTrue();
        result.EstimatedError.Should().BeLessThan(1e-12);
        result.FinalResidual.Should().BeGreaterThanOrEqualTo(1);
    }

    [Fact]
    public void Secant_SteepRootWithIrreducibleResidual_StillConverges()
    {
        // The root sits halfway between two adjacent doubles, so f is never zero at a double
        // and |f| >= 1e20*ulp/2, far above tolerance. NR small-step stopping
        // supplies the flag; the independently known root supplies the accuracy check.
        const double anchor = 0.3;
        var halfUlp = (Math.BitIncrement(anchor) - anchor) / 2;
        var solver = new SecantSolver<double>(1e-12, maxIterations: 200);

        var result = solver.Solve(x => 1e20 * ((x - anchor) + halfUlp), 0.0, 1.0);

        result.Converged.Should().BeTrue();
        Math.Abs(result.FinalResidual).Should().BeGreaterThan(1e-12, "the residual cannot certify this root");
        result.Root.Should().BeApproximately(anchor - halfUlp, 1e-12);
    }

    // ── Bisection: an absolute bracket tolerance below ulp(root) ─────────

    [Fact]
    public void Bisection_BracketToleranceBelowUlp_ConvergesAtMachineResolution()
    {
        // Root ≈ 100000.0000035, where ulp = 1.455e-11 > bracketTolerance = 1e-12 and the
        // residual at the nearest double (~1.9e-6) never meets functionTolerance.
        var solver = new BisectionSolver<double>(1e-12, 1e-12, maxIterations: 500);

        var result = solver.Solve(x => x * x - 1.00000000007e10, 0.0, 2e5);

        result.Converged.Should().BeTrue();
        result.Iterations.Should().BeLessThanOrEqualTo(64, "the bracket reaches adjacent doubles in ~54 halvings");
        result.Root.Should().BeApproximately(Math.Sqrt(1.00000000007e10), 2e-11);
    }

    [Fact]
    public void NewtonRaphson_Bracketed_BracketToleranceBelowUlp_ConvergesAtMachineResolution()
    {
        // Same class as the bisection case: the only width test is absolute, so a bracket
        // that reaches adjacent doubles above the tolerance can never stop.
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-12, 0.0, maxIterations: 500);

        var result = solver.Solve(x => x * x - 1.00000000007e10, 0.0, 2e5);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(Math.Sqrt(1.00000000007e10), 2e-11);
    }

    // ── Newton: finite-difference step scaled to the working type ────────

    [Fact]
    public void NewtonRaphson_Float_Unbracketed_NumericalDerivative_Converges()
    {
        var solver = new NewtonRaphsonSolver<float>(1e-5f, 1e-6f, 0f);

        var result = solver.Solve(x => x * x - 2f, 1f);

        result.Converged.Should().BeTrue();
        ((double)result.Root).Should().BeApproximately(Math.Sqrt(2.0), 1e-5);
    }

    [Fact]
    public void NewtonRaphson_Float_Bracketed_NumericalDerivative_TakesNewtonSteps()
    {
        // Pure bisection needs ~20 halvings to reach 1e-6 on [1, 2]; Newton needs a handful.
        var solver = new NewtonRaphsonSolver<float>(1e-5f, 1e-6f, 0f);

        var result = solver.Solve(x => x * x - 2f, 1f, 2f);

        result.Converged.Should().BeTrue();
        result.Iterations.Should().BeLessThanOrEqualTo(8);
    }

    [Fact]
    public void NewtonRaphson_Half_Unbracketed_NumericalDerivative_Converges()
    {
        var solver = new NewtonRaphsonSolver<Half>((Half)0.01f, (Half)0.001f, Half.Zero);

        var result = solver.Solve(x => x * x - (Half)2f, (Half)1f);

        result.Converged.Should().BeTrue();
        ((double)result.Root).Should().BeApproximately(Math.Sqrt(2.0), 0.01);
    }

    // ── Newton: rtsafe's slow-decrease bisection trigger ─────────────────

    [Fact]
    public void NewtonRaphson_Bracketed_MultipleRoot_BisectsWhenNewtonDecreasesSlowly()
    {
        // (x − 0.3)⁹ has a ninefold root: Newton's step shrinks by only 8/9 per iteration.
        // NR rtsafe bisects when |2f| > |dxold·f′| and needs 71 iterations; without that
        // test the solver takes 145, more than four times bisection's 34.
        static double F(double x) => Math.Pow(x - 0.3, 9);
        static double Df(double x) => 9.0 * Math.Pow(x - 0.3, 8);
        var solver = new NewtonRaphsonSolver<double>(1e-300, 1e-10, 0.0, maxIterations: 100, derivative: Df);

        var result = solver.Solve(F, 0.0, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(0.3, 1e-10);
    }

    // ── Brent: stopping is on the bracket, not on the residual ───────────

    [Fact]
    public void Brent_FlatFunction_DoesNotAcceptEndpointOnResidual()
    {
        // |f(0)| = 5e-9 < tolerance, but the root is 0.5 away.
        var solver = new BrentSolver<double>(1e-6);

        var result = solver.Solve(x => 1e-8 * (x - 0.5), 0.0, 1.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(0.5, 1e-6);
    }

    [Fact]
    public void Brent_FlatFunction_InteriorIterateNotAcceptedOnResidual()
    {
        // Endpoint residuals exceed the tolerance, but every point within 0.79 of the root
        // has |f| < tolerance; a residual stop returns the first such interior iterate.
        var solver = new BrentSolver<double>(1e-6);

        var result = solver.Solve(x => 2e-6 * Math.Pow(x - 0.3, 3), -1.0, 2.0);

        result.Converged.Should().BeTrue();
        result.Root.Should().BeApproximately(0.3, 1e-6);
    }
}
