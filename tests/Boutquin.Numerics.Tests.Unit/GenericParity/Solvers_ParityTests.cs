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

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

/// <summary>
/// Parity tests verifying that the generic solver types instantiated at
/// <c>T = double</c> produce identical results to the pre-migration legacy
/// concrete-typed implementations. These tests are the gate that proves
/// no behaviour changed for existing consumers during the generic-math migration.
/// </summary>
public sealed class Solvers_ParityTests
{
    // ─── BrentSolver ──────────────────────────────────────────────────

    [Fact]
    public void BrentSolver_GenericMatchesLegacy_AtDouble()
    {
        var legacy = new BrentSolver(tolerance: 1e-12, maxIterations: 100);
        var generic = new BrentSolver<double>(tolerance: 1e-12, maxIterations: 100);

        Func<double, double> f = x => x * x - 2.0;

        var lr = legacy.Solve(f, 0.0, 2.0);
        var gr = generic.Solve(f, 0.0, 2.0);

        gr.Root.Should().Be(lr.Root, because: "generic Brent root must be bit-identical to legacy");
        gr.Converged.Should().Be(lr.Converged);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.FinalResidual.Should().Be(lr.FinalResidual);
        gr.EstimatedError.Should().Be(lr.EstimatedError);
    }

    // ─── BisectionSolver ──────────────────────────────────────────────

    [Fact]
    public void BisectionSolver_GenericMatchesLegacy_AtDouble()
    {
        var legacy = new BisectionSolver(functionTolerance: 1e-12, bracketTolerance: 1e-12, maxIterations: 200);
        var generic = new BisectionSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12, maxIterations: 200);

        Func<double, double> f = x => x * x * x - 1.0;

        var lr = legacy.Solve(f, 0.0, 2.0);
        var gr = generic.Solve(f, 0.0, 2.0);

        gr.Root.Should().Be(lr.Root, because: "generic Bisection root must be bit-identical to legacy");
        gr.Converged.Should().Be(lr.Converged);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.FinalResidual.Should().Be(lr.FinalResidual);
        gr.EstimatedError.Should().Be(lr.EstimatedError);
    }

    // ─── SecantSolver ─────────────────────────────────────────────────

    [Fact]
    public void SecantSolver_GenericMatchesLegacy_AtDouble()
    {
        var legacy = new SecantSolver(tolerance: 1e-12, maxIterations: 50);
        var generic = new SecantSolver<double>(tolerance: 1e-12, maxIterations: 50);

        Func<double, double> f = x => Math.Exp(x) - 3.0;

        var lr = legacy.Solve(f, 1.0);
        var gr = generic.Solve(f, 1.0);

        gr.Root.Should().Be(lr.Root, because: "generic Secant root must be bit-identical to legacy");
        gr.Converged.Should().Be(lr.Converged);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.FinalResidual.Should().Be(lr.FinalResidual);
        gr.EstimatedError.Should().Be(lr.EstimatedError);
    }

    // ─── MullerSolver ─────────────────────────────────────────────────

    [Fact]
    public void MullerSolver_GenericMatchesLegacy_AtDouble()
    {
        var legacy = new MullerSolver(tolerance: 1e-12, maxIterations: 50);
        var generic = new MullerSolver<double>(tolerance: 1e-12, maxIterations: 50);

        Func<double, double> f = x => x * x - 5.0;

        var lr = legacy.Solve(f, 2.0);
        var gr = generic.Solve(f, 2.0);

        gr.Root.Should().Be(lr.Root, because: "generic Muller root must be bit-identical to legacy");
        gr.Converged.Should().Be(lr.Converged);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.FinalResidual.Should().Be(lr.FinalResidual);
        gr.EstimatedError.Should().Be(lr.EstimatedError);
    }

    // ─── NewtonRaphsonSolver ──────────────────────────────────────────

    [Fact]
    public void NewtonRaphsonSolver_GenericMatchesLegacy_Bracketed_AtDouble()
    {
        var legacy = new NewtonRaphsonSolver(functionTolerance: 1e-12, bracketTolerance: 1e-12);
        var generic = new NewtonRaphsonSolver<double>(functionTolerance: 1e-12, bracketTolerance: 1e-12, stepTolerance: 0.0);

        Func<double, double> f = x => x * x - 7.0;

        var lr = legacy.Solve(f, 1.0, 4.0);
        var gr = generic.Solve(f, 1.0, 4.0);

        gr.Root.Should().Be(lr.Root, because: "generic Newton-Raphson root must be bit-identical to legacy");
        gr.Converged.Should().Be(lr.Converged);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.FinalResidual.Should().Be(lr.FinalResidual);
        gr.EstimatedError.Should().Be(lr.EstimatedError);
    }

    [Fact]
    public void NewtonRaphsonSolver_GenericMatchesLegacy_Unbracketed_AtDouble()
    {
        Func<double, double> f = Math.Sin;
        Func<double, double> df = Math.Cos;

        var legacy = new NewtonRaphsonSolver(
            functionTolerance: 1e-12, bracketTolerance: 1e-12, derivative: df);
        var generic = new NewtonRaphsonSolver<double>(
            functionTolerance: 1e-12, bracketTolerance: 1e-12, stepTolerance: 0.0, derivative: df);

        var lr = legacy.Solve(f, 3.0);
        var gr = generic.Solve(f, 3.0);

        gr.Root.Should().Be(lr.Root, because: "generic Newton-Raphson (unbracketed) root must be bit-identical to legacy");
        gr.Converged.Should().Be(lr.Converged);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.FinalResidual.Should().Be(lr.FinalResidual);
        gr.EstimatedError.Should().Be(lr.EstimatedError);
    }

    // ─── LevenbergMarquardtSolver ─────────────────────────────────────

    [Fact]
    public void LevenbergMarquardtSolver_GenericMatchesLegacy_AtDouble()
    {
        // Rosenbrock function: r = [10*(x2 - x1^2), 1 - x1]
        static double[] Residuals(double[] p) => [10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]];

        var legacy = new LevenbergMarquardtSolver(
            maxIterations: 200, functionTolerance: 1e-10, parameterTolerance: 1e-10,
            gradientTolerance: 1e-10, initialDamping: 1e-3);
        var generic = new LevenbergMarquardtSolver<double>(200, 1e-10, 1e-10, 1e-10, 1e-3);

        var lr = legacy.Solve(Residuals, [-1.2, 1.0]);
        var gr = generic.Solve(Residuals, [-1.2, 1.0]);

        for (var i = 0; i < lr.Parameters.Length; i++)
        {
            gr.Parameters[i].Should().Be(lr.Parameters[i],
                because: $"generic LM parameter[{i}] must be bit-identical to legacy");
        }

        gr.FinalCost.Should().Be(lr.FinalCost);
        gr.Iterations.Should().Be(lr.Iterations);
        gr.Converged.Should().Be(lr.Converged);
        gr.TerminationReason.Should().Be(lr.TerminationReason);
    }

    // ─── OrdinaryLeastSquares ─────────────────────────────────────────

    [Fact]
    public void OrdinaryLeastSquares_GenericMatchesLegacy_AtDouble()
    {
        // Simple y = 2x + 1 with noise-free data.
        var x = new double[5, 1] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 } };
        var y = new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 };

        var lr = OrdinaryLeastSquares.Fit(x, y);
        var gr = OrdinaryLeastSquares<double>.Fit(x, y);

        for (var i = 0; i < lr.Coefficients.Length; i++)
        {
            gr.Coefficients[i].Should().Be(lr.Coefficients[i],
                because: $"generic OLS coefficient[{i}] must be bit-identical to legacy");
        }

        gr.ResidualSumOfSquares.Should().Be(lr.ResidualSumOfSquares);
        gr.RSquared.Should().Be(lr.RSquared);
        gr.DegreesOfFreedom.Should().Be(lr.DegreesOfFreedom);
    }
}
