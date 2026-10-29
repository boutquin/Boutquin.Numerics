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

public sealed class ArenaR2B02RootSolverTests
{
    // ── Muller: a stalled step is not a root ─────────────────────────────

    [Theory]
    [InlineData(1e14, 1.0, 1e-6)]
    [InlineData(1e20, 1e-5, 1e-12)]
    [InlineData(1e10, 0.5, 1e-4)]
    [InlineData(1e8, 1.0, 1e-3)]
    public void Muller_SteepRootFreeParabola_DoesNotReportConvergence(double curvature, double floor, double tolerance)
    {
        // curvature·(x − 2)² + floor ≥ floor > tolerance everywhere: no real root exists.
        var solver = new MullerSolver<double>(tolerance, maxIterations: 200);

        var result = solver.Solve(x => curvature * (x - 2.0) * (x - 2.0) + floor, 2.5);

        result.Converged.Should().BeFalse(
            "the function has no real root; the residual at the returned point is {0}", result.FinalResidual);
    }

    [Fact]
    public void Muller_MonotoneFlatteningRootFreeFunction_DoesNotReportConvergence()
    {
        // e^x + 0.5 > 0.5 everywhere: steps shrink towards −∞ but no root exists.
        var solver = new MullerSolver<double>(1e-6, maxIterations: 200);

        var result = solver.Solve(x => Math.Exp(x) + 0.5, 1.0);

        result.Converged.Should().BeFalse(
            "e^x + 0.5 has no real root; the residual at the returned point is {0}", result.FinalResidual);
    }

    [Fact]
    public void Muller_MonotoneFlatteningRootFreeFunction_Float_DoesNotReportConvergence()
    {
        var solver = new MullerSolver<float>(1e-6f, maxIterations: 200);

        var result = solver.Solve(x => MathF.Exp(x) + 0.5f, 1.0f);

        result.Converged.Should().BeFalse(
            "e^x + 0.5 has no real root; the residual at the returned point is {0}", result.FinalResidual);
    }

    [Theory]
    [InlineData(1e-6)]
    [InlineData(3e-12)]
    [InlineData(0.5)]
    public void Muller_RootFreeFunctionOnSubToleranceScale_DoesNotReportConvergence(double initialGuess)
    {
        // (1e12·x)² + 1 ≥ 1 everywhere; its x-scale (1e-12) is below the tolerance, so an
        // absolute sub-tolerance step says nothing about a root.
        var solver = new MullerSolver<double>(1e-10, maxIterations: 100);

        var result = solver.Solve(x => (1e12 * x) * (1e12 * x) + 1.0, initialGuess);

        result.Converged.Should().BeFalse(
            "the function has no real root; the residual at the returned point is {0}", result.FinalResidual);
    }

    [Fact]
    public void Muller_SignChangeAcrossRelativelySmallStep_CertifiesLargeMagnitudeRoot()
    {
        // The root sits halfway between two adjacent doubles near 1.87e6, so |f| ≥ 1e4·ulp/2
        // ≈ 1.2e-6 at every double — far above the 1e-10 tolerance — and ulp(root) ≈ 2.3e-10
        // exceeds the absolute tolerance, so no absolute step can certify it. Muller's
        // stopping rule is the relative step |xᵢ − xᵢ₋₁|/|xᵢ| (Muller 1956, p. 210); gated on
        // the sign change that makes the step a bracket, it certifies this root.
        const double anchor = 1871255.6827213876;
        const double k = 1e4;
        const double c3 = 5.2261705534654535E-14;
        var halfUlp = (Math.BitIncrement(anchor) - anchor) / 2;
        var root = anchor + halfUlp;
        var solver = new MullerSolver<double>(1e-10, maxIterations: 100);

        var result = solver.Solve(
            x => k * ((x - anchor) - halfUlp) + k * c3 * Math.Pow(x - root, 3),
            2036084.7168888669);

        result.Converged.Should().BeTrue();
        Math.Abs(result.FinalResidual).Should().BeGreaterThan(1e-10, "the residual cannot certify this root");
        result.Root.Should().BeApproximately(root, 2 * halfUlp);
    }

    // ── Bracketed Newton: never evaluate outside the caller's bracket ────

    [Theory]
    [InlineData(0.0, 1.0, 1e-3)]
    [InlineData(0.0, 4.0, 1e-4)]
    public void NewtonBracketed_NumericalDerivative_StaysInsideCallerBracket_Sqrt(double lower, double upper, double target)
    {
        // √x − target is undefined (NaN) for x < 0: the solver must not evaluate there.
        var solver = new NewtonRaphsonSolver<double>(1e-14, 1e-15, 0.0, maxIterations: 200);
        var minEvaluated = double.PositiveInfinity;
        var maxEvaluated = double.NegativeInfinity;

        RootSolverResult<double>? result = null;

        var thrown = Record.Exception(() => result = solver.Solve(
            x =>
            {
                minEvaluated = Math.Min(minEvaluated, x);
                maxEvaluated = Math.Max(maxEvaluated, x);
                return Math.Sqrt(x) - target;
            },
            lower,
            upper));

        minEvaluated.Should().BeGreaterThanOrEqualTo(lower);
        maxEvaluated.Should().BeLessThanOrEqualTo(upper);
        thrown.Should().BeNull();
        result!.Converged.Should().BeTrue();
        result!.Root.Should().BeApproximately(target * target, 1e-12);
    }

    [Fact]
    public void NewtonBracketed_NumericalDerivative_StaysInsideCallerBracket_Log()
    {
        // log(x) + 20 has its root at e^−20 ≈ 2.06e-9, far below the ∛ε finite-difference floor.
        const double lower = 1e-12;
        const double upper = 1.0;
        var solver = new NewtonRaphsonSolver<double>(1e-12, 1e-20, 0.0, maxIterations: 200);
        var minEvaluated = double.PositiveInfinity;

        RootSolverResult<double>? result = null;

        var thrown = Record.Exception(() => result = solver.Solve(
            x =>
            {
                minEvaluated = Math.Min(minEvaluated, x);
                return Math.Log(x) + 20.0;
            },
            lower,
            upper));

        minEvaluated.Should().BeGreaterThanOrEqualTo(lower);
        thrown.Should().BeNull();
        result!.Converged.Should().BeTrue();
        result!.Root.Should().BeApproximately(Math.Exp(-20.0), 1e-15);
    }
}
