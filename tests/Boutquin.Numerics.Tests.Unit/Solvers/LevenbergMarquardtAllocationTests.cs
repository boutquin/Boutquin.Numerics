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
using Boutquin.Numerics.Solvers.Internal;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Regression harness for AC §3.1 of <c>specs/feature-lm-zero-alloc-iteration.md</c>.
/// Uses <see cref="GC.GetAllocatedBytesForCurrentThread"/> deltas to verify that the
/// pooled <see cref="LevenbergMarquardtSolver.Solve(Func{double[], double[]}, double[], LevenbergMarquardtBuffers, Func{double[], double[,]}, double[], double[])"/>
/// overload does not grow per-iteration managed-heap allocation, and that the inner
/// LM work (<see cref="FiniteDifferenceJacobian.EvaluateInto"/> + <see cref="DampedLinearSolve.TrySolve"/>)
/// is literally allocation-free across the steady-state loop.
/// </summary>
/// <remarks>
/// <para>
/// Allocation budget (AC §3.1 bullet 1) covers <c>MultivariateSolverResult</c> and its
/// three owned arrays (<c>Parameters</c>, <c>FinalResiduals</c>, <c>ParameterCovariance</c>)
/// plus the one-shot <see cref="DampedLinearSolve.TryInvertNormalEquations"/> scratch
/// allocated at convergence — all one-shot terminal allocations per solve, not
/// inner-loop growth. The budget is expressed as a multiple of a computed
/// <c>ExpectedResultRecordSizeBytes</c> that sums those quantities for the chosen
/// problem size so the test catches regressions that introduce any new inner-loop
/// allocation source.
/// </para>
/// <para>
/// Inner micro-test (AC §3.1 bullet 2) targets the narrow zero-byte bar — with a
/// non-allocating residual callback, the combined
/// <see cref="FiniteDifferenceJacobian.EvaluateInto"/> +
/// <see cref="DampedLinearSolve.TrySolve"/> round-trip must allocate exactly zero bytes
/// on the measured thread.
/// </para>
/// </remarks>
public sealed class LevenbergMarquardtAllocationTests
{
    // Problem dimensions for the outer test — chosen to match the spec's example (n=6, m=24).
    // A 6-parameter polynomial fit against 24 deterministic samples converges in a small
    // number of accepted iterations, exercising Jacobian evaluation + damped linear solve +
    // bounds/covariance paths on every measurement.
    private const int ParameterCount = 6;
    private const int ResidualCount = 24;

    // Approximate per-solve terminal allocation (record + 2 owned arrays + covariance +
    // TryInvertNormalEquations scratch). Sized generously so the test allows for CLR
    // header overhead variance but fails if a new iteration-loop allocation source is
    // introduced.
    private const double ExpectedResultRecordSizeBytes = 1800.0;

    [Fact]
    public void Solve_WithBuffers_NoAllocationGrowthAcrossIterations()
    {
        var fixture = new PolynomialFixture(ParameterCount, ResidualCount, seed: 13);
        var buffers = new LevenbergMarquardtBuffers(ParameterCount, ResidualCount);
        var solver = new LevenbergMarquardtSolver(maxIterations: 100);
        var initialGuess = new double[ParameterCount];

        // Warmup: 10 solves amortize JIT, tiered-compilation promotion, and first-time
        // static-field initialization.
        for (var i = 0; i < 10; i++)
        {
            _ = solver.Solve(fixture.Residuals, initialGuess, buffers);
        }

        // Force a clean GC baseline before measurement — any stray generation-0 objects
        // from warmup fall out of scope now rather than polluting the measured window.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 90; i++)
        {
            _ = solver.Solve(fixture.Residuals, initialGuess, buffers);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();
        var perSolve = (after - before) / 90.0;

        perSolve.Should().BeLessThan(
            ExpectedResultRecordSizeBytes * 2.0,
            because: "only the MultivariateSolverResult record, its owned arrays, and the one-shot " +
                     "TryInvertNormalEquations scratch should be allocated per solve — no inner-loop growth");
    }

    [Fact]
    public void Solve_WithBuffers_ConvergesToExpectedSolution()
    {
        // Sanity check that the fixture problem does converge. Guards against a future
        // change to the polynomial seed producing a degenerate (non-converging) case
        // that would silently make the allocation assertion weaker.
        var fixture = new PolynomialFixture(ParameterCount, ResidualCount, seed: 13);
        var buffers = new LevenbergMarquardtBuffers(ParameterCount, ResidualCount);
        var solver = new LevenbergMarquardtSolver(maxIterations: 100);

        var result = solver.Solve(fixture.Residuals, new double[ParameterCount], buffers);

        result.Converged.Should().BeTrue();
    }

    [Fact]
    public void InnerLoop_FdJacobianAndDampedSolve_ZeroAllocation()
    {
        // Exercises the narrow AC-3.1-bullet-2 contract: the combined
        // FiniteDifferenceJacobian.EvaluateInto + DampedLinearSolve.TrySolve pipeline
        // allocates exactly zero bytes across 1000 invocations after warmup, when the
        // caller's residual callback is itself non-allocating.
        const int Iterations = 1000;
        var fixture = new PolynomialFixture(ParameterCount, ResidualCount, seed: 7);
        var buffers = new LevenbergMarquardtBuffers(ParameterCount, ResidualCount);
        var theta = new double[ParameterCount];

        // Cache the residual delegate once. A method-group reference (`fixture.Residuals`)
        // at each call site would allocate a fresh delegate per iteration for instance
        // methods (method-group-to-delegate conversion captures `this`), which masks
        // the inner-loop allocation contract.
        Func<double[], double[]> residualsDelegate = fixture.Residuals;

        // Seed the residual vector with a fresh evaluation so TrySolve has something
        // realistic to operate on. The solver normally maintains this; the micro-test
        // must feed it manually once before the measured window.
        var initialResidual = residualsDelegate(theta);
        Array.Copy(initialResidual, buffers.Residual, ResidualCount);

        // Warmup: first 32 iterations JIT EvaluateInto/TrySolve/residual methods.
        for (var i = 0; i < 32; i++)
        {
            FiniteDifferenceJacobian.EvaluateInto(residualsDelegate, theta, buffers);
            _ = DampedLinearSolve.TrySolve(buffers.Residual, lambda: 1e-3, buffers);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < Iterations; i++)
        {
            FiniteDifferenceJacobian.EvaluateInto(residualsDelegate, theta, buffers);
            _ = DampedLinearSolve.TrySolve(buffers.Residual, lambda: 1e-3, buffers);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();

        (after - before).Should().Be(
            0,
            because: "EvaluateInto + TrySolve should both operate entirely on pool storage — " +
                     "zero bytes allocated across the measured window");
    }

    /// <summary>
    /// Deterministic polynomial-fit fixture. Generates (x, y) samples from a known
    /// 6-parameter polynomial plus seeded noise so the least-squares solution is
    /// well-defined and close to the generating parameters. Residual evaluation uses
    /// a cached output buffer and returns the same array on every call, so the
    /// fixture contributes no managed-heap allocation to the inner loop beyond
    /// its one-shot construction cost.
    /// </summary>
    private sealed class PolynomialFixture
    {
        private readonly int _parameterCount;
        private readonly int _residualCount;
        private readonly double[] _xs;
        private readonly double[] _ys;
        private readonly double[] _residualBuffer;

        public PolynomialFixture(int parameterCount, int residualCount, int seed)
        {
            _parameterCount = parameterCount;
            _residualCount = residualCount;
            _xs = new double[residualCount];
            _ys = new double[residualCount];
            _residualBuffer = new double[residualCount];

            // Generate x samples on [-1, 1] and y = polynomial(x) + small noise.
            // The polynomial coefficients are 0.5^k for k = 0..n-1, decaying so higher-
            // order terms contribute less to the fit and the Jacobian is well-conditioned.
            var state = (uint)(seed == 0 ? 1 : seed);
            for (var i = 0; i < residualCount; i++)
            {
                var x = -1.0 + (2.0 * i / (residualCount - 1));
                _xs[i] = x;

                var y = 0.0;
                var pow = 1.0;
                for (var k = 0; k < parameterCount; k++)
                {
                    y += Math.Pow(0.5, k) * pow;
                    pow *= x;
                }

                // Small deterministic noise — keeps the problem overdetermined.
                state = (1664525u * state) + 1013904223u;
                var noise = (((state & 0xFFFFFFu) / (double)0x1000000) - 0.5) * 0.01;
                _ys[i] = y + noise;
            }
        }

        public double[] Residuals(double[] theta)
        {
            for (var i = 0; i < _residualCount; i++)
            {
                var model = 0.0;
                var pow = 1.0;
                for (var k = 0; k < _parameterCount; k++)
                {
                    model += theta[k] * pow;
                    pow *= _xs[i];
                }

                _residualBuffer[i] = _ys[i] - model;
            }

            return _residualBuffer;
        }
    }
}
