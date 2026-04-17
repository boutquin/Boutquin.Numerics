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

public sealed class LevenbergMarquardtSolverTests
{
    // ── Rosenbrock (§3.1 AC #1) ──────────────────────────────────
    // Classic least-squares form: r(x, y) = (10·(y − x²), 1 − x). Minimum at (1, 1) with residual zero.

    [Fact]
    public void Solve_Rosenbrock_ConvergesFromStandardStartingPoint()
    {
        // Spec AC uses analytic-Jacobian iteration counts as the reference;
        // FD-Jacobian takes a few more iterations. Test the stated bound against
        // the analytic form.
        static double[,] RosenbrockJacobian(double[] theta)
        {
            var x = theta[0];
            return new double[,]
            {
                { -20.0 * x, 10.0 },
                { -1.0, 0.0 },
            };
        }

        var solver = new LevenbergMarquardtSolver();

        var result = solver.Solve(
            RosenbrockResiduals,
            initialGuess: [-1.2, 1.0],
            jacobian: RosenbrockJacobian);

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(1.0, 1e-6);
        result.Parameters[1].Should().BeApproximately(1.0, 1e-6);
        result.FinalCost.Should().BeLessThan(1e-10);
        result.Iterations.Should().BeLessThanOrEqualTo(30);
    }

    // ── Exponential fit (§3.1 AC #2) ──────────────────────────────

    [Fact]
    public void Solve_ExponentialFit_RecoversParametersWithinThreeSigma()
    {
        // Deterministic noisy data: y = 2·exp(0.3·x) + ε, with reproducible pseudo-noise.
        const double aTrue = 2.0;
        const double bTrue = 0.3;
        var rng = new DeterministicNormal(seed: 42);
        var xs = Enumerable.Range(0, 50).Select(i => i * 0.1).ToArray();
        var ys = xs.Select(x => aTrue * Math.Exp(bTrue * x) + 0.05 * rng.Next()).ToArray();

        var solver = new LevenbergMarquardtSolver();
        var result = solver.Solve(ExponentialResiduals(xs, ys), initialGuess: [1.0, 0.1]);

        result.Converged.Should().BeTrue();

        result.ParameterCovariance.Should().NotBeNull();
        var sigmaA = Math.Sqrt(result.ParameterCovariance![0, 0]);
        var sigmaB = Math.Sqrt(result.ParameterCovariance[1, 1]);

        Math.Abs(result.Parameters[0] - aTrue).Should().BeLessThan(3.0 * sigmaA);
        Math.Abs(result.Parameters[1] - bTrue).Should().BeLessThan(3.0 * sigmaB);
    }

    // ── Analytic vs finite-difference Jacobian (§3.1 AC #3) ──────

    [Fact]
    public void Solve_AnalyticAndFiniteDifferenceJacobian_AgreeTo1e6()
    {
        // Analytic Jacobian for Rosenbrock residuals.
        static double[,] AnalyticJacobian(double[] theta)
        {
            var x = theta[0];
            var j = new double[2, 2];
            j[0, 0] = -20.0 * x; // ∂r0/∂x
            j[0, 1] = 10.0;      // ∂r0/∂y
            j[1, 0] = -1.0;      // ∂r1/∂x
            j[1, 1] = 0.0;       // ∂r1/∂y
            return j;
        }

        var solver = new LevenbergMarquardtSolver();
        var fdResult = solver.Solve(RosenbrockResiduals, [-1.2, 1.0]);
        var analyticResult = solver.Solve(RosenbrockResiduals, [-1.2, 1.0], AnalyticJacobian);

        fdResult.Parameters[0].Should().BeApproximately(analyticResult.Parameters[0], 1e-6);
        fdResult.Parameters[1].Should().BeApproximately(analyticResult.Parameters[1], 1e-6);
    }

    // ── Bounds enforcement (§3.1 AC #4) ──────────────────────────

    [Fact]
    public void Solve_BoundsActive_WhenOptimumLiesOutsideBounds()
    {
        // Residual r(x) = x − 5. Unconstrained optimum x = 5. Bound [0, 3] → optimum at x = 3.
        static double[] Residual(double[] theta) => [theta[0] - 5.0];

        var solver = new LevenbergMarquardtSolver();
        var result = solver.Solve(
            Residual,
            initialGuess: [1.0],
            jacobian: null,
            lowerBounds: [0.0],
            upperBounds: [3.0]);

        result.BoundsActive.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(3.0, 1e-10);
    }

    // ── Parameter covariance (§3.1 AC #5) ─────────────────────────

    [Fact]
    public void Solve_ParameterCovariance_EqualsJtJInverseTimesSigmaSquared()
    {
        // Linear problem r_i = y_i − (a + b·x_i). J is constant.
        var xs = new[] { 0.0, 1.0, 2.0, 3.0, 4.0 };
        var ys = new[] { 1.1, 1.9, 3.1, 3.9, 5.05 };

        double[] Residuals(double[] theta) =>
            xs.Zip(ys, (x, y) => y - (theta[0] + theta[1] * x)).ToArray();

        var solver = new LevenbergMarquardtSolver();
        var result = solver.Solve(Residuals, [0.0, 0.0]);

        result.ParameterCovariance.Should().NotBeNull();

        // Expected σ̂² = RSS / (n − p) = 2·cost / (5 − 2).
        var rss = 2.0 * result.FinalCost;
        var sigmaSq = rss / (xs.Length - 2);

        // Reconstruct expected covariance: σ̂² · (JᵀJ)⁻¹ where J[i,0]=−1, J[i,1]=−x_i.
        // JᵀJ = [[n, Σx], [Σx, Σx²]].
        var n = (double)xs.Length;
        var sumX = xs.Sum();
        var sumX2 = xs.Select(x => x * x).Sum();
        var det = (n * sumX2) - (sumX * sumX);

        var expected00 = sigmaSq * sumX2 / det;
        var expected11 = sigmaSq * n / det;
        var expected01 = -sigmaSq * sumX / det;

        result.ParameterCovariance![0, 0].Should().BeApproximately(expected00, 1e-10);
        result.ParameterCovariance[1, 1].Should().BeApproximately(expected11, 1e-10);
        result.ParameterCovariance[0, 1].Should().BeApproximately(expected01, 1e-10);
        result.ParameterCovariance[1, 0].Should().BeApproximately(expected01, 1e-10);
    }

    // ── Input validation ──────────────────────────────────────────

    [Fact]
    public void Solve_NullResiduals_Throws()
    {
        var solver = new LevenbergMarquardtSolver();
        var act = () => solver.Solve(null!, [1.0]);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Solve_EmptyInitialGuess_Throws()
    {
        var solver = new LevenbergMarquardtSolver();
        var act = () => solver.Solve(theta => [0.0], []);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Solve_NonFiniteInitialGuess_Throws()
    {
        var solver = new LevenbergMarquardtSolver();
        var act = () => solver.Solve(theta => [0.0], [double.NaN]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Solve_BoundsLengthMismatch_Throws()
    {
        var solver = new LevenbergMarquardtSolver();
        var act = () => solver.Solve(
            theta => [theta[0] - 1.0],
            [0.0],
            jacobian: null,
            lowerBounds: [0.0, 0.0],
            upperBounds: null);
        act.Should().Throw<ArgumentException>();
    }

    // ── Max-iterations termination ───────────────────────────────

    [Fact]
    public void Solve_MaxIterationsReached_WhenBudgetTooSmall()
    {
        var solver = new LevenbergMarquardtSolver(maxIterations: 1);
        var result = solver.Solve(RosenbrockResiduals, [-1.2, 1.0]);

        result.Converged.Should().BeFalse();
        result.TerminationReason.Should().Be(LmTerminationReason.MaxIterationsReached);
    }

    // ── Deterministic output ─────────────────────────────────────

    [Fact]
    public void Solve_SameInputs_ProduceBitIdenticalOutputs()
    {
        var solver = new LevenbergMarquardtSolver();
        var result1 = solver.Solve(RosenbrockResiduals, [-1.2, 1.0]);
        var result2 = solver.Solve(RosenbrockResiduals, [-1.2, 1.0]);

        result1.Parameters.Should().Equal(result2.Parameters);
        result1.FinalCost.Should().Be(result2.FinalCost);
        result1.Iterations.Should().Be(result2.Iterations);
    }

    // ── Pooled vs non-pooled bit-identical results (§3.2 AC) ─────

    [Fact]
    public void Solve_PooledAndNonPooled_ProduceBitIdenticalResults_FiniteDifferenceJacobian()
    {
        // The pooled overload must produce results indistinguishable from the legacy
        // one-shot overload. Same arithmetic, same order of operations — only the
        // storage is different.
        var solver = new LevenbergMarquardtSolver();

        var legacy = solver.Solve(RosenbrockResiduals, [-1.2, 1.0]);

        var buffers = new LevenbergMarquardtBuffers(parameterCount: 2, residualCount: 2);
        var pooled = solver.Solve(RosenbrockResiduals, [-1.2, 1.0], buffers);

        pooled.Parameters.Should().Equal(legacy.Parameters);
        pooled.FinalCost.Should().Be(legacy.FinalCost);
        pooled.Iterations.Should().Be(legacy.Iterations);
        pooled.Converged.Should().Be(legacy.Converged);
        pooled.TerminationReason.Should().Be(legacy.TerminationReason);
        pooled.BoundsActive.Should().Be(legacy.BoundsActive);
        pooled.FinalResiduals.Should().Equal(legacy.FinalResiduals);
    }

    [Fact]
    public void Solve_PooledAndNonPooled_ProduceBitIdenticalResults_AnalyticJacobian()
    {
        static double[,] AnalyticJacobian(double[] theta)
        {
            var x = theta[0];
            return new double[,]
            {
                { -20.0 * x, 10.0 },
                { -1.0, 0.0 },
            };
        }

        var solver = new LevenbergMarquardtSolver();
        var legacy = solver.Solve(RosenbrockResiduals, [-1.2, 1.0], AnalyticJacobian);

        var buffers = new LevenbergMarquardtBuffers(parameterCount: 2, residualCount: 2);
        var pooled = solver.Solve(RosenbrockResiduals, [-1.2, 1.0], buffers, AnalyticJacobian);

        pooled.Parameters.Should().Equal(legacy.Parameters);
        pooled.FinalCost.Should().Be(legacy.FinalCost);
        pooled.Iterations.Should().Be(legacy.Iterations);
    }

    [Fact]
    public void Solve_PooledAndNonPooled_ProduceBitIdenticalResults_WithBounds()
    {
        static double[] Residual(double[] theta) => [theta[0] - 5.0];

        var solver = new LevenbergMarquardtSolver();
        var legacy = solver.Solve(
            Residual, initialGuess: [1.0], jacobian: null,
            lowerBounds: [0.0], upperBounds: [3.0]);

        var buffers = new LevenbergMarquardtBuffers(parameterCount: 1, residualCount: 1);
        var pooled = solver.Solve(
            Residual, initialGuess: [1.0], buffers, jacobian: null,
            lowerBounds: [0.0], upperBounds: [3.0]);

        pooled.Parameters.Should().Equal(legacy.Parameters);
        pooled.BoundsActive.Should().Be(legacy.BoundsActive);
        pooled.TerminationReason.Should().Be(legacy.TerminationReason);
    }

    [Fact]
    public void Solve_PooledOverload_ReusesBufferAcrossMultipleSolves_BitIdentical()
    {
        // Reusing the same pool across solves must not contaminate state. Two sequential
        // solves of the same problem with the same pool produce the same result as a
        // fresh pool for each solve.
        var solver = new LevenbergMarquardtSolver();
        var sharedPool = new LevenbergMarquardtBuffers(parameterCount: 2, residualCount: 2);

        var firstUsingShared = solver.Solve(RosenbrockResiduals, [-1.2, 1.0], sharedPool);
        var secondUsingShared = solver.Solve(RosenbrockResiduals, [-1.2, 1.0], sharedPool);

        secondUsingShared.Parameters.Should().Equal(firstUsingShared.Parameters);
        secondUsingShared.FinalCost.Should().Be(firstUsingShared.FinalCost);
        secondUsingShared.Iterations.Should().Be(firstUsingShared.Iterations);
    }

    // ── Helpers ───────────────────────────────────────────────────

    private static double[] RosenbrockResiduals(double[] theta)
    {
        // Rosenbrock as least-squares: residuals r0 = 10·(y − x²), r1 = 1 − x.
        // f(x, y) = ½(r0² + r1²) = 50·(y − x²)² + ½·(1 − x)² — minimized at (1, 1).
        var x = theta[0];
        var y = theta[1];
        return [10.0 * (y - x * x), 1.0 - x];
    }

    private static Func<double[], double[]> ExponentialResiduals(double[] xs, double[] ys)
    {
        return theta =>
        {
            var a = theta[0];
            var b = theta[1];
            var r = new double[xs.Length];
            for (var i = 0; i < xs.Length; i++)
            {
                r[i] = ys[i] - a * Math.Exp(b * xs[i]);
            }

            return r;
        };
    }

    /// <summary>
    /// Deterministic standard-normal generator (Box–Muller on a linear-congruential
    /// uniform stream). Used to keep the exponential-fit test reproducible across runs
    /// without introducing a dependency on the production RNG pipeline.
    /// </summary>
    private sealed class DeterministicNormal
    {
        private uint _state;
        private double? _pendingGaussian;

        public DeterministicNormal(uint seed)
        {
            _state = seed == 0 ? 1u : seed;
        }

        public double Next()
        {
            if (_pendingGaussian is { } cached)
            {
                _pendingGaussian = null;
                return cached;
            }

            double u1, u2;
            do
            {
                u1 = NextUniform();
                u2 = NextUniform();
            }
            while (u1 <= double.Epsilon);

            var radius = Math.Sqrt(-2.0 * Math.Log(u1));
            var angle = 2.0 * Math.PI * u2;

            _pendingGaussian = radius * Math.Sin(angle);
            return radius * Math.Cos(angle);
        }

        private double NextUniform()
        {
            // Numerical Recipes LCG — deterministic and adequate for test noise.
            _state = (1664525u * _state) + 1013904223u;
            return (_state & 0xFFFFFFu) / (double)0x1000000;
        }
    }
}
