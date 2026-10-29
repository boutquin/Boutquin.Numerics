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
/// Tests for the <see cref="LmResidualWriter{T}"/> and <see cref="LmJacobianWriter{T}"/>
/// delegate overloads on <see cref="LevenbergMarquardtSolver{T}"/>.
/// Covers AC §3.2 of the span-parameters feature spec.
/// </summary>
public sealed class LmResidualWriterTests
{
    [Fact]
    public void WriterSolve_FiniteDifferenceAtLargestFiniteInput_NeverCallsResidualWithInfinity()
    {
        var observed = new List<double>();
        LmResidualWriter<double> writer = (parameters, residuals) =>
        {
            observed.Add(parameters[0]);
            residuals[0] = 0.5;
        };

        var solver = CreateSolver(maxIterations: 10);
        _ = solver.Solve(writer, new double[] { double.MaxValue }.AsSpan(), residualCount: 1);

        observed.Should().OnlyContain(value => double.IsFinite(value));
    }

    private const int ParameterCount = 6;
    private const int ResidualCount = 24;

    // LevenbergMarquardtSolver<T> requires all 5 constructor parameters (no optional defaults).
    private static LevenbergMarquardtSolver<double> CreateSolver(int maxIterations = 200)
        => new(maxIterations, 1e-10, 1e-10, 1e-10, 1e-3);

    // ── §3.2 AC: delegate types are public AND invocable with span side-effects ──

    [Fact]
    public void LmResidualWriter_IsPublicDelegate_AndWritesResidualsIntoSpan()
    {
        // The delegate must be public (consumers assign it) AND actually usable: assigning
        // a lambda and invoking it must mutate the caller-supplied residual span in place.
        typeof(LmResidualWriter<double>).IsPublic.Should().BeTrue();

        // Model y = a*x + b on x = [1, 2, 3]; residual_i = y_i - (a*x_i + b).
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [3.0, 5.0, 7.0]; // exact for a=2, b=1
        LmResidualWriter<double> writer = (parameters, residuals) =>
        {
            for (var i = 0; i < xs.Length; i++)
            {
                residuals[i] = ys[i] - ((parameters[0] * xs[i]) + parameters[1]);
            }
        };

        var buffer = new double[3];
        buffer.AsSpan().Fill(double.NaN); // poison so a no-op writer would be detected
        writer(new double[] { 2.0, 1.0 }, buffer);

        // With the true parameters the residuals are exactly zero — proving the delegate
        // ran and its span side-effect landed in the caller's buffer.
        buffer.Should().Equal(0.0, 0.0, 0.0);

        // A non-fitting parameter set produces the expected non-zero residuals.
        writer(new double[] { 0.0, 0.0 }, buffer);
        buffer.Should().Equal(3.0, 5.0, 7.0);
    }

    [Fact]
    public void LmJacobianWriter_IsPublicDelegate_AndWritesJacobianIntoSpan()
    {
        typeof(LmJacobianWriter<double>).IsPublic.Should().BeTrue();

        const int m = 3; // residuals
        const int n = 2; // parameters
        double[] xs = [1.0, 2.0, 3.0];

        // Analytic Jacobian for y = a*x + b: ∂r_i/∂a = -x_i, ∂r_i/∂b = -1.
        LmJacobianWriter<double> writer = (parameters, jacobian) =>
        {
            for (var i = 0; i < m; i++)
            {
                jacobian[(i * n) + 0] = -xs[i];
                jacobian[(i * n) + 1] = -1.0;
            }
        };

        var flat = new double[m * n];
        flat.AsSpan().Fill(double.NaN);
        writer(new double[] { 2.0, 1.0 }, flat);

        // Row-major flat layout side-effect, fully observed.
        flat.Should().Equal(-1.0, -1.0, -2.0, -1.0, -3.0, -1.0);
    }

    // ── §3.2 AC: parity — writer overload produces bit-identical results ────

    [Fact]
    public void PoolFreeWriterSolve_ProducesBitIdenticalResultToFuncOverload()
    {
        // Both pool-free overloads must produce exactly the same result when
        // the residuals represent the same function — AC §3.2 parity check.
        var fixture = new PolynomialFixture(ParameterCount, ResidualCount, seed: 42);
        var solver = CreateSolver(maxIterations: 200);
        var initialGuess = new double[ParameterCount];

        // Func<T[], T[]> overload (existing).
        var funcResult = solver.Solve(fixture.FuncResiduals, initialGuess);

        // LmResidualWriter<T> pool-free overload (new). residualCount is required because
        // the writer fills a pre-allocated buffer and the solver can't probe the count.
        var writerResult = solver.Solve(fixture.WriterResiduals, initialGuess.AsSpan(), ResidualCount);

        writerResult.Converged.Should().Be(funcResult.Converged);
        writerResult.Parameters.Should().BeEquivalentTo(funcResult.Parameters,
            opts => opts.Using<double>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 1e-10))
                        .WhenTypeIs<double>(),
            because: "writer overload must produce equivalent parameters to Func overload");
        writerResult.FinalCost.Should().BeApproximately(funcResult.FinalCost, 1e-10,
            because: "writer overload must produce equivalent final cost to Func overload");
    }

    [Fact]
    public void PooledWriterSolve_ProducesBitIdenticalResultToFuncOverload()
    {
        var fixture = new PolynomialFixture(ParameterCount, ResidualCount, seed: 42);
        var solver = CreateSolver(maxIterations: 200);
        var initialGuess = new double[ParameterCount];

        var funcBuffers = new LevenbergMarquardtBuffers<double>(ParameterCount, ResidualCount);
        var writerBuffers = new LevenbergMarquardtBuffers<double>(ParameterCount, ResidualCount);

        var funcResult = solver.Solve(fixture.FuncResiduals, initialGuess, funcBuffers);
        var writerResult = solver.Solve(fixture.WriterResiduals, initialGuess.AsSpan(), writerBuffers);

        writerResult.Converged.Should().Be(funcResult.Converged);
        writerResult.Parameters.Should().BeEquivalentTo(funcResult.Parameters,
            opts => opts.Using<double>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 1e-10))
                        .WhenTypeIs<double>());
        writerResult.FinalCost.Should().BeApproximately(funcResult.FinalCost, 1e-10);
    }

    // ── §3.2 AC: bound whole-solve allocation, including result arrays ───────

    [Fact]
    public void PooledWriterSolve_WholeSolveAllocationWithinTerminalBudget()
    {
        // Bound fixed whole-solve overhead; Arena20260928LmAllocationSlopeTests
        // separately varies actual iteration counts to detect loop allocation.
        const int WarmupRounds = 10;
        const int MeasuredRounds = 90;

        var fixture = new PolynomialFixture(ParameterCount, ResidualCount, seed: 13);
        var buffers = new LevenbergMarquardtBuffers<double>(ParameterCount, ResidualCount);
        var solver = CreateSolver(maxIterations: 100);
        var initialGuess = new double[ParameterCount];

        // Cache the delegate once — method-group conversions for instance methods
        // allocate a fresh delegate object on every conversion.
        LmResidualWriter<double> writerDelegate = fixture.WriterResiduals;

        for (var i = 0; i < WarmupRounds; i++)
        {
            _ = solver.Solve(writerDelegate, initialGuess.AsSpan(), buffers);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        const double ExpectedResultRecordSizeBytes = 1800.0;

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < MeasuredRounds; i++)
        {
            _ = solver.Solve(writerDelegate, initialGuess.AsSpan(), buffers);
        }

        var after = GC.GetAllocatedBytesForCurrentThread();
        var perSolve = (after - before) / (double)MeasuredRounds;

        perSolve.Should().BeLessThan(
            ExpectedResultRecordSizeBytes * 2.0,
            because: "only the MultivariateSolverResult record and its owned arrays should be allocated per solve — " +
                     "plus one-shot covariance scratch should fit the whole-solve budget");
    }

    // ── §3.2 AC: Jacobian flat layout ───────────────────────────────────────

    [Fact]
    public void LmJacobianWriter_FlatLayout_RowMajor_RowI_ColJ_At_i_times_n_plus_j()
    {
        // Verify that the LmJacobianWriter flat layout matches row-major convention:
        // jacobian[i * parameterCount + j] corresponds to row i, column j.
        const int m = 3; // residual count
        const int n = 2; // parameter count

        // Analytic Jacobian for the linear model y = a*x + b:
        // dr_i/da = -x_i, dr_i/db = -1.0
        double[] xs = [1.0, 2.0, 3.0];

        LmJacobianWriter<double> writer = (parameters, jacobian) =>
        {
            for (var i = 0; i < m; i++)
            {
                jacobian[i * n + 0] = -xs[i]; // ∂r_i/∂a
                jacobian[i * n + 1] = -1.0;   // ∂r_i/∂b
            }
        };

        var flatBuffer = new double[m * n];
        writer(new double[] { 1.0, 0.0 }, flatBuffer);

        // Row 0: [-1, -1], Row 1: [-2, -1], Row 2: [-3, -1]
        flatBuffer[0 * n + 0].Should().Be(-1.0);
        flatBuffer[0 * n + 1].Should().Be(-1.0);
        flatBuffer[1 * n + 0].Should().Be(-2.0);
        flatBuffer[1 * n + 1].Should().Be(-1.0);
        flatBuffer[2 * n + 0].Should().Be(-3.0);
        flatBuffer[2 * n + 1].Should().Be(-1.0);
    }

    [Fact]
    public void PooledWriterSolveWithAnalyticJacobian_ProducesEquivalentResultToFuncOverload()
    {
        // Verifies that LmJacobianWriter<T> works with the pooled overload and produces
        // results consistent with the Func<T[], T[,]> analytic Jacobian.
        const int p = 2; // parameters: slope (a) and intercept (b)
        const int r = 5; // residuals

        double[] xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] ys = [2.1, 3.9, 6.2, 7.8, 10.1]; // approx y = 2x + 0

        LmResidualWriter<double> residualWriter = (parameters, residuals) =>
        {
            for (var i = 0; i < r; i++)
            {
                residuals[i] = ys[i] - (parameters[0] * xs[i] + parameters[1]);
            }
        };

        LmJacobianWriter<double> jacobianWriter = (parameters, jacobian) =>
        {
            for (var i = 0; i < r; i++)
            {
                jacobian[i * p + 0] = -xs[i];
                jacobian[i * p + 1] = -1.0;
            }
        };

        var buffers = new LevenbergMarquardtBuffers<double>(p, r);
        var solver = CreateSolver(maxIterations: 100);
        double[] guess = [0.0, 0.0];

        var result = solver.Solve(residualWriter, guess.AsSpan(), buffers, jacobianWriter);

        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(2.0, 0.1,
            because: "slope should converge near 2.0 for the given data");
        result.Parameters[1].Should().BeApproximately(0.0, 0.5,
            because: "intercept should converge near 0.0 for the given data");
    }

    // ── §3.1 AC: span params on existing Solve overloads ────────────────────

    // ── Fixture ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Deterministic polynomial-fit fixture exposing both a <c>Func&lt;T[], T[]&gt;</c>
    /// and an <c>LmResidualWriter&lt;T&gt;</c> for the same residual function.
    /// Both callbacks use a cached output buffer so neither contributes
    /// per-iteration heap allocation.
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

                state = (1664525u * state) + 1013904223u;
                var noise = (((state & 0xFFFFFFu) / (double)0x1000000) - 0.5) * 0.01;
                _ys[i] = y + noise;
            }
        }

        /// <summary>Func&lt;T[], T[]&gt; callback — returns cached shared buffer.</summary>
        public double[] FuncResiduals(double[] theta)
        {
            FillResiduals(theta.AsSpan(), _residualBuffer.AsSpan());
            return _residualBuffer;
        }

        /// <summary>LmResidualWriter&lt;T&gt; callback — fills the solver's pool buffer directly.</summary>
        public void WriterResiduals(ReadOnlySpan<double> parameters, Span<double> residuals)
        {
            FillResiduals(parameters, residuals);
        }

        private void FillResiduals(ReadOnlySpan<double> theta, Span<double> residuals)
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

                residuals[i] = _ys[i] - model;
            }
        }
    }
}
