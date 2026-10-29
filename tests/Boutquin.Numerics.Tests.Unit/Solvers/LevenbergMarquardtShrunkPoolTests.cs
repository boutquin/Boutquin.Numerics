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
/// Regression tests for the pooled Levenberg–Marquardt overloads' pre-loop bootstrap
/// on a grow-only pool that has been shrunk via <see cref="LevenbergMarquardtBuffers{T}.Reset(int, int)"/>.
/// After <c>new LevenbergMarquardtBuffers&lt;T&gt;(5, 5)</c> followed by <c>Reset(2, 2)</c> the
/// physical arrays retain their high-water capacity (length 5) while the logical
/// <see cref="LevenbergMarquardtBuffers{T}.ParameterCount"/> / <see cref="LevenbergMarquardtBuffers{T}.ResidualCount"/>
/// are 2. The pre-loop bootstrap must operate on the logical dimensions, not the physical ones.
/// </summary>
public sealed class LevenbergMarquardtShrunkPoolTests
{
    // Known 2-parameter linear problem: y = a + b·x with a = 0.3, b = 0.5 on x = {0, 1}.
    //   r0 = 0.3 − a           (target y0 = 0.3 at x = 0)
    //   r1 = 0.8 − a − b       (target y1 = 0.8 at x = 1)
    // Exact minimizer a = 0.3, b = 0.5, both inside [0, 1].
    private static LevenbergMarquardtSolver<double> NewSolver() =>
        new(maxIterations: 100, functionTolerance: 1e-12,
            parameterTolerance: 1e-12, gradientTolerance: 1e-12, initialDamping: 1e-3);

    [Fact]
    public void PooledWriter_ShrunkPool_WithBounds_DoesNotThrow()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(5, 5);
        buffers.Reset(2, 2);
        var solver = NewSolver();

        static void Writer(ReadOnlySpan<double> p, Span<double> r)
        {
            r[0] = 0.3 - p[0];
            r[1] = 0.8 - p[0] - p[1];
        }

        var lower = new[] { 0.0, 0.0 };
        var upper = new[] { 1.0, 1.0 };

        var act = () => solver.Solve(Writer, new[] { 0.1, 0.1 }, buffers, jacobian: null, lower, upper);

        act.Should().NotThrow(
            "the pooled writer bootstrap must clip only the logical parameter count, " +
            "not the physical (high-water) array length");
    }

    [Fact]
    public void PooledFunc_ShrunkPool_WithBounds_DoesNotThrow()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(5, 5);
        buffers.Reset(2, 2);
        var solver = NewSolver();

        static double[] Residuals(double[] t) => new[] { 0.3 - t[0], 0.8 - t[0] - t[1] };

        var lower = new[] { 0.0, 0.0 };
        var upper = new[] { 1.0, 1.0 };

        var act = () => solver.Solve(Residuals, new[] { 0.1, 0.1 }, buffers, jacobian: null, lower, upper);

        act.Should().NotThrow(
            "the pooled Func bootstrap must clip only the logical parameter count, " +
            "not the physical (high-water) array length");
    }

    [Fact]
    public void PooledWriter_ShrunkPool_NoBounds_WriterSeesLogicalSpanLengths()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(5, 5);
        buffers.Reset(2, 2);
        var solver = NewSolver();

        var firstParamLen = -1;
        var firstResidLen = -1;

        void Writer(ReadOnlySpan<double> p, Span<double> r)
        {
            if (firstParamLen < 0)
            {
                firstParamLen = p.Length;
                firstResidLen = r.Length;
            }

            r[0] = 0.3 - p[0];
            r[1] = 0.8 - p[0] - p[1];
        }

        // No bounds: ClipToBounds is a no-op, isolating the span-length contract of the
        // pooled writer bootstrap from the bounds-clipping crash face.
        var result = solver.Solve(Writer, new[] { 0.1, 0.1 }, buffers);

        firstParamLen.Should().Be(2,
            "the writer's parameter span must be sliced to the logical parameter count");
        firstResidLen.Should().Be(2,
            "the writer's residual span must be sliced to the logical residual count");

        result.Parameters[0].Should().BeApproximately(0.3, 1e-9);
        result.Parameters[1].Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void PooledWriter_ShrunkPool_AnalyticJacobian_NoExtraAllocationVsContiguousPool()
    {
        // Same 2-param linear problem, analytic-Jacobian writer path. The contiguous pool
        // (fresh 2×2) writes the Jacobian in place with zero per-iteration allocation; the
        // shrunk pool (5×5 → Reset(2,2)) must match it. Pre-fix, the shrunk-pool analytic
        // branch allocates a fresh T[m·n] scratch on every Jacobian evaluation, so its
        // per-solve allocation exceeds the contiguous pool's by 32·(#Jacobian evals) bytes.
        static void Writer(ReadOnlySpan<double> p, Span<double> r)
        {
            r[0] = 0.3 - p[0];
            r[1] = 0.8 - p[0] - p[1];
        }

        // Row-major flat Jacobian, stride = parameterCount = 2:
        //   ∂r0/∂a = -1, ∂r0/∂b =  0
        //   ∂r1/∂a = -1, ∂r1/∂b = -1
        static void Jacobian(ReadOnlySpan<double> p, Span<double> j)
        {
            j[0] = -1.0; j[1] = 0.0;
            j[2] = -1.0; j[3] = -1.0;
        }

        var solver = NewSolver();
        var guess = new[] { 0.1, 0.1 };

        double MeasurePerSolve(Func<LevenbergMarquardtBuffers<double>> poolFactory)
        {
            var buffers = poolFactory();

            for (var i = 0; i < 20; i++)
            {
                _ = solver.Solve(Writer, guess, buffers, Jacobian);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            const int Reps = 200;
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < Reps; i++)
            {
                _ = solver.Solve(Writer, guess, buffers, Jacobian);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            return (after - before) / (double)Reps;
        }

        var contiguousPerSolve = MeasurePerSolve(() => new LevenbergMarquardtBuffers<double>(2, 2));
        var shrunkPerSolve = MeasurePerSolve(() =>
        {
            var b = new LevenbergMarquardtBuffers<double>(5, 5);
            b.Reset(2, 2);
            return b;
        });

        (shrunkPerSolve - contiguousPerSolve).Should().BeLessThan(
            16.0,
            "the shrunk-pool analytic-Jacobian path must reuse a cached scratch buffer, " +
            "adding no per-solve allocation beyond the contiguous pool");
    }

    // ---------------------------------------------------------------------------------------
    // Analytic-Jacobian parameter-span slicing on a shrunk grow-only pool.
    // (4, 20) → Reset(2, 5): parameter capacity 4 > logical 2, so the Jacobian slab width
    // (4) exceeds the logical parameter count (2) and the shrunk-branch scatter path runs.
    // The analytic-Jacobian callback must receive a parameter argument sliced to the LOGICAL
    // parameter count (2), not the physical pool capacity (4). A contract-conforming writer
    // that derives its column stride from parameters.Length otherwise overflows the m·n
    // output span (writer path → IndexOutOfRangeException) or fails the shape check
    // (Func path → InvalidOperationException).
    // Known 2-param linear problem y = a + b·x over 5 points, exact minimizer a = 0.3, b = 0.5.
    // ---------------------------------------------------------------------------------------

    private static readonly double[] s_x5 = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    private static readonly double[] s_y5 = { 0.3, 0.8, 1.3, 1.8, 2.3 };

    private static void LinearResidualWriter(ReadOnlySpan<double> p, Span<double> r)
    {
        for (var i = 0; i < 5; i++)
        {
            r[i] = s_y5[i] - (p[0] + (p[1] * s_x5[i]));
        }
    }

    [Fact]
    public void PooledWriter_ShrunkPool_AnalyticJacobianWriter_SeesLogicalParameterLength()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(4, 20);
        buffers.Reset(2, 5);
        var solver = NewSolver();

        var firstJacobianParamLen = -1;

        // Hardcode the logical stride 2 so this test isolates the observed parameter length
        // from the overflow crash face — it records what the writer sees, nothing more.
        void Jacobian(ReadOnlySpan<double> p, Span<double> j)
        {
            if (firstJacobianParamLen < 0)
            {
                firstJacobianParamLen = p.Length;
            }

            for (var i = 0; i < 5; i++)
            {
                j[(i * 2) + 0] = -1.0;
                j[(i * 2) + 1] = -s_x5[i];
            }
        }

        _ = solver.Solve(LinearResidualWriter, new[] { 0.1, 0.1 }, buffers, Jacobian);

        firstJacobianParamLen.Should().Be(2,
            "the analytic-Jacobian writer's parameter span must be sliced to the logical " +
            "parameter count, not the physical (high-water) pool capacity");
    }

    [Fact]
    public void PooledWriter_ShrunkPool_ConformingAnalyticJacobianWriter_SolvesCorrectly()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(4, 20);
        buffers.Reset(2, 5);
        var solver = NewSolver();

        // Contract-conforming writer: derives the row stride from parameters.Length.
        // On the physical-length span (4) this indexes j[i*4+1] up to 17 on a length-10
        // (m·n = 5·2) span → IndexOutOfRangeException.
        static void Jacobian(ReadOnlySpan<double> p, Span<double> j)
        {
            var n = p.Length;
            for (var i = 0; i < 5; i++)
            {
                j[(i * n) + 0] = -1.0;
                j[(i * n) + 1] = -s_x5[i];
            }
        }

        var result = solver.Solve(LinearResidualWriter, new[] { 0.1, 0.1 }, buffers, Jacobian);

        result.Parameters[0].Should().BeApproximately(0.3, 1e-9);
        result.Parameters[1].Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void PooledFunc_ShrunkPool_ConformingAnalyticJacobian_SolvesCorrectly()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(4, 20);
        buffers.Reset(2, 5);
        var solver = NewSolver();

        static double[] Residuals(double[] p)
        {
            var r = new double[5];
            for (var i = 0; i < 5; i++)
            {
                r[i] = s_y5[i] - (p[0] + (p[1] * s_x5[i]));
            }

            return r;
        }

        // Contract-conforming analytic Jacobian: sizes its column count from parameters.Length.
        // On the physical-length array (4) it returns a [5, 4] matrix, which fails the solver's
        // shape check (expected [5, 2]) → InvalidOperationException.
        static double[,] Jacobian(double[] p)
        {
            var n = p.Length;
            var jac = new double[5, n];
            for (var i = 0; i < 5; i++)
            {
                jac[i, 0] = -1.0;
                jac[i, 1] = -s_x5[i];
            }

            return jac;
        }

        var result = solver.Solve(Residuals, new[] { 0.1, 0.1 }, buffers, Jacobian);

        result.Parameters[0].Should().BeApproximately(0.3, 1e-9);
        result.Parameters[1].Should().BeApproximately(0.5, 1e-9);
    }

    [Fact]
    public void PooledFunc_ShrunkPool_ResidualSeesLogicalParameterLength()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(4, 20);
        buffers.Reset(2, 5);
        var solver = NewSolver();

        var firstResidualParamLen = -1;

        double[] Residuals(double[] p)
        {
            if (firstResidualParamLen < 0)
            {
                firstResidualParamLen = p.Length;
            }

            var r = new double[5];
            for (var i = 0; i < 5; i++)
            {
                r[i] = s_y5[i] - (p[0] + (p[1] * s_x5[i]));
            }

            return r;
        }

        _ = solver.Solve(Residuals, new[] { 0.1, 0.1 }, buffers);

        firstResidualParamLen.Should().Be(2,
            "the residual Func's parameter array must be sliced to the logical parameter " +
            "count, not the physical (high-water) pool capacity; a contract-conforming " +
            "residual that iterates over parameters.Length would otherwise read a stale tail");
    }

    // ---------------------------------------------------------------------------------------
    // DEFAULT finite-difference Jacobian path (no analytic Jacobian supplied) on a shrunk
    // grow-only pool. FiniteDifferenceJacobian<T>.EvaluateInto reuses buffers.PerturbedParameters
    // — a high-water physical array (length 4) whose tail [n..capacity) retains STALE values
    // from an earlier larger solve after Reset(2, 5). Array.Copy refreshes only the first
    // parameterCount (2) elements, so a contract-conforming residual that iterates
    // parameters.Length reads the stale tail during Jacobian evaluation. For a residual that
    // is nonlinear in the parameters this corrupts the finite-difference Jacobian columns
    // (the stale tail shifts the operating point away from the main length-n residual), and
    // the solver converges with Converged == true to GARBAGE. The parameter argument handed to
    // the user residual on the finite-difference path must be sliced to the LOGICAL parameter
    // count (2). RED before fix: max observed parameters length is 4 and convergence lands on
    // garbage; GREEN after: length 2, converges to the correct minimizer.
    //
    // Model: y = exp(a + b·x) with a = -0.2, b = 0.35 over 5 points. A conforming residual
    // iterates parameters.Length: expo = Σ_k p[k]·x^k, r = y − exp(expo). At logical n = 2 this
    // is exp(a + b·x); reading a stale x^2/x^3 tail blows the exponent up and wrecks the FD
    // Jacobian.
    // ---------------------------------------------------------------------------------------

    private static readonly double[] s_expX = { 0.0, 0.5, 1.0, 1.5, 2.0 };
    private static readonly double[] s_expY =
        { Math.Exp(-0.2), Math.Exp(-0.2 + (0.35 * 0.5)), Math.Exp(-0.2 + (0.35 * 1.0)),
          Math.Exp(-0.2 + (0.35 * 1.5)), Math.Exp(-0.2 + (0.35 * 2.0)) };

    // 6-point cubic used only to prime the pool's finite-difference scratch with a nonzero
    // stale tail via a size-4 solve before the pool is shrunk to size 2.
    private static readonly double[] s_primeX = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    private static readonly double[] s_primeY =
        { CubicModel(0.0), CubicModel(1.0), CubicModel(2.0),
          CubicModel(3.0), CubicModel(4.0), CubicModel(5.0) };

    private static double CubicModel(double x) => 1.0 + (2.0 * x) + (3.0 * x * x) + (0.5 * x * x * x);

    private static double[] PrimeResiduals(double[] p)
    {
        var r = new double[6];
        for (var i = 0; i < 6; i++)
        {
            var model = 0.0;
            for (var k = 0; k < p.Length; k++)
            {
                model += p[k] * Math.Pow(s_primeX[i], k);
            }

            r[i] = s_primeY[i] - model;
        }

        return r;
    }

    private static void PrimeResidualWriter(ReadOnlySpan<double> p, Span<double> r)
    {
        for (var i = 0; i < 6; i++)
        {
            var model = 0.0;
            for (var k = 0; k < p.Length; k++)
            {
                model += p[k] * Math.Pow(s_primeX[i], k);
            }

            r[i] = s_primeY[i] - model;
        }
    }

    [Fact]
    public void PooledFunc_ShrunkPool_DefaultFiniteDifference_ResidualSeesLogicalLength_AndSolvesCorrectly()
    {
        // Prime the pool's finite-difference scratch with a nonzero stale tail via a size-4 solve.
        var buffers = new LevenbergMarquardtBuffers<double>(4, 20);
        buffers.Reset(4, 6);
        _ = NewSolver().Solve(PrimeResiduals, new[] { 0.5, 0.5, 0.5, 0.5 }, buffers);

        // Shrink logically to 2 params / 5 residuals; the physical scratch stays length 4.
        buffers.Reset(2, 5);
        var solver = NewSolver();

        var maxObservedParamLen = 0;

        double[] Residuals(double[] p)
        {
            if (p.Length > maxObservedParamLen)
            {
                maxObservedParamLen = p.Length;
            }

            var r = new double[5];
            for (var i = 0; i < 5; i++)
            {
                var expo = 0.0;
                for (var k = 0; k < p.Length; k++)
                {
                    expo += p[k] * Math.Pow(s_expX[i], k);
                }

                r[i] = s_expY[i] - Math.Exp(expo);
            }

            return r;
        }

        // Default finite-difference path: no analytic Jacobian supplied.
        var result = solver.Solve(Residuals, new[] { 0.0, 0.0 }, buffers);

        maxObservedParamLen.Should().Be(2,
            "every residual invocation on the default finite-difference path — including the " +
            "internal perturbation evaluations — must receive a parameter array sliced to the " +
            "logical parameter count (2), not the physical high-water pool capacity (4)");
        result.Parameters[0].Should().BeApproximately(-0.2, 1e-6,
            "reading a stale finite-difference tail corrupts the Jacobian and converges to garbage");
        result.Parameters[1].Should().BeApproximately(0.35, 1e-6);
    }

    [Fact]
    public void PooledWriter_ShrunkPool_DefaultFiniteDifference_ResidualSeesLogicalLength_AndSolvesCorrectly()
    {
        var buffers = new LevenbergMarquardtBuffers<double>(4, 20);
        buffers.Reset(4, 6);
        _ = NewSolver().Solve(PrimeResidualWriter, new[] { 0.5, 0.5, 0.5, 0.5 }, buffers);

        buffers.Reset(2, 5);
        var solver = NewSolver();

        var maxObservedParamLen = 0;

        void Writer(ReadOnlySpan<double> p, Span<double> r)
        {
            if (p.Length > maxObservedParamLen)
            {
                maxObservedParamLen = p.Length;
            }

            for (var i = 0; i < 5; i++)
            {
                var expo = 0.0;
                for (var k = 0; k < p.Length; k++)
                {
                    expo += p[k] * Math.Pow(s_expX[i], k);
                }

                r[i] = s_expY[i] - Math.Exp(expo);
            }
        }

        // Default finite-difference path: no analytic Jacobian supplied.
        var result = solver.Solve(Writer, new[] { 0.0, 0.0 }, buffers);

        maxObservedParamLen.Should().Be(2,
            "every writer invocation on the default finite-difference path — including the " +
            "internal perturbation evaluations — must receive a parameter span sliced to the " +
            "logical parameter count (2), not the physical high-water pool capacity (4)");
        result.Parameters[0].Should().BeApproximately(-0.2, 1e-6,
            "reading a stale finite-difference tail corrupts the Jacobian and converges to garbage");
        result.Parameters[1].Should().BeApproximately(0.35, 1e-6);
    }
}
