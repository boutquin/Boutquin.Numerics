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

using BenchmarkDotNet.Attributes;
using Boutquin.Numerics.Solvers;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Demonstrates the allocation-reduction claim of <see cref="LevenbergMarquardtBuffers"/>
/// from <c>specs/feature-lm-zero-alloc-iteration.md</c> §3.1 on a representative
/// hot-path workload (6 parameters, 24 residuals — matching the Lanczos3 NIST StRD shape).
/// </summary>
/// <remarks>
/// <para>
/// Three benchmarks compare per-solve allocation across equivalent calls:
/// </para>
/// <list type="bullet">
/// <item><description>
/// <b>PoolFreeSolve</b> — the legacy overload. Allocates a private
/// <see cref="LevenbergMarquardtBuffers"/> instance per call.
/// </description></item>
/// <item><description>
/// <b>PooledSolve</b> — the new overload taking a caller-owned pool. The pool is
/// constructed once in <c>[GlobalSetup]</c> and reused across every invocation.
/// </description></item>
/// <item><description>
/// <b>PooledSolveAllocatingResidual</b> — the pooled overload with a residual callback
/// that returns a fresh array per call (the common-case consumer pattern). Shows the
/// residual-callback allocation bound the solver cannot eliminate.
/// </description></item>
/// </list>
/// <para>
/// Use the BenchmarkDotNet <c>[MemoryDiagnoser]</c> output to compare the <c>Allocated</c>
/// column: the pooled overload with a zero-allocating residual callback should show only
/// the <see cref="MultivariateSolverResult"/> record and its owned arrays plus the one-shot
/// covariance scratch allocated at convergence.
/// </para>
/// </remarks>
[MemoryDiagnoser]
public class LmAllocationBenchmark
{
    private const int ParameterCount = 6;
    private const int ResidualCount = 24;

    private LevenbergMarquardtSolver _solver = null!;
    private LevenbergMarquardtBuffers _pool = null!;
    private double[] _initialGuess = null!;
    private double[] _xs = null!;
    private double[] _ys = null!;

    // Shared residual buffer for the zero-allocation callback path. The solver copies
    // the returned array into its pool storage on every call, so returning the same
    // instance repeatedly is safe.
    private double[] _sharedResidualBuffer = null!;

    // Delegate fields cache method-group conversions so the hot loop does not
    // allocate a fresh Func<> per invocation (a 64-byte/call cost otherwise).
    private Func<double[], double[]> _nonAllocatingResiduals = null!;
    private Func<double[], double[]> _allocatingResiduals = null!;

    [GlobalSetup]
    public void Setup()
    {
        _solver = new LevenbergMarquardtSolver(maxIterations: 100);
        _pool = new LevenbergMarquardtBuffers(ParameterCount, ResidualCount);
        _initialGuess = new double[ParameterCount];

        // Deterministic polynomial fit — 6-parameter polynomial in x ∈ [-1, 1] with seeded
        // noise. Converges in a small number of accepted iterations on every run.
        _xs = new double[ResidualCount];
        _ys = new double[ResidualCount];
        _sharedResidualBuffer = new double[ResidualCount];

        var state = 13u;
        for (var i = 0; i < ResidualCount; i++)
        {
            var x = -1.0 + (2.0 * i / (ResidualCount - 1));
            _xs[i] = x;

            var y = 0.0;
            var pow = 1.0;
            for (var k = 0; k < ParameterCount; k++)
            {
                y += Math.Pow(0.5, k) * pow;
                pow *= x;
            }

            state = (1664525u * state) + 1013904223u;
            var noise = (((state & 0xFFFFFFu) / (double)0x1000000) - 0.5) * 0.01;
            _ys[i] = y + noise;
        }

        _nonAllocatingResiduals = NonAllocatingResiduals;
        _allocatingResiduals = AllocatingResiduals;
    }

    /// <summary>
    /// Baseline: legacy overload builds a private buffer pool on every call. Representative
    /// of callers who fit a single model once and do not participate in the hot-path
    /// allocation contract.
    /// </summary>
    [Benchmark(Baseline = true)]
    public MultivariateSolverResult PoolFreeSolve() =>
        _solver.Solve(_nonAllocatingResiduals, _initialGuess);

    /// <summary>
    /// Pooled overload + non-allocating residual callback. This is the tight allocation
    /// bound — only the <see cref="MultivariateSolverResult"/> record and its owned arrays
    /// plus the one-shot covariance scratch survive per solve.
    /// </summary>
    [Benchmark]
    public MultivariateSolverResult PooledSolve() =>
        _solver.Solve(_nonAllocatingResiduals, _initialGuess, _pool);

    /// <summary>
    /// Pooled overload + residual callback that returns a fresh array per call. Shows the
    /// residual-side allocation bound the solver cannot eliminate when callers write their
    /// residual function in the obvious way.
    /// </summary>
    [Benchmark]
    public MultivariateSolverResult PooledSolveAllocatingResidual() =>
        _solver.Solve(_allocatingResiduals, _initialGuess, _pool);

    private double[] NonAllocatingResiduals(double[] theta)
    {
        for (var i = 0; i < ResidualCount; i++)
        {
            var model = 0.0;
            var pow = 1.0;
            for (var k = 0; k < ParameterCount; k++)
            {
                model += theta[k] * pow;
                pow *= _xs[i];
            }

            _sharedResidualBuffer[i] = _ys[i] - model;
        }

        return _sharedResidualBuffer;
    }

    private double[] AllocatingResiduals(double[] theta)
    {
        var r = new double[ResidualCount];
        for (var i = 0; i < ResidualCount; i++)
        {
            var model = 0.0;
            var pow = 1.0;
            for (var k = 0; k < ParameterCount; k++)
            {
                model += theta[k] * pow;
                pow *= _xs[i];
            }

            r[i] = _ys[i] - model;
        }

        return r;
    }
}
