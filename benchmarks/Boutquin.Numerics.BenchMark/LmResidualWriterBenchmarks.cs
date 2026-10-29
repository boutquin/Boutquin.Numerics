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
/// Extends <see cref="LmAllocationBenchmark"/> to include the span-callback overloads.
/// </summary>
/// <remarks>
/// <para>
/// Three-way comparison: legacy <c>Func&lt;T[], T[]&gt;</c> (allocates a pool per call);
/// pooled <c>Func&lt;T[], T[]&gt;</c> (caller-owned pool, reused across calls); pooled
/// <c>LmResidualWriter&lt;T&gt;</c> (zero per-iteration allocation).
/// </para>
/// <para>
/// Acceptance criterion: <c>PooledResidualWriter.Gen0 = 0</c> and
/// <c>Allocated ≤ MultivariateSolverResult size + convergence scratch</c>.
/// </para>
/// </remarks>
[MemoryDiagnoser]
public class LmResidualWriterBenchmarks
{
    private const int ParameterCount = 6;
    private const int ResidualCount = 24;

    private LevenbergMarquardtSolver<double> _solver = null!;
    private LevenbergMarquardtBuffers<double> _pool = null!;
    private double[] _initialGuess = null!;
    private double[] _xs = null!;
    private double[] _ys = null!;
    private double[] _sharedResidualBuffer = null!;
    private Func<double[], double[]> _legacyCallback = null!;
    private LmResidualWriter<double> _spanWriter = null!;

    [GlobalSetup]
    public void Setup()
    {
        _solver = new LevenbergMarquardtSolver<double>(
            maxIterations: 100,
            functionTolerance: 1e-10,
            parameterTolerance: 1e-10,
            gradientTolerance: 1e-10,
            initialDamping: 1e-3);
        _pool = new LevenbergMarquardtBuffers<double>(ParameterCount, ResidualCount);
        _initialGuess = new double[ParameterCount];
        _sharedResidualBuffer = new double[ResidualCount];

        _xs = new double[ResidualCount];
        _ys = new double[ResidualCount];

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
            _ys[i] = y + (((state & 0xFFFFFFu) / (double)0x1000000) - 0.5) * 0.01;
        }

        _legacyCallback = Residuals;
        _spanWriter = ResidualSpan;
    }

    /// <summary>
    /// Baseline: legacy pool-free overload. Allocates a private
    /// <see cref="LevenbergMarquardtBuffers{T}"/> instance per call.
    /// </summary>
    [Benchmark(Baseline = true)]
    public MultivariateSolverResult<double> LegacyFuncCallback()
        => _solver.Solve(_legacyCallback, _initialGuess);

    /// <summary>
    /// Pooled overload with a non-allocating residual callback. The pool is constructed
    /// once in <c>[GlobalSetup]</c> and reused across every invocation.
    /// </summary>
    [Benchmark]
    public MultivariateSolverResult<double> PooledFuncCallback()
        => _solver.Solve(_legacyCallback, _initialGuess, _pool);

    /// <summary>
    /// Pooled span-callback overload. Zero per-iteration allocation — the writer fills the
    /// residual <see cref="Span{T}"/> in-place; no array is returned or boxed.
    /// </summary>
    [Benchmark]
    public MultivariateSolverResult<double> PooledResidualWriter()
        => _solver.Solve(_spanWriter, _initialGuess.AsSpan(), _pool);

    private double[] Residuals(double[] theta)
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

    private void ResidualSpan(ReadOnlySpan<double> theta, Span<double> residuals)
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

            residuals[i] = _ys[i] - model;
        }
    }
}
