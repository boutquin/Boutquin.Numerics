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
/// Active-set QP solver throughput for minimum-variance and mean-variance portfolio
/// construction across three basket sizes and both <c>double</c> and <c>decimal</c> types.
/// </summary>
/// <remarks>
/// <para>
/// <c>N=20</c> — small basket; <c>N=50</c> — mid-basket where inner Cholesky cost dominates
/// active-set iteration; <c>N=100</c> — realistic institutional basket. N is capped at 100:
/// at N=200 a full active-set run at <c>T=decimal</c> may exceed 10 s, making CI impractical.
/// </para>
/// <para>
/// The covariance matrix is constructed as <c>Cov = L·Lᵀ + N·I</c> where <c>L</c> is a
/// seeded random lower-triangular matrix, guaranteeing a positive-definite input at every N.
/// Weight bounds are <c>[0, 0.15]</c> (15 % cap), matching a realistic long-only constraint.
/// The MinVar vs MeanVar split isolates whether the mean-return inner products add measurable
/// cost over the pure minimum-variance solve.
/// </para>
/// </remarks>
[MemoryDiagnoser]
public class QpSolverBenchmarks
{
    [Params(20, 50, 100)]
    public int N { get; set; }

    private double[,] _covDouble = null!;
    private decimal[,] _covDecimal = null!;
    private double[] _muDouble = null!;
    private decimal[] _muDecimal = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(3141);

        // Build PSD covariance: Cov = L·Lᵀ + N·I, where L is a random lower-triangular matrix.
        var l = new double[N, N];
        for (var i = 0; i < N; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                l[i, j] = rng.NextDouble();
            }
        }

        _covDouble = new double[N, N];
        for (var i = 0; i < N; i++)
        {
            for (var j = 0; j < N; j++)
            {
                var s = 0.0;
                for (var k = 0; k <= Math.Min(i, j); k++)
                {
                    s += l[i, k] * l[j, k];
                }

                _covDouble[i, j] = s + (i == j ? N : 0);
            }
        }

        _muDouble = new double[N];
        for (var i = 0; i < N; i++)
        {
            _muDouble[i] = rng.NextDouble() * 0.001;
        }

        _covDecimal = new decimal[N, N];
        _muDecimal = new decimal[N];
        for (var i = 0; i < N; i++)
        {
            for (var j = 0; j < N; j++)
            {
                _covDecimal[i, j] = (decimal)_covDouble[i, j];
            }

            _muDecimal[i] = (decimal)_muDouble[i];
        }
    }

    [Benchmark(Baseline = true)]
    public double[] MinVarDouble()
        => ActiveSetQpSolver<double>.SolveMinVariance(_covDouble, minWeight: 0.0, maxWeight: 0.15);

    [Benchmark]
    public decimal[] MinVarDecimal()
        => ActiveSetQpSolver<decimal>.SolveMinVariance(_covDecimal, minWeight: 0m, maxWeight: 0.15m);

    [Benchmark]
    public double[] MeanVarDouble()
        => ActiveSetQpSolver<double>.SolveMeanVariance(_covDouble, _muDouble,
               riskAversion: 2.0, minWeight: 0.0, maxWeight: 0.15);

    [Benchmark]
    public decimal[] MeanVarDecimal()
        => ActiveSetQpSolver<decimal>.SolveMeanVariance(_covDecimal, _muDecimal,
               riskAversion: 2m, minWeight: 0m, maxWeight: 0.15m);
}
