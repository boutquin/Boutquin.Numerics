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
using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Cholesky vs pivoted Cholesky vs Jacobi eigendecomposition throughput on
/// synthetic SPD matrices of increasing size.
/// </summary>
[MemoryDiagnoser]
public class LinearAlgebraBenchmarks
{
    [Params(10, 50, 200)]
    public int N { get; set; }

    private decimal[,] _matrix = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Build an SPD matrix: A · Aᵀ + n·I.
        var rng = new System.Random(1234);
        var a = new decimal[N, N];
        for (var i = 0; i < N; i++)
        {
            for (var j = 0; j < N; j++)
            {
                a[i, j] = (decimal)(rng.NextDouble() - 0.5);
            }
        }

        _matrix = new decimal[N, N];
        for (var i = 0; i < N; i++)
        {
            for (var j = 0; j < N; j++)
            {
                var sum = 0m;
                for (var k = 0; k < N; k++)
                {
                    sum += a[i, k] * a[j, k];
                }

                _matrix[i, j] = sum + (i == j ? N : 0);
            }
        }
    }

    [Benchmark(Baseline = true)]
    public decimal[,] Cholesky() => CholeskyDecomposition.Decompose(_matrix);

    [Benchmark]
    public PivotedCholeskyResult PivotedCholesky() =>
        CholeskyDecomposition.DecomposePivoted(_matrix);

    [Benchmark]
    public EigenResult Jacobi() => JacobiEigenDecomposition.Decompose(_matrix);
}
