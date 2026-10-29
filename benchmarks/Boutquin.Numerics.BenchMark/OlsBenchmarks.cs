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
/// QR-factorisation throughput for <see cref="OrdinaryLeastSquares{T}"/> across six
/// representative matrix shapes and both <c>double</c> and <c>decimal</c> types.
/// </summary>
/// <remarks>
/// <para>
/// Sizes: the Cartesian product of <c>Rows = {50, 500, 2000}</c> and
/// <c>Cols = {5, 20}</c>. <c>FitDouble</c> is the baseline;
/// <c>FitDecimal</c> quantifies the cost of 28-digit decimal arithmetic on the same
/// QR factorisation.
/// </para>
/// <para>
/// Design matrix columns are filled with values drawn from a seeded uniform distribution
/// on <c>[-1, 1]</c>; the response vector uses known true coefficients <c>{1, 2, …, Cols}</c>
/// plus small IID noise, giving the solver a well-conditioned problem at every size.
/// </para>
/// </remarks>
[MemoryDiagnoser]
public class OlsBenchmarks
{
    [Params(50, 500, 2000)]
    public int Rows { get; set; }

    [Params(5, 20)]
    public int Cols { get; set; }

    private double[,] _xDouble = null!;
    private double[] _yDouble = null!;
    private decimal[,] _xDecimal = null!;
    private decimal[] _yDecimal = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(1729);
        _xDouble = new double[Rows, Cols];
        _yDouble = new double[Rows];
        _xDecimal = new decimal[Rows, Cols];
        _yDecimal = new decimal[Rows];

        for (var i = 0; i < Rows; i++)
        {
            var y = 0.0;
            for (var j = 0; j < Cols; j++)
            {
                var v = rng.NextDouble() * 2.0 - 1.0;
                _xDouble[i, j] = v;
                _xDecimal[i, j] = (decimal)v;
                y += (j + 1) * v;
            }

            var noise = (rng.NextDouble() - 0.5) * 0.1;
            _yDouble[i] = y + noise;
            _yDecimal[i] = (decimal)_yDouble[i];
        }
    }

    [Benchmark(Baseline = true)]
    public OlsResult<double> FitDouble() => OrdinaryLeastSquares<double>.Fit(_xDouble, _yDouble);

    [Benchmark]
    public OlsResult<decimal> FitDecimal() => OrdinaryLeastSquares<decimal>.Fit(_xDecimal, _yDecimal);
}
