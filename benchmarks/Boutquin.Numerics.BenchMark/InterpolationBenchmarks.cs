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
using Boutquin.Numerics.Interpolation;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Build + 10,000-query throughput for linear, monotone-cubic, and natural
/// cubic-spline interpolators across 10 / 100 / 1000 knot grids.
/// </summary>
[MemoryDiagnoser]
public class InterpolationBenchmarks
{
    [Params(10, 100, 1000)]
    public int KnotCount { get; set; }

    private double[] _xs = null!;
    private double[] _ys = null!;
    private double[] _queries = null!;
    private CubicSplineInterpolator _spline = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(2026);
        _xs = new double[KnotCount];
        _ys = new double[KnotCount];
        for (var i = 0; i < KnotCount; i++)
        {
            _xs[i] = i;
            _ys[i] = Math.Sin(i * 0.1) + rng.NextDouble() * 0.01;
        }

        _queries = new double[10_000];
        for (var i = 0; i < _queries.Length; i++)
        {
            _queries[i] = rng.NextDouble() * (KnotCount - 1);
        }

        _spline = new CubicSplineInterpolator(_xs, _ys);
    }

    [Benchmark(Baseline = true)]
    public double Linear()
    {
        var xs = _xs.AsSpan();
        var ys = _ys.AsSpan();
        var acc = 0.0;
        for (var i = 0; i < _queries.Length; i++)
        {
            acc += LinearInterpolator.Instance.Interpolate(_queries[i], xs, ys);
        }

        return acc;
    }

    [Benchmark]
    public double MonotoneCubic()
    {
        var xs = _xs.AsSpan();
        var ys = _ys.AsSpan();
        var acc = 0.0;
        for (var i = 0; i < _queries.Length; i++)
        {
            acc += MonotoneCubicInterpolator.Instance.Interpolate(_queries[i], xs, ys);
        }

        return acc;
    }

    [Benchmark]
    public double CubicSplineQueries()
    {
        var acc = 0.0;
        for (var i = 0; i < _queries.Length; i++)
        {
            acc += _spline.Interpolate(_queries[i]);
        }

        return acc;
    }

    [Benchmark]
    public CubicSplineInterpolator CubicSplineBuild() => new(_xs, _ys);
}
