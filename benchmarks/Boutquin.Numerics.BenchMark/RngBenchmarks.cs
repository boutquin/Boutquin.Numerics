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
using Boutquin.Numerics.Random;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Random-number and Gaussian sampler throughput. Compares the framework
/// <see cref="System.Random"/> against PCG64 and xoshiro256**, and measures the
/// amortized cost of a 10,000-sample Gaussian batch via Marsaglia polar.
/// </summary>
[MemoryDiagnoser]
public class RngBenchmarks
{
    private const int Iterations = 10_000;

    private System.Random _system = null!;
    private IRandomSource _pcg = null!;
    private IRandomSource _xoshiro = null!;
    private GaussianSampler _gauss = null!;

    [GlobalSetup]
    public void Setup()
    {
        _system = new System.Random(42);
        _pcg = new Pcg64RandomSource(42UL);
        _xoshiro = new Xoshiro256StarStarRandomSource(42UL);
        _gauss = new GaussianSampler(new Pcg64RandomSource(42UL));
    }

    [Benchmark(Baseline = true)]
    public double SystemRandom()
    {
        var acc = 0.0;
        for (var i = 0; i < Iterations; i++)
        {
            acc += _system.NextDouble();
        }

        return acc;
    }

    [Benchmark]
    public double Pcg64()
    {
        var acc = 0.0;
        for (var i = 0; i < Iterations; i++)
        {
            acc += _pcg.NextDouble();
        }

        return acc;
    }

    [Benchmark]
    public double Xoshiro256StarStar()
    {
        var acc = 0.0;
        for (var i = 0; i < Iterations; i++)
        {
            acc += _xoshiro.NextDouble();
        }

        return acc;
    }

    [Benchmark]
    public double GaussianBatch()
    {
        var acc = 0.0;
        for (var i = 0; i < Iterations; i++)
        {
            acc += _gauss.Next();
        }

        return acc;
    }
}
