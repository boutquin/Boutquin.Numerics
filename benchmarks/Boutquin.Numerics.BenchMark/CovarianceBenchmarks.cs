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
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Throughput comparison across covariance estimators on all four combinations
/// of T = 252 or 1260 observations and N = 10 or 50 variables. The independent
/// BenchmarkDotNet parameters form a Cartesian product.
/// </summary>
[MemoryDiagnoser]
public class CovarianceBenchmarks
{
    [Params(252, 1260)]
    public int T { get; set; }

    [Params(10, 50)]
    public int N { get; set; }

    private decimal[,] _returns = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(4242);
        _returns = new decimal[T, N];
        for (var i = 0; i < T; i++)
        {
            for (var j = 0; j < N; j++)
            {
                _returns[i, j] = (decimal)(rng.NextDouble() * 0.02 - 0.01);
            }
        }
    }

    [Benchmark(Baseline = true)]
    public decimal[,] Sample() => new SampleCovarianceEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] LedoitWolfLinear() => new LedoitWolfShrinkageEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] LedoitWolfCC() => new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] LedoitWolfFM() => new LedoitWolfSingleFactorEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] QIS() => new QuadraticInverseShrinkageEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] OAS() => new OracleApproximatingShrinkageEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] Denoised() => new DenoisedCovarianceEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] TracyWidom() => new TracyWidomDenoisedCovarianceEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] Nercome() => new NercomeCovarianceEstimator<decimal>().Estimate(_returns);

    [Benchmark]
    public decimal[,] Poet() => new PoetCovarianceEstimator<decimal>(1, 0.5m).Estimate(_returns);
}
