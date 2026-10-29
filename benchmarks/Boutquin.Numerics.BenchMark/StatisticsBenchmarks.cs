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
/// Throughput and allocation for four statistics not covered by existing benchmark classes,
/// at two realistic series lengths and across both <c>double</c> and <c>decimal</c> types.
/// </summary>
/// <remarks>
/// <para>
/// <c>N=252</c> — one trading year of daily observations; <c>N=1260</c> — five years.
/// Returns are drawn from a seeded uniform distribution on <c>[-2%, +2%]</c>.
/// </para>
/// <para>
/// Welford and Newey-West process the generated series in the chosen working type;
/// Newey-West uses arithmetic autocovariance sums without a transcendental tail.
/// Haircut uses fixed scalar inputs and does not accumulate the generated series.
/// Relative latency is measured by the benchmark results, not assumed here.
/// </para>
/// </remarks>
[MemoryDiagnoser]
public class StatisticsBenchmarks
{
    [Params(252, 1260)]
    public int N { get; set; }

    private double[] _returnsDouble = null!;
    private decimal[] _returnsDecimal = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(2718);
        _returnsDouble = new double[N];
        _returnsDecimal = new decimal[N];
        for (var i = 0; i < N; i++)
        {
            var r = (rng.NextDouble() - 0.5) * 0.04;
            _returnsDouble[i] = r;
            _returnsDecimal[i] = (decimal)r;
        }
    }

    [Benchmark(Baseline = true)]
    public (double Mean, double Variance) WelfordDouble()
        => WelfordMoments<double>.Compute(_returnsDouble.AsSpan());

    [Benchmark]
    public (decimal Mean, decimal Variance) WelfordDecimal()
        => WelfordMoments<decimal>.Compute(_returnsDecimal.AsSpan());

    [Benchmark]
    public double PearsonDouble()
        => PearsonCorrelation<double>.Compute(_returnsDouble.AsSpan()[..(N / 2)],
                                              _returnsDouble.AsSpan()[(N / 2)..]);

    [Benchmark]
    public decimal PearsonDecimal()
        => PearsonCorrelation<decimal>.Compute(_returnsDecimal.AsSpan()[..(N / 2)],
                                               _returnsDecimal.AsSpan()[(N / 2)..]);

    [Benchmark]
    public double NeweyWestDouble()
        => NeweyWestVariance<double>.MeanVariance(_returnsDouble, lags: 5);

    [Benchmark]
    public decimal NeweyWestDecimal()
        => NeweyWestVariance<decimal>.MeanVariance(_returnsDecimal, lags: 5);

    [Benchmark]
    public HaircutSharpeResult<double> HaircutDouble()
        => HaircutSharpe<double>.Compute(
               observedSharpe: 1.5,
               numTrials: 100,
               backTestYears: 3.0);

    [Benchmark]
    public HaircutSharpeResult<decimal> HaircutDecimal()
        => HaircutSharpe<decimal>.Compute(
               observedSharpe: 1.5m,
               numTrials: 100,
               backTestYears: 3.0m);
}
