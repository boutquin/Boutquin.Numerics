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
using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Bootstrap-flavor resample throughput + multi-statistic engine comparison
/// on a one-year daily returns series.
/// </summary>
[MemoryDiagnoser]
public class BootstrapBenchmarks
{
    private const int SeriesLength = 252;
    private decimal[] _series = null!;
    private BootstrapResampler _iid = null!;
    private StationaryBootstrapResampler _stationary = null!;
    private MovingBlockBootstrapResampler _mbb = null!;
    private WildBootstrapResampler _wild = null!;
    private BootstrapMonteCarloEngine _engine = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new System.Random(31);
        _series = new decimal[SeriesLength];
        for (var i = 0; i < SeriesLength; i++)
        {
            _series[i] = (decimal)(rng.NextDouble() * 0.02 - 0.01);
        }

        var source1 = new Pcg64RandomSource(1UL);
        var source2 = new Pcg64RandomSource(2UL);
        var source3 = new Pcg64RandomSource(3UL);
        var source4 = new Pcg64RandomSource(4UL);
        var source5 = new Pcg64RandomSource(5UL);

        _iid = new BootstrapResampler(1, source1);
        _stationary = new StationaryBootstrapResampler(5.0, source2);
        _mbb = new MovingBlockBootstrapResampler(10, source3);
        _wild = new WildBootstrapResampler(WildBootstrapWeights.Rademacher, source4);
        _engine = new BootstrapMonteCarloEngine(1000, source5);
    }

    [Benchmark(Baseline = true)]
    public decimal[] IidBlock1() => _iid.Resample(_series);

    [Benchmark]
    public decimal[] StationaryGeometric() => _stationary.Resample(_series);

    [Benchmark]
    public decimal[] MovingBlock() => _mbb.Resample(_series);

    [Benchmark]
    public decimal[] Wild() => _wild.Resample(_series);

    [Benchmark]
    public BootstrapMonteCarloResult EngineSingleStat()
        => _engine.Run(_series, static s =>
        {
            decimal sum = 0m;
            for (var i = 0; i < s.Length; i++)
            {
                sum += s[i];
            }

            return sum / s.Length;
        });

    [Benchmark]
    public MultiStatisticBootstrapResult EngineMultiStat()
        => _engine.RunMulti(_series, 4, static (s, w) =>
        {
            decimal sum = 0m, sumSq = 0m, min = decimal.MaxValue, max = decimal.MinValue;
            for (var i = 0; i < s.Length; i++)
            {
                var v = s[i];
                sum += v;
                sumSq += v * v;
                if (v < min)
                {
                    min = v;
                }

                if (v > max)
                {
                    max = v;
                }
            }

            var mean = sum / s.Length;
            w[0] = mean;
            w[1] = sumSq / s.Length - mean * mean;
            w[2] = min;
            w[3] = max;
        });
}
