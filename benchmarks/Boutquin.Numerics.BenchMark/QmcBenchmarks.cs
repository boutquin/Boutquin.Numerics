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

namespace Boutquin.Numerics.BenchMark;

/// <summary>
/// Sobol vs Halton low-discrepancy sequence throughput in moderate
/// dimensions (4 / 8 / 16). Both generators emit one point per <c>Next</c>.
/// </summary>
[MemoryDiagnoser]
public class QmcBenchmarks
{
    private const int Points = 10_000;

    [Params(4, 8, 16)]
    public int Dimension { get; set; }

    private SobolSequence _sobol = null!;
    private HaltonSequence _halton = null!;

    [IterationSetup]
    public void IterationSetup()
    {
        _sobol = new SobolSequence(Dimension);
        _halton = new HaltonSequence(Dimension);
    }

    [Benchmark(Baseline = true)]
    public double Sobol()
    {
        var acc = 0.0;
        for (var i = 0; i < Points; i++)
        {
            var p = _sobol.Next();
            for (var d = 0; d < Dimension; d++)
            {
                acc += p[d];
            }
        }

        return acc;
    }

    [Benchmark]
    public double Halton()
    {
        var acc = 0.0;
        for (var i = 0; i < Points; i++)
        {
            var p = _halton.Next();
            for (var d = 0; d < Dimension; d++)
            {
                acc += p[d];
            }
        }

        return acc;
    }
}
