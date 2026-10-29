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

using System.Runtime.InteropServices;

using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class Arena414074R1MonteCarloTests
{
    // Expected block lengths come from a 700-digit decimal evaluation of the selected
    // author listing: common rows, separately centred lag columns, denominator N-M-1.
    // The interior pattern alone gives bandwidth 13 and the capped optimum 22.
    private const double WeightedOutlierOracle = 20.96593115367117;

    [Theory]
    [InlineData(1, 1e20, WeightedOutlierOracle)]
    [InlineData(0, 1e300, 22.0)]
    public void EstimateDouble_RemoteOutlier_DoesNotEraseRetainedColumnSpread(int position, double outlier, double expected)
    {
        // A midpoint taken from the whole series rounds every other observation to the
        // same value. x[1] lies only in the lag-12 column of the second step; x[0] lies
        // only in the zero-weight lag-13 column, so the oracle keeps the capped 22.
        var series = Pattern(1.0);
        series[position] = outlier;

        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(expected, 1e-12);
    }

    [Theory]
    [InlineData(false, WeightedOutlierOracle)]
    [InlineData(true, 4.0)]
    public void EstimateDouble_OppositeExtremes_DoNotUnderflowRetainedColumns(bool shortMemory, double expected)
    {
        // The midpoint is exactly zero, so translation is exact. Scaling every row by
        // the full-series extreme underflows the squares of the retained columns. The
        // periodic interior keeps M at the search horizon; the short-memory interior
        // has M = 2, which the first-step correlations must reproduce.
        var series = shortMemory ? ShortMemory() : Pattern(1.0);
        series[0] = -1e308;
        series[1] = 1e308;

        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void EstimateDouble_ExtremeOnlyInZeroWeightColumn_DoesNotSetTheCommonScale()
    {
        // x[0] lies only in the kernel's zero-weight lag-M column. Its scale exceeds the
        // retained columns' by more than 2^1023, so neither the alignment exponent nor
        // the weighted sums may include it; the oracle keeps the capped 22.
        var series = Pattern(1e-20);
        series[0] = 1e308;

        PolitisWhiteBlockLength.EstimateDouble(series).Should().BeApproximately(22.0, 1e-12);
    }

    [Fact]
    public void Resample_MaximumLengthWrappingBlock_DoesNotOverflowCircularIndex()
    {
        // Start at the last element; the 64-element block wraps after one step, and
        // start + k exceeds Int32.MaxValue from k = 58. The second draw stops the run,
        // so only the first pages of either array are touched.
        var source = new Half[Array.MaxLength];
        var random = new WrapThenStop(Array.MaxLength - 1);
        var resampler = new BootstrapResampler<Half>(64, random);

        var act = () => resampler.Resample(source);

        act.Should().Throw<StopResampling>();
        random.Draws.Should().Be(2);
    }

    [Fact]
    public void ResamplePaired_MaximumLengthWrappingBlock_DoesNotOverflowCircularIndex()
    {
        var source = new Half[Array.MaxLength];
        var random = new WrapThenStop(Array.MaxLength - 1);
        var resampler = new BootstrapResampler<Half>(64, random);

        var act = () => resampler.ResamplePaired(source, source);

        act.Should().Throw<StopResampling>();
        random.Draws.Should().Be(2);
    }

    [Fact]
    public void GaussianNext_OppositeSignAffineForm_StaysFiniteForNFloat()
    {
        // Uniforms (1/2, 25/32) give u = 0, v = 9/16 and normal = 1.5170552328818643.
        // The exact value Max * (normal - 1) is finite; the double FMA path returns it.
        var sampler = new GaussianSampler<NFloat>(new Uniforms(.5, .78125));

        var value = sampler.Next(-NFloat.MaxValue, NFloat.MaxValue);

        ((double)value).Should().Be(9.295066424963632E+307);
    }

    private static double[] Pattern(double scale)
        => Enumerable.Range(0, 64).Select(i => scale * ((i % 7) - 3)).ToArray();

    // Integers in [-3, 3] from the ANSI C linear congruential generator, seed 1.
    private static double[] ShortMemory()
    {
        var series = new double[64];
        var state = 1L;
        for (var i = 0; i < series.Length; i++)
        {
            state = (1103515245L * state + 12345L) % (1L << 31);
            series[i] = ((state >> 16) % 7) - 3;
        }

        return series;
    }

    private sealed class StopResampling : Exception
    {
    }

    private sealed class WrapThenStop(int start) : IRandomSource
    {
        public int Draws { get; private set; }

        public ulong NextULong() => throw new NotSupportedException();

        public int NextInt(int upperExclusive)
        {
            if (Draws++ > 0)
            {
                throw new StopResampling();
            }

            return start;
        }
    }

    private sealed class Uniforms(params double[] values) : IRandomSource
    {
        private int _reads;

        public ulong NextULong() => throw new NotSupportedException();

        public double NextDouble() => values[_reads++];
    }
}
