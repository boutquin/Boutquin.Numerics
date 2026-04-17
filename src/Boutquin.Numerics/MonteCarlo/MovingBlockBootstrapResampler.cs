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

using System.Numerics;

using Boutquin.Numerics.Random;

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Moving Block Bootstrap (Künsch 1989, Liu-Singh 1992). Block start
/// positions are drawn uniformly from <c>[0, T − blockSize]</c> — blocks
/// never wrap, so trending series are not contaminated by edge wrap-around.
/// </summary>
/// <typeparam name="T">Floating-point type for the series values.</typeparam>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics, 17(3), 1217–1241.</description></item>
/// <item><description>Liu, R. Y. &amp; Singh, K. (1992). "Moving Blocks Jackknife and Bootstrap Capture Weak Dependence." In <em>Exploring the Limits of Bootstrap</em>, 225–248.</description></item>
/// </list>
/// </para>
/// <para>
/// Compared to the circular block bootstrap (<see cref="BootstrapResampler{T}"/>),
/// MBB introduces a small bias from edge effects but avoids spurious
/// correlations across the wrap point. For non-stationary or trending data
/// this is the safer default.
/// </para>
/// <para>
/// Tier A: Arithmetic resampling operations on floating-point types.
/// </para>
/// </remarks>
public sealed class MovingBlockBootstrapResampler<T>
    where T : IFloatingPoint<T>
{
    private readonly int _blockSize;
    private readonly IRandomSource _rng;

    /// <summary>Initializes the resampler with block size and random source.</summary>
    public MovingBlockBootstrapResampler(int blockSize, IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "Block size must be positive.");
        }

        _blockSize = blockSize;
        _rng = random;
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static MovingBlockBootstrapResampler<T> FromSeed(int blockSize, int? seed = null)
        => new(
            blockSize,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<T>.GetEntropySeed()));

    /// <summary>Block size used for resampling.</summary>
    public int BlockSize => _blockSize;

    /// <summary>Resamples a series using non-wrapping blocks.</summary>
    public T[] Resample(T[] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Length < _blockSize)
        {
            throw new ArgumentException(
                $"Source length ({source.Length}) must be at least block size ({_blockSize}).",
                nameof(source));
        }

        var n = source.Length;
        var result = new T[n];
        var maxStart = n - _blockSize + 1;
        var write = 0;
        while (write < n)
        {
            var start = _rng.NextInt(maxStart);
            var remaining = Math.Min(_blockSize, n - write);
            for (var k = 0; k < remaining; k++)
            {
                result[write + k] = source[start + k];
            }

            write += remaining;
        }

        return result;
    }
}

/// <summary>
/// Moving Block Bootstrap (Künsch 1989, Liu-Singh 1992). Block start
/// positions are drawn uniformly from <c>[0, T − blockSize]</c> — blocks
/// never wrap, so trending series are not contaminated by edge wrap-around.
/// </summary>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics, 17(3), 1217–1241.</description></item>
/// <item><description>Liu, R. Y. &amp; Singh, K. (1992). "Moving Blocks Jackknife and Bootstrap Capture Weak Dependence." In <em>Exploring the Limits of Bootstrap</em>, 225–248.</description></item>
/// </list>
/// </para>
/// <para>
/// Compared to the circular block bootstrap (<see cref="BootstrapResampler"/>),
/// MBB introduces a small bias from edge effects but avoids spurious
/// correlations across the wrap point. For non-stationary or trending data
/// this is the safer default.
/// </para>
/// <para>
/// Tier A: Delegates to <see cref="MovingBlockBootstrapResampler{T}"/> with T = <see cref="decimal"/>.
/// </para>
/// </remarks>
public sealed class MovingBlockBootstrapResampler
{
    private readonly MovingBlockBootstrapResampler<decimal> _impl;

    /// <summary>Initializes the resampler with block size and random source.</summary>
    public MovingBlockBootstrapResampler(int blockSize, IRandomSource random)
    {
        _impl = new MovingBlockBootstrapResampler<decimal>(blockSize, random);
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static MovingBlockBootstrapResampler FromSeed(int blockSize, int? seed = null)
        => new(
            blockSize,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<decimal>.GetEntropySeed()));

    /// <summary>Block size used for resampling.</summary>
    public int BlockSize => _impl.BlockSize;

    /// <summary>Resamples a series using non-wrapping blocks.</summary>
    public decimal[] Resample(decimal[] source) => _impl.Resample(source);
}
