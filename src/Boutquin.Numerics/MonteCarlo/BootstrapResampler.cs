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
/// Circular block bootstrap (Politis-Romano 1992). Pulls contiguous blocks
/// of fixed length from the source series at uniformly random starting
/// positions and concatenates them until the target length is filled.
/// Blocks that run past the end wrap around to the beginning, preserving
/// short-range autocorrelation without introducing edge effects.
/// </summary>
/// <typeparam name="T">Floating-point type for the series values.</typeparam>
/// <remarks>
/// <para>
/// Block bootstrap is appropriate for weakly dependent time series where a
/// plain IID bootstrap would destroy autocorrelation structure. A block
/// size around 21 trading days (~1 calendar month) is a common choice for
/// daily return series. See <c>MovingBlockBootstrapResampler</c> for the
/// non-wrapping variant and <c>StationaryBootstrapResampler</c> for
/// geometric-block-length resampling.
/// </para>
/// <para>
/// Uses an <see cref="IRandomSource"/> for deterministic reproducibility
/// across .NET runtime versions.
/// </para>
/// <para>
/// Tier A: Arithmetic resampling operations on floating-point types.
/// </para>
/// </remarks>
public sealed class BootstrapResampler<T>
    where T : IFloatingPoint<T>
{
    private readonly int _blockSize;
    private readonly IRandomSource _rng;

    /// <summary>
    /// Initializes the resampler with the given block size and random source.
    /// </summary>
    /// <param name="blockSize">Block length in source indices. Must be positive.</param>
    /// <param name="random">Random source. Supply a seeded <see cref="Pcg64RandomSource"/> for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="blockSize"/> is non-positive.</exception>
    public BootstrapResampler(int blockSize, IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(blockSize), blockSize, "Block size must be positive.");
        }

        _blockSize = blockSize;
        _rng = random;
    }

    /// <summary>
    /// Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.
    /// Equivalent to
    /// <c>new BootstrapResampler(blockSize, new Pcg64RandomSource((ulong)seed.GetValueOrDefault(entropy)))</c>.
    /// </summary>
    public static BootstrapResampler<T> FromSeed(int blockSize, int? seed = null)
        => new(
            blockSize,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(GetEntropySeed()));

    /// <summary>Gets the configured block size.</summary>
    public int BlockSize => _blockSize;

    /// <summary>
    /// Resamples a single series into a new path of the same length.
    /// </summary>
    /// <param name="source">Source series (length must be ≥ 1).</param>
    /// <returns>A resampled path of the same length as <paramref name="source"/>.</returns>
    /// <exception cref="ArgumentException"><paramref name="source"/> is empty.</exception>
    public T[] Resample(T[] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Length == 0)
        {
            throw new ArgumentException("Source must contain at least one element.", nameof(source));
        }

        var targetLength = source.Length;
        var result = new T[targetLength];
        FillResampledPath(source, result);
        return result;
    }

    /// <summary>
    /// Resamples two paired series (e.g., portfolio returns + benchmark
    /// returns) using the same randomly sampled block positions, preserving
    /// their time alignment.
    /// </summary>
    /// <param name="sourceA">First source series.</param>
    /// <param name="sourceB">Second source series (must have the same length).</param>
    /// <returns>Two resampled paths, same length, indexed identically.</returns>
    /// <exception cref="ArgumentException">Inputs have different lengths, or either is empty.</exception>
    public (T[] A, T[] B) ResamplePaired(T[] sourceA, T[] sourceB)
    {
        ArgumentNullException.ThrowIfNull(sourceA);
        ArgumentNullException.ThrowIfNull(sourceB);
        if (sourceA.Length != sourceB.Length)
        {
            throw new ArgumentException("Paired series must have the same length.", nameof(sourceB));
        }

        if (sourceA.Length == 0)
        {
            throw new ArgumentException("Source must contain at least one element.", nameof(sourceA));
        }

        var targetLength = sourceA.Length;
        var resultA = new T[targetLength];
        var resultB = new T[targetLength];
        FillResampledPathPaired(sourceA, sourceB, resultA, resultB);
        return (resultA, resultB);
    }

    private void FillResampledPath(T[] source, T[] destination)
    {
        var targetLength = destination.Length;
        var writeIdx = 0;
        while (writeIdx < targetLength)
        {
            var blockStart = _rng.NextInt(source.Length);
            var remaining = Math.Min(_blockSize, targetLength - writeIdx);
            for (var k = 0; k < remaining; k++)
            {
                var srcIdx = (blockStart + k) % source.Length;
                destination[writeIdx + k] = source[srcIdx];
            }

            writeIdx += remaining;
        }
    }

    private void FillResampledPathPaired(
        T[] sourceA,
        T[] sourceB,
        T[] destinationA,
        T[] destinationB)
    {
        var targetLength = destinationA.Length;
        var writeIdx = 0;
        while (writeIdx < targetLength)
        {
            var blockStart = _rng.NextInt(sourceA.Length);
            var remaining = Math.Min(_blockSize, targetLength - writeIdx);
            for (var k = 0; k < remaining; k++)
            {
                var srcIdx = (blockStart + k) % sourceA.Length;
                destinationA[writeIdx + k] = sourceA[srcIdx];
                destinationB[writeIdx + k] = sourceB[srcIdx];
            }

            writeIdx += remaining;
        }
    }

    internal static ulong GetEntropySeed()
    {
        var bytes = new byte[8];
        System.Security.Cryptography.RandomNumberGenerator.Fill(bytes);
        return BitConverter.ToUInt64(bytes, 0);
    }
}

/// <summary>
/// Circular block bootstrap (Politis-Romano 1992). Pulls contiguous blocks
/// of fixed length from the source series at uniformly random starting
/// positions and concatenates them until the target length is filled.
/// Blocks that run past the end wrap around to the beginning, preserving
/// short-range autocorrelation without introducing edge effects.
/// </summary>
/// <remarks>
/// <para>
/// Block bootstrap is appropriate for weakly dependent time series where a
/// plain IID bootstrap would destroy autocorrelation structure. A block
/// size around 21 trading days (~1 calendar month) is a common choice for
/// daily return series. See <c>MovingBlockBootstrapResampler</c> for the
/// non-wrapping variant and <c>StationaryBootstrapResampler</c> for
/// geometric-block-length resampling.
/// </para>
/// <para>
/// Uses an <see cref="IRandomSource"/> for deterministic reproducibility
/// across .NET runtime versions.
/// </para>
/// <para>
/// Tier A: Delegates to <see cref="BootstrapResampler{T}"/> with T = <see cref="decimal"/>.
/// </para>
/// </remarks>
public sealed class BootstrapResampler
{
    private readonly BootstrapResampler<decimal> _impl;

    /// <summary>
    /// Initializes the resampler with the given block size and random source.
    /// </summary>
    /// <param name="blockSize">Block length in source indices. Must be positive.</param>
    /// <param name="random">Random source. Supply a seeded <see cref="Pcg64RandomSource"/> for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="blockSize"/> is non-positive.</exception>
    public BootstrapResampler(int blockSize, IRandomSource random)
    {
        _impl = new BootstrapResampler<decimal>(blockSize, random);
    }

    /// <summary>
    /// Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.
    /// Equivalent to
    /// <c>new BootstrapResampler(blockSize, new Pcg64RandomSource((ulong)seed.GetValueOrDefault(entropy)))</c>.
    /// </summary>
    public static BootstrapResampler FromSeed(int blockSize, int? seed = null)
        => new(
            blockSize,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<decimal>.GetEntropySeed()));

    /// <summary>Gets the configured block size.</summary>
    public int BlockSize => _impl.BlockSize;

    /// <summary>
    /// Resamples a single series into a new path of the same length.
    /// </summary>
    /// <param name="source">Source series (length must be ≥ 1).</param>
    /// <returns>A resampled path of the same length as <paramref name="source"/>.</returns>
    /// <exception cref="ArgumentException"><paramref name="source"/> is empty.</exception>
    public decimal[] Resample(decimal[] source) => _impl.Resample(source);

    /// <summary>
    /// Resamples two paired series (e.g., portfolio returns + benchmark
    /// returns) using the same randomly sampled block positions, preserving
    /// their time alignment.
    /// </summary>
    /// <param name="sourceA">First source series.</param>
    /// <param name="sourceB">Second source series (must have the same length).</param>
    /// <returns>Two resampled paths, same length, indexed identically.</returns>
    /// <exception cref="ArgumentException">Inputs have different lengths, or either is empty.</exception>
    public (decimal[] A, decimal[] B) ResamplePaired(decimal[] sourceA, decimal[] sourceB)
        => _impl.ResamplePaired(sourceA, sourceB);
}
