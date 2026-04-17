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
/// Stationary bootstrap (Politis-Romano 1994). Block lengths are drawn from
/// a geometric distribution with mean <c>1/p</c>, yielding strictly
/// stationary resampled paths — unlike the fixed-block variants where
/// stationarity holds only approximately. At each step, with probability
/// <c>p</c> the next index restarts at a uniformly random source position;
/// with probability <c>1 − p</c> it advances one position from the current.
/// </summary>
/// <typeparam name="T">Floating-point type for the series values.</typeparam>
/// <remarks>
/// <para>
/// Reference: Politis, D. N. &amp; Romano, J. P. (1994). "The Stationary
/// Bootstrap." Journal of the American Statistical Association, 89(428),
/// 1303–1313.
/// </para>
/// <para>
/// The <see cref="MeanBlockLength"/> parameter controls the expected block
/// size (= 1/p). For automatic selection, use
/// <see cref="PolitisWhiteBlockLength.Estimate{T}(T[])"/> which
/// implements the Politis-White (2004) data-driven optimal block length.
/// </para>
/// <para>
/// Tier A: Arithmetic resampling operations on floating-point types.
/// </para>
/// </remarks>
public sealed class StationaryBootstrapResampler<T>
    where T : IFloatingPoint<T>
{
    private readonly double _restartProbability;
    private readonly IRandomSource _rng;

    /// <summary>Mean expected block length (<c>1 / restartProbability</c>).</summary>
    public double MeanBlockLength { get; }

    /// <summary>
    /// Initializes the resampler with the given mean block length and random source.
    /// </summary>
    /// <param name="meanBlockLength">Expected block length. Must be ≥ 1.</param>
    /// <param name="random">Random source.</param>
    public StationaryBootstrapResampler(double meanBlockLength, IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (meanBlockLength < 1.0 || double.IsNaN(meanBlockLength))
        {
            throw new ArgumentOutOfRangeException(
                nameof(meanBlockLength), meanBlockLength,
                "Mean block length must be at least 1.");
        }

        MeanBlockLength = meanBlockLength;
        _restartProbability = 1.0 / meanBlockLength;
        _rng = random;
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static StationaryBootstrapResampler<T> FromSeed(double meanBlockLength, int? seed = null)
        => new(
            meanBlockLength,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<T>.GetEntropySeed()));

    /// <summary>Resamples a single series into a new path of the same length.</summary>
    public T[] Resample(T[] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Length == 0)
        {
            throw new ArgumentException("Source must contain at least one element.", nameof(source));
        }

        var n = source.Length;
        var result = new T[n];

        var idx = _rng.NextInt(n);
        for (var t = 0; t < n; t++)
        {
            result[t] = source[idx];
            if (_rng.NextDouble() < _restartProbability)
            {
                idx = _rng.NextInt(n);
            }
            else
            {
                idx = (idx + 1) % n;
            }
        }

        return result;
    }

    /// <summary>Resamples two paired series with shared block positions.</summary>
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

        var n = sourceA.Length;
        var resultA = new T[n];
        var resultB = new T[n];

        var idx = _rng.NextInt(n);
        for (var t = 0; t < n; t++)
        {
            resultA[t] = sourceA[idx];
            resultB[t] = sourceB[idx];
            if (_rng.NextDouble() < _restartProbability)
            {
                idx = _rng.NextInt(n);
            }
            else
            {
                idx = (idx + 1) % n;
            }
        }

        return (resultA, resultB);
    }
}

/// <summary>
/// Stationary bootstrap (Politis-Romano 1994). Block lengths are drawn from
/// a geometric distribution with mean <c>1/p</c>, yielding strictly
/// stationary resampled paths — unlike the fixed-block variants where
/// stationarity holds only approximately. At each step, with probability
/// <c>p</c> the next index restarts at a uniformly random source position;
/// with probability <c>1 − p</c> it advances one position from the current.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Politis, D. N. &amp; Romano, J. P. (1994). "The Stationary
/// Bootstrap." Journal of the American Statistical Association, 89(428),
/// 1303–1313.
/// </para>
/// <para>
/// The <see cref="MeanBlockLength"/> parameter controls the expected block
/// size (= 1/p). For automatic selection, use
/// <see cref="PolitisWhiteBlockLength.Estimate{T}(T[])"/> which
/// implements the Politis-White (2004) data-driven optimal block length.
/// </para>
/// <para>
/// Tier A: Delegates to <see cref="StationaryBootstrapResampler{T}"/> with T = <see cref="decimal"/>.
/// </para>
/// </remarks>
public sealed class StationaryBootstrapResampler
{
    private readonly StationaryBootstrapResampler<decimal> _impl;

    /// <summary>Mean expected block length (<c>1 / restartProbability</c>).</summary>
    public double MeanBlockLength => _impl.MeanBlockLength;

    /// <summary>
    /// Initializes the resampler with the given mean block length and random source.
    /// </summary>
    /// <param name="meanBlockLength">Expected block length. Must be ≥ 1.</param>
    /// <param name="random">Random source.</param>
    public StationaryBootstrapResampler(double meanBlockLength, IRandomSource random)
    {
        _impl = new StationaryBootstrapResampler<decimal>(meanBlockLength, random);
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static StationaryBootstrapResampler FromSeed(double meanBlockLength, int? seed = null)
        => new(
            meanBlockLength,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<decimal>.GetEntropySeed()));

    /// <summary>Resamples a single series into a new path of the same length.</summary>
    public decimal[] Resample(decimal[] source) => _impl.Resample(source);

    /// <summary>Resamples two paired series with shared block positions.</summary>
    public (decimal[] A, decimal[] B) ResamplePaired(decimal[] sourceA, decimal[] sourceB)
        => _impl.ResamplePaired(sourceA, sourceB);
}
