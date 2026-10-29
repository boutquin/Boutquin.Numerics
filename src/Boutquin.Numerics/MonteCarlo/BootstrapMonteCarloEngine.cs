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
/// Summary of a bootstrap Monte Carlo run over a scalar statistic.
/// </summary>
/// <typeparam name="T">Floating-point type for the statistic values.</typeparam>
/// <param name="SimulationCount">Number of bootstrap iterations executed.</param>
/// <param name="Statistics">Sorted (ascending) distribution of the simulated statistic.</param>
/// <param name="Median">50th percentile.</param>
/// <param name="Percentile5">5th percentile (lower tail).</param>
/// <param name="Percentile95">95th percentile (upper tail).</param>
/// <param name="Mean">Arithmetic mean of the simulated statistic.</param>
/// <remarks>
/// Tier A: Generic result record for bootstrap Monte Carlo simulations.
/// </remarks>
public sealed record BootstrapMonteCarloResult<T>(
    int SimulationCount,
    IReadOnlyList<T> Statistics,
    T Median,
    T Percentile5,
    T Percentile95,
    T Mean)
    where T : IFloatingPoint<T>
{
    private IReadOnlyList<T> _statistics = BootstrapResultSnapshots.Snapshot(Statistics);

    /// <summary>Sorted simulation statistics, isolated from caller-owned storage.</summary>
    public IReadOnlyList<T> Statistics
    {
        get => _statistics;
        init => _statistics = BootstrapResultSnapshots.Snapshot(value);
    }
}

/// <summary>
/// Summary of a bootstrap Monte Carlo run over a vector of statistics.
/// Each named statistic has its own sorted distribution; summaries
/// (median, 5th/95th percentile, mean) are provided per-statistic.
/// </summary>
/// <typeparam name="T">Floating-point type for the statistic values.</typeparam>
/// <param name="SimulationCount">Number of bootstrap iterations executed.</param>
/// <param name="Names">Statistic names in the order they were produced.</param>
/// <param name="Statistics">Per-statistic sorted distribution (row = statistic, column = simulation).</param>
/// <param name="Means">Arithmetic mean of each statistic.</param>
/// <param name="Medians">50th percentile of each statistic.</param>
/// <param name="Percentile5s">5th percentile of each statistic.</param>
/// <param name="Percentile95s">95th percentile of each statistic.</param>
/// <remarks>
/// Tier A: Generic result record for multi-statistic bootstrap Monte Carlo simulations.
/// </remarks>
public sealed record MultiStatisticBootstrapResult<T>(
    int SimulationCount,
    IReadOnlyList<string> Names,
    IReadOnlyList<T[]> Statistics,
    IReadOnlyList<T> Means,
    IReadOnlyList<T> Medians,
    IReadOnlyList<T> Percentile5s,
    IReadOnlyList<T> Percentile95s)
    where T : IFloatingPoint<T>
{
    private IReadOnlyList<string> _names = BootstrapResultSnapshots.Snapshot(Names);
    private IReadOnlyList<T[]> _statistics = new DefensiveArrayList<T>(Statistics);
    private IReadOnlyList<T> _means = BootstrapResultSnapshots.Snapshot(Means);
    private IReadOnlyList<T> _medians = BootstrapResultSnapshots.Snapshot(Medians);
    private IReadOnlyList<T> _percentile5s = BootstrapResultSnapshots.Snapshot(Percentile5s);
    private IReadOnlyList<T> _percentile95s = BootstrapResultSnapshots.Snapshot(Percentile95s);

    /// <summary>Statistic names isolated from caller-owned storage.</summary>
    public IReadOnlyList<string> Names
    {
        get => _names;
        init => _names = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Distributions whose rows are copied on each access.</summary>
    public IReadOnlyList<T[]> Statistics
    {
        get => _statistics;
        init => _statistics = new DefensiveArrayList<T>(value);
    }

    /// <summary>Mean values isolated from caller-owned storage.</summary>
    public IReadOnlyList<T> Means
    {
        get => _means;
        init => _means = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Median values isolated from caller-owned storage.</summary>
    public IReadOnlyList<T> Medians
    {
        get => _medians;
        init => _medians = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Fifth percentiles isolated from caller-owned storage.</summary>
    public IReadOnlyList<T> Percentile5s
    {
        get => _percentile5s;
        init => _percentile5s = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Ninety-fifth percentiles isolated from caller-owned storage.</summary>
    public IReadOnlyList<T> Percentile95s
    {
        get => _percentile95s;
        init => _percentile95s = BootstrapResultSnapshots.Snapshot(value);
    }
}

internal static class BootstrapResultSnapshots
{
    internal static IReadOnlyList<TValue> Snapshot<TValue>(IReadOnlyList<TValue> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return Array.AsReadOnly(values.ToArray());
    }
}

internal sealed class DefensiveArrayList<T> : IReadOnlyList<T[]>
{
    private readonly T[][] _rows;

    public DefensiveArrayList(IReadOnlyList<T[]> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        _rows = new T[rows.Count][];
        for (var i = 0; i < rows.Count; i++)
        {
            ArgumentNullException.ThrowIfNull(rows[i]);
            _rows[i] = (T[])rows[i].Clone();
        }
    }

    public int Count => _rows.Length;

    public T[] this[int index] => (T[])_rows[index].Clone();

    public IEnumerator<T[]> GetEnumerator()
    {
        for (var i = 0; i < _rows.Length; i++)
        {
            yield return (T[])_rows[i].Clone();
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>
/// IID bootstrap Monte Carlo engine (Efron 1979). Resamples with replacement
/// from a single series — each observation drawn independently and uniformly —
/// applies a user-supplied statistic to each resampled path, and summarizes
/// the empirical distribution of the statistic. Appropriate for IID data;
/// for serially-dependent data prefer <see cref="BootstrapResampler{T}"/>
/// (circular block), <see cref="MovingBlockBootstrapResampler{T}"/>,
/// <see cref="StationaryBootstrapResampler{T}"/>, or <see cref="WildBootstrapResampler{T}"/>
/// as dictated by the dependence structure.
/// </summary>
/// <typeparam name="T">Floating-point type for the observation and statistic values.</typeparam>
/// <remarks>
/// <para>
/// Supports two statistic APIs:
/// <list type="bullet">
/// <item><description><see cref="Run"/> — scalar statistic.</description></item>
/// <item><description><see cref="RunMulti"/> — vector of statistics computed in a single pass over each resampled path.</description></item>
/// </list>
/// The multi-statistic overload avoids the O(K) passes required to compute K separate
/// statistics on the same resampled data — useful for bundles like
/// (Sharpe, Sortino, MaxDD, IR) that share intermediate calculations.
/// </para>
/// <para>
/// Uses an <see cref="IRandomSource"/> for deterministic reproducibility.
/// Callers that need non-deterministic runs should construct the engine
/// with a system-entropy-seeded source.
/// </para>
/// <para>
/// The supplied <see cref="IRandomSource"/> is consumed sequentially; do not
/// share a single source instance across engine instances or threads. See
/// <see cref="IRandomSource"/> for thread-safety semantics.
/// </para>
/// <para>
/// Tier A: Arithmetic resampling and statistic computation on floating-point types.
/// </para>
/// </remarks>
public sealed class BootstrapMonteCarloEngine<T>
    where T : IFloatingPoint<T>
{
    private readonly int _simulationCount;
    private readonly IRandomSource _rng;

    /// <summary>Initializes the engine with the given simulation count and random source.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="simulationCount"/> is non-positive.</exception>
    public BootstrapMonteCarloEngine(int simulationCount, IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (simulationCount <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(simulationCount), simulationCount, "Simulation count must be positive.");
        }

        _simulationCount = simulationCount;
        _rng = random;
    }

    /// <summary>
    /// Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.
    /// </summary>
    public static BootstrapMonteCarloEngine<T> FromSeed(int simulationCount = 1000, int? seed = null)
        => new(
            simulationCount,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<T>.GetEntropySeed()));

    /// <summary>Number of bootstrap iterations.</summary>
    public int SimulationCount => _simulationCount;

    /// <summary>
    /// Runs the bootstrap over a scalar statistic.
    /// </summary>
    public BootstrapMonteCarloResult<T> Run(ReadOnlySpan<T> observations, Func<ReadOnlySpan<T>, T> statistic)
    {
        ArgumentNullException.ThrowIfNull(statistic);
        if (observations.Length == 0)
        {
            throw new ArgumentException("Observations must contain at least one element.", nameof(observations));
        }

        var statistics = new T[_simulationCount];
        var buffer = new T[observations.Length];
        for (var sim = 0; sim < _simulationCount; sim++)
        {
            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = observations[_rng.NextInt(observations.Length)];
            }

            statistics[sim] = statistic(buffer.AsSpan());
        }

        Array.Sort(statistics);

        var mean = ComputeMean(statistics);

        return new BootstrapMonteCarloResult<T>(
            SimulationCount: _simulationCount,
            Statistics: Array.AsReadOnly(statistics),
            Median: Percentile.Compute(statistics, 0.50),
            Percentile5: Percentile.Compute(statistics, 0.05),
            Percentile95: Percentile.Compute(statistics, 0.95),
            Mean: mean);
    }

    /// <summary>
    /// Runs the bootstrap over a vector of <paramref name="statisticCount"/>
    /// statistics produced jointly by <paramref name="statisticWriter"/>. The
    /// writer receives the resampled buffer and a pre-allocated output span
    /// sized to <paramref name="statisticCount"/>.
    /// </summary>
    /// <param name="observations">Input series.</param>
    /// <param name="statisticCount">Number of scalar statistics per simulation.</param>
    /// <param name="statisticWriter">Delegate that fills a length-<paramref name="statisticCount"/> buffer given the resampled path.</param>
    /// <param name="names">Optional statistic names (length must match <paramref name="statisticCount"/>). Defaults to "stat_0", "stat_1", …</param>
    public MultiStatisticBootstrapResult<T> RunMulti(
        ReadOnlySpan<T> observations,
        int statisticCount,
        Action<ReadOnlySpan<T>, Span<T>> statisticWriter,
        IReadOnlyList<string>? names = null)
    {
        ArgumentNullException.ThrowIfNull(statisticWriter);
        if (observations.Length == 0)
        {
            throw new ArgumentException("Observations must contain at least one element.", nameof(observations));
        }

        if (statisticCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(statisticCount), statisticCount, "Statistic count must be positive.");
        }

        if (names is not null && names.Count != statisticCount)
        {
            throw new ArgumentException("Names length must match statistic count.", nameof(names));
        }

        var statistics = new T[statisticCount][];
        for (var k = 0; k < statisticCount; k++)
        {
            statistics[k] = new T[_simulationCount];
        }

        var buffer = new T[observations.Length];
        var statBuffer = new T[statisticCount];
        for (var sim = 0; sim < _simulationCount; sim++)
        {
            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = observations[_rng.NextInt(observations.Length)];
            }

            Array.Clear(statBuffer, 0, statisticCount);
            statisticWriter(buffer.AsSpan(), statBuffer.AsSpan());
            for (var k = 0; k < statisticCount; k++)
            {
                statistics[k][sim] = statBuffer[k];
            }
        }

        var means = new T[statisticCount];
        var medians = new T[statisticCount];
        var p5s = new T[statisticCount];
        var p95s = new T[statisticCount];

        for (var k = 0; k < statisticCount; k++)
        {
            Array.Sort(statistics[k]);
            means[k] = ComputeMean(statistics[k]);
            medians[k] = Percentile.Compute(statistics[k], 0.50);
            p5s[k] = Percentile.Compute(statistics[k], 0.05);
            p95s[k] = Percentile.Compute(statistics[k], 0.95);
        }

        IReadOnlyList<string> finalNames;
        if (names is not null)
        {
            finalNames = Array.AsReadOnly(names.ToArray());
        }
        else
        {
            var generated = new string[statisticCount];
            for (var i = 0; i < statisticCount; i++)
            {
                generated[i] = $"stat_{i}";
            }

            finalNames = Array.AsReadOnly(generated);
        }

        // The result constructor owns the defensive snapshot of these working rows.
        return new MultiStatisticBootstrapResult<T>(
            SimulationCount: _simulationCount,
            Names: finalNames,
            Statistics: statistics,
            Means: Array.AsReadOnly(means),
            Medians: Array.AsReadOnly(medians),
            Percentile5s: Array.AsReadOnly(p5s),
            Percentile95s: Array.AsReadOnly(p95s));
    }

    private static T ComputeMean(ReadOnlySpan<T> statistics)
    {
        if (typeof(T) == typeof(decimal))
        {
            BigInteger sum = BigInteger.Zero;
            var decimalScale = 0;
            for (var i = 0; i < statistics.Length; i++)
            {
                var bits = decimal.GetBits(decimal.CreateChecked(statistics[i]));
                var itemScale = (bits[3] >> 16) & 0xff;
                if (itemScale > decimalScale)
                {
                    sum *= BigInteger.Pow(10, itemScale - decimalScale);
                    decimalScale = itemScale;
                }

                var mantissa = (BigInteger)(uint)bits[0]
                    | ((BigInteger)(uint)bits[1] << 32)
                    | ((BigInteger)(uint)bits[2] << 64);
                if ((bits[3] & int.MinValue) != 0)
                {
                    mantissa = -mantissa;
                }

                sum += mantissa * BigInteger.Pow(10, decimalScale - itemScale);
            }

            var denominator = statistics.Length * BigInteger.Pow(10, decimalScale);
            var magnitude = BigInteger.Abs(sum);
            var maximum = (BigInteger.One << 96) - BigInteger.One;
            for (var outputScale = 28; outputScale >= 0; outputScale--)
            {
                var quotient = BigInteger.DivRem(magnitude * BigInteger.Pow(10, outputScale), denominator, out var remainder);
                var twiceRemainder = remainder * 2;
                if (twiceRemainder > denominator || (twiceRemainder == denominator && !quotient.IsEven))
                {
                    quotient++;
                }

                if (quotient <= maximum)
                {
                    var result = new decimal(
                        (int)(uint)(quotient & uint.MaxValue),
                        (int)(uint)((quotient >> 32) & uint.MaxValue),
                        (int)(uint)((quotient >> 64) & uint.MaxValue),
                        sum.Sign < 0,
                        (byte)outputScale);
                    return T.CreateChecked(result);
                }
            }

            throw new OverflowException("Bootstrap mean exceeds the Decimal range.");
        }

        var scale = T.Zero;
        for (var i = 0; i < statistics.Length; i++)
        {
            scale = T.Max(scale, T.Abs(statistics[i]));
        }

        if (scale == T.Zero)
        {
            return T.Zero;
        }

        if (typeof(T) == typeof(Half))
        {
            var scaleDouble = double.CreateChecked(scale);
            var normalizedSum = 0.0;
            for (var i = 0; i < statistics.Length; i++)
            {
                normalizedSum += double.CreateChecked(statistics[i]) / scaleDouble;
            }

            return T.CreateChecked(normalizedSum / statistics.Length * scaleDouble);
        }

        // Normalize before accumulating so a finite mean does not require a
        // potentially unrepresentable sum (notably for Half).
        var mean = T.Zero;
        for (var i = 0; i < statistics.Length; i++)
        {
            mean += (statistics[i] / scale - mean) / T.CreateSaturating(i + 1);
        }

        return mean * scale;
    }
}

/// <summary>
/// Summary of a bootstrap Monte Carlo run over a scalar statistic.
/// </summary>
/// <param name="SimulationCount">Number of bootstrap iterations executed.</param>
/// <param name="Statistics">Sorted (ascending) distribution of the simulated statistic.</param>
/// <param name="Median">50th percentile.</param>
/// <param name="Percentile5">5th percentile (lower tail).</param>
/// <param name="Percentile95">95th percentile (upper tail).</param>
/// <param name="Mean">Arithmetic mean of the simulated statistic.</param>
/// <remarks>
/// Tier A: Delegates to <see cref="BootstrapMonteCarloResult{T}"/> with T = <see cref="decimal"/>.
/// </remarks>
public sealed record BootstrapMonteCarloResult(
    int SimulationCount,
    IReadOnlyList<decimal> Statistics,
    decimal Median,
    decimal Percentile5,
    decimal Percentile95,
    decimal Mean)
{
    private IReadOnlyList<decimal> _statistics = BootstrapResultSnapshots.Snapshot(Statistics);

    /// <summary>Sorted simulation statistics, isolated from caller-owned storage.</summary>
    public IReadOnlyList<decimal> Statistics
    {
        get => _statistics;
        init => _statistics = BootstrapResultSnapshots.Snapshot(value);
    }
}

/// <summary>
/// Summary of a bootstrap Monte Carlo run over a vector of statistics.
/// Each named statistic has its own sorted distribution; summaries
/// (median, 5th/95th percentile, mean) are provided per-statistic.
/// </summary>
/// <param name="SimulationCount">Number of bootstrap iterations executed.</param>
/// <param name="Names">Statistic names in the order they were produced.</param>
/// <param name="Statistics">Per-statistic sorted distribution (row = statistic, column = simulation).</param>
/// <param name="Means">Arithmetic mean of each statistic.</param>
/// <param name="Medians">50th percentile of each statistic.</param>
/// <param name="Percentile5s">5th percentile of each statistic.</param>
/// <param name="Percentile95s">95th percentile of each statistic.</param>
/// <remarks>
/// Tier A: Delegates to <see cref="MultiStatisticBootstrapResult{T}"/> with T = <see cref="decimal"/>.
/// </remarks>
public sealed record MultiStatisticBootstrapResult(
    int SimulationCount,
    IReadOnlyList<string> Names,
    IReadOnlyList<decimal[]> Statistics,
    IReadOnlyList<decimal> Means,
    IReadOnlyList<decimal> Medians,
    IReadOnlyList<decimal> Percentile5s,
    IReadOnlyList<decimal> Percentile95s)
{
    private IReadOnlyList<string> _names = BootstrapResultSnapshots.Snapshot(Names);
    private IReadOnlyList<decimal[]> _statistics = new DefensiveArrayList<decimal>(Statistics);
    private IReadOnlyList<decimal> _means = BootstrapResultSnapshots.Snapshot(Means);
    private IReadOnlyList<decimal> _medians = BootstrapResultSnapshots.Snapshot(Medians);
    private IReadOnlyList<decimal> _percentile5s = BootstrapResultSnapshots.Snapshot(Percentile5s);
    private IReadOnlyList<decimal> _percentile95s = BootstrapResultSnapshots.Snapshot(Percentile95s);

    /// <summary>Statistic names isolated from caller-owned storage.</summary>
    public IReadOnlyList<string> Names
    {
        get => _names;
        init => _names = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Distributions whose rows are copied on each access.</summary>
    public IReadOnlyList<decimal[]> Statistics
    {
        get => _statistics;
        init => _statistics = new DefensiveArrayList<decimal>(value);
    }

    /// <summary>Mean values isolated from caller-owned storage.</summary>
    public IReadOnlyList<decimal> Means
    {
        get => _means;
        init => _means = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Median values isolated from caller-owned storage.</summary>
    public IReadOnlyList<decimal> Medians
    {
        get => _medians;
        init => _medians = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Fifth percentiles isolated from caller-owned storage.</summary>
    public IReadOnlyList<decimal> Percentile5s
    {
        get => _percentile5s;
        init => _percentile5s = BootstrapResultSnapshots.Snapshot(value);
    }

    /// <summary>Ninety-fifth percentiles isolated from caller-owned storage.</summary>
    public IReadOnlyList<decimal> Percentile95s
    {
        get => _percentile95s;
        init => _percentile95s = BootstrapResultSnapshots.Snapshot(value);
    }
}

/// <summary>
/// IID bootstrap Monte Carlo engine (Efron 1979). Resamples with replacement
/// from a single series — each observation drawn independently and uniformly —
/// applies a user-supplied statistic to each resampled path, and summarizes
/// the empirical distribution of the statistic. Appropriate for IID data;
/// for serially-dependent data prefer <see cref="BootstrapResampler"/>
/// (circular block), <see cref="MovingBlockBootstrapResampler"/>,
/// <see cref="StationaryBootstrapResampler"/>, or <see cref="WildBootstrapResampler"/>
/// as dictated by the dependence structure.
/// </summary>
/// <remarks>
/// <para>
/// Supports two statistic APIs:
/// <list type="bullet">
/// <item><description><see cref="Run(ReadOnlySpan{decimal}, Func{ReadOnlySpan{decimal}, decimal})"/> — scalar statistic.</description></item>
/// <item><description><see cref="RunMulti(ReadOnlySpan{decimal}, int, Action{ReadOnlySpan{decimal}, Span{decimal}}, IReadOnlyList{string})"/> — vector of statistics computed in a single pass over each resampled path.</description></item>
/// </list>
/// The multi-statistic overload avoids the O(K) passes required to compute K separate
/// statistics on the same resampled data — useful for bundles like
/// (Sharpe, Sortino, MaxDD, IR) that share intermediate calculations.
/// </para>
/// <para>
/// Uses an <see cref="IRandomSource"/> for deterministic reproducibility.
/// Callers that need non-deterministic runs should construct the engine
/// with a system-entropy-seeded source.
/// </para>
/// <para>
/// The supplied <see cref="IRandomSource"/> is consumed sequentially; do not
/// share a single source instance across engine instances or threads. See
/// <see cref="IRandomSource"/> for thread-safety semantics.
/// </para>
/// <para>
/// <b>Working type: <see cref="decimal"/>.</b> This non-generic facade delegates
/// to <see cref="BootstrapMonteCarloEngine{T}"/> instantiated at
/// <see cref="decimal"/>; observations, statistics, and summaries are all
/// <see cref="decimal"/>. For <see cref="double"/>-precision runs, use the generic
/// <see cref="BootstrapMonteCarloEngine{T}"/> with <c>T = double</c>. (Tier A.)
/// </para>
/// </remarks>
public sealed class BootstrapMonteCarloEngine
{
    private readonly BootstrapMonteCarloEngine<decimal> _impl;

    /// <summary>Initializes the engine with the given simulation count and random source.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="simulationCount"/> is non-positive.</exception>
    public BootstrapMonteCarloEngine(int simulationCount, IRandomSource random)
    {
        _impl = new BootstrapMonteCarloEngine<decimal>(simulationCount, random);
    }

    /// <summary>
    /// Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.
    /// </summary>
    public static BootstrapMonteCarloEngine FromSeed(int simulationCount = 1000, int? seed = null)
        => new(
            simulationCount,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<decimal>.GetEntropySeed()));

    /// <summary>Number of bootstrap iterations.</summary>
    public int SimulationCount => _impl.SimulationCount;

    /// <summary>
    /// Runs the bootstrap over a scalar statistic.
    /// </summary>
    public BootstrapMonteCarloResult Run(ReadOnlySpan<decimal> observations, Func<ReadOnlySpan<decimal>, decimal> statistic)
    {
        var result = _impl.Run(observations, statistic);
        return new BootstrapMonteCarloResult(
            result.SimulationCount,
            result.Statistics,
            result.Median,
            result.Percentile5,
            result.Percentile95,
            result.Mean);
    }

    /// <summary>
    /// Runs the bootstrap over a vector of <paramref name="statisticCount"/>
    /// statistics produced jointly by <paramref name="statisticWriter"/>. The
    /// writer receives the resampled buffer and a pre-allocated output span
    /// sized to <paramref name="statisticCount"/>.
    /// </summary>
    /// <param name="observations">Input series.</param>
    /// <param name="statisticCount">Number of scalar statistics per simulation.</param>
    /// <param name="statisticWriter">Delegate that fills a length-<paramref name="statisticCount"/> buffer given the resampled path.</param>
    /// <param name="names">Optional statistic names (length must match <paramref name="statisticCount"/>). Defaults to "stat_0", "stat_1", …</param>
    public MultiStatisticBootstrapResult RunMulti(
        ReadOnlySpan<decimal> observations,
        int statisticCount,
        Action<ReadOnlySpan<decimal>, Span<decimal>> statisticWriter,
        IReadOnlyList<string>? names = null)
    {
        var result = _impl.RunMulti(observations, statisticCount, statisticWriter, names);
        return new MultiStatisticBootstrapResult(
            result.SimulationCount,
            result.Names,
            result.Statistics,
            result.Means,
            result.Medians,
            result.Percentile5s,
            result.Percentile95s);
    }
}
