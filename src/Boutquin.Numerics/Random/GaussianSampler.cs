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

namespace Boutquin.Numerics.Random;

/// <summary>
/// Standard normal sampler using the Marsaglia polar method. Reads two
/// uniform draws per pair of normals; one normal is cached and returned on
/// the next call so the cost amortizes to ~1.27 uniform draws per normal.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Marsaglia, G. &amp; Bray, T. A. (1964). "A Convenient Method
/// for Generating Normal Variables." SIAM Review, 6(3), 260–264.
/// </para>
/// <para>
/// Not thread-safe: holds cached state across calls. Instantiate one sampler
/// per thread or wrap with external synchronization.
/// </para>
/// </remarks>
public sealed class GaussianSampler
{
    private readonly IRandomSource _source;
    private double _cached;
    private bool _hasCached;

    /// <summary>Initializes the sampler over a given random source.</summary>
    public GaussianSampler(IRandomSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        _source = source;
    }

    /// <summary>
    /// Draws a single sample from <c>N(0, 1)</c> — mean 0, variance 1 — using
    /// the Marsaglia polar method. Returns the cached second variate on alternate
    /// calls, so amortized cost is ~1.27 uniform draws per normal (the rejection
    /// factor for the unit-disk acceptance region).
    /// </summary>
    public double Next()
    {
        if (_hasCached)
        {
            _hasCached = false;
            return _cached;
        }

        double u, v, s;
        do
        {
            u = 2.0 * _source.NextDouble() - 1.0;
            v = 2.0 * _source.NextDouble() - 1.0;
            s = u * u + v * v;
        }
        while (s >= 1.0 || s == 0.0);

        var factor = Math.Sqrt(-2.0 * Math.Log(s) / s);
        _cached = v * factor;
        _hasCached = true;
        return u * factor;
    }

    /// <summary>Draws <paramref name="count"/> standard normal values.</summary>
    public double[] NextBatch(int count)
    {
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Count must be non-negative.");
        }

        var result = new double[count];
        for (var i = 0; i < count; i++)
        {
            result[i] = Next();
        }

        return result;
    }

    /// <summary>Draws a normal with the given <paramref name="mean"/> and <paramref name="stdDev"/>.</summary>
    public double Next(double mean, double stdDev) => mean + stdDev * Next();
}

/// <summary>
/// Generic standard normal sampler using the Marsaglia polar method with
/// type-parameterized floating-point output.
/// </summary>
/// <typeparam name="T">The floating-point type for random values.</typeparam>
/// <remarks>
/// Tier C: Wraps <see cref="GaussianSampler"/> and casts from double.
/// </remarks>
public sealed class GaussianSampler<T>
    where T : IFloatingPoint<T>
{
    private readonly GaussianSampler _inner;

    /// <summary>Initializes the sampler over a given random source.</summary>
    public GaussianSampler(IRandomSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        _inner = new GaussianSampler(source);
    }

    /// <summary>
    /// Draws a single sample from <c>N(0, 1)</c> — mean 0, variance 1 — using
    /// the Marsaglia polar method. Returns the cached second variate on alternate
    /// calls, so amortized cost is ~1.27 uniform draws per normal (the rejection
    /// factor for the unit-disk acceptance region).
    /// </summary>
    public T Next() => T.CreateChecked(_inner.Next());

    /// <summary>Draws <paramref name="count"/> standard normal values.</summary>
    public T[] NextBatch(int count)
    {
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Count must be non-negative.");
        }

        var result = new T[count];
        for (var i = 0; i < count; i++)
        {
            result[i] = Next();
        }

        return result;
    }

    /// <summary>Draws a normal with the given <paramref name="mean"/> and <paramref name="stdDev"/>.</summary>
    public T Next(T mean, T stdDev) => mean + stdDev * Next();
}
