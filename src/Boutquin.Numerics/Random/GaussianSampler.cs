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
/// uniform draws per accepted pair. The first coordinate is cached unless it
/// is zero, matching the Numerical Recipes zero-sentinel listing. The ideal
/// continuous-model cost is approximately 1.27 uniform draws per normal.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Press et al. (2007), Numerical Recipes, third edition,
/// §7.3.4, pp. 364–365, Normaldev_BM listing.
/// </para>
/// <para>
/// Not thread-safe: holds cached state across calls. Instantiate one sampler
/// per thread or wrap with external synchronization.
/// </para>
/// <para>
/// <b>Reproducibility:</b> reproducible normal streams require a <b>fresh</b>
/// <see cref="GaussianSampler"/>. The sampler caches the first polar variate
/// between calls; that cache lives on the sampler, not on the underlying
/// <see cref="IRandomSource"/>. Re-seeding (or otherwise resetting) the source
/// while reusing the same sampler does <b>not</b> clear the cache, so the next
/// draw may return a stale cached normal computed from the pre-reset source —
/// silently breaking reproducibility. To replay a normal stream from a seed,
/// construct a new sampler over a freshly seeded source rather than re-seeding
/// an existing one.
/// </para>
/// </remarks>
public sealed class GaussianSampler
{
    private readonly IRandomSource _source;
    private double _cached;

    /// <summary>Initializes the sampler over a given random source.</summary>
    public GaussianSampler(IRandomSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        _source = source;
    }

    /// <summary>
    /// Draws a single sample from <c>N(0, 1)</c> — mean 0, variance 1 — using
    /// the Marsaglia polar method. Returns the second coordinate first and retains
    /// the first for the next call. A cached zero instead triggers a fresh pair,
    /// as in the Numerical Recipes listing. Ideal continuous-model cost is
    /// approximately 1.27 uniform draws per normal.
    /// </summary>
    public double Next()
    {
        if (_cached != 0.0)
        {
            var stored = _cached;
            _cached = 0.0;
            return stored;
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
        _cached = u * factor;
        return v * factor;
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

    /// <summary>Draws a normal with the given finite <paramref name="mean"/> and finite non-negative <paramref name="stdDev"/>.</summary>
    public double Next(double mean, double stdDev)
    {
        if (!double.IsFinite(mean))
        {
            throw new ArgumentOutOfRangeException(nameof(mean), mean, "Mean must be finite.");
        }

        if (!double.IsFinite(stdDev) || stdDev < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(stdDev), stdDev, "Standard deviation must be finite and non-negative.");
        }

        return Math.FusedMultiplyAdd(stdDev, Next(), mean);
    }
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
    /// the Marsaglia polar method. Returns the second coordinate first and retains
    /// the first for the next call. A cached zero instead triggers a fresh pair,
    /// as in the Numerical Recipes listing. Ideal continuous-model cost is
    /// approximately 1.27 uniform draws per normal.
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

    /// <summary>Draws a normal with the given finite <paramref name="mean"/> and finite non-negative <paramref name="stdDev"/>.</summary>
    public T Next(T mean, T stdDev)
    {
        if (!T.IsFinite(mean))
        {
            throw new ArgumentOutOfRangeException(nameof(mean), mean, "Mean must be finite.");
        }

        if (!T.IsFinite(stdDev) || stdDev < T.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(stdDev), stdDev, "Standard deviation must be finite and non-negative.");
        }

        var normal = Next();
        if (typeof(T) == typeof(double))
        {
            return T.CreateChecked(Math.FusedMultiplyAdd(double.CreateChecked(stdDev), double.CreateChecked(normal), double.CreateChecked(mean)));
        }

        if (typeof(T) == typeof(float))
        {
            return T.CreateChecked(MathF.FusedMultiplyAdd(float.CreateChecked(stdDev), float.CreateChecked(normal), float.CreateChecked(mean)));
        }

        if (typeof(T) == typeof(Half))
        {
            return T.CreateChecked(Half.FusedMultiplyAdd(Half.CreateChecked(stdDev), Half.CreateChecked(normal), Half.CreateChecked(mean)));
        }

        if ((mean < T.Zero && normal > T.Zero) || (mean > T.Zero && normal < T.Zero))
        {
            // Opposite signs: a finite product makes the sum safe. A product that
            // overflows (decimal throws, IEEE types round to infinity) can still have a
            // representable sum, so rearrange around mean / stdDev, which is bounded
            // here because stdDev exceeds the type maximum divided by |normal|.
            try
            {
                var product = stdDev * normal;
                if (T.IsFinite(product))
                {
                    return mean + product;
                }
            }
            catch (OverflowException)
            {
            }

            return stdDev * (normal + mean / stdDev);
        }

        return mean + stdDev * normal;
    }
}
