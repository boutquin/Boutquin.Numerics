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
/// Multiplier weight family for the wild bootstrap. The bootstrap path is
/// formed by element-wise multiplication of the original residuals by IID
/// draws from the chosen distribution.
/// </summary>
public enum WildBootstrapWeights
{
    /// <summary>Mammen (1993) two-point distribution. Mean 0, variance 1, third moment 1.</summary>
    Mammen,

    /// <summary>Rademacher (±1 with equal probability). Mean 0, variance 1; symmetric.</summary>
    Rademacher,

    /// <summary>Standard normal N(0, 1). Smooth weights, no moment-matching guarantees beyond variance.</summary>
    Gaussian,
}

/// <summary>
/// Wild bootstrap (Wu 1986; Liu 1988; Mammen 1993) — heteroskedastic-consistent
/// bootstrap that multiplies each residual by an IID weight with mean 0 and
/// variance 1 (and, in Mammen's variant, third moment 1 to match higher-order
/// asymptotics).
/// </summary>
/// <typeparam name="T">Floating-point type for the residual values.</typeparam>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Mammen, E. (1993). "Bootstrap and Wild Bootstrap for High Dimensional Linear Models." Annals of Statistics, 21(1), 255–285.</description></item>
/// <item><description>Davidson, R. &amp; Flachaire, E. (2008). "The Wild Bootstrap, Tamed at Last." Journal of Econometrics, 146(1), 162–169.</description></item>
/// </list>
/// </para>
/// <para>
/// Use the wild bootstrap when residuals are heteroskedastic — the IID
/// pairs/block bootstraps assume homoskedasticity. The Rademacher variant
/// is the recommended default for robust inference per Davidson-Flachaire 2008;
/// Mammen's two-point is preferred when third-moment matching matters.
/// </para>
/// <para>
/// Tier A: Arithmetic resampling operations on floating-point types.
/// </para>
/// </remarks>
public sealed class WildBootstrapResampler<T>
    where T : IFloatingPoint<T>
{
    private readonly WildBootstrapWeights _weights;
    private readonly IRandomSource _rng;
    private readonly GaussianSampler? _gauss;

    private static readonly double s_mammenP = (Math.Sqrt(5.0) + 1.0) / (2.0 * Math.Sqrt(5.0));
    private static readonly double s_mammenLow = -(Math.Sqrt(5.0) - 1.0) / 2.0;
    private static readonly double s_mammenHigh = (Math.Sqrt(5.0) + 1.0) / 2.0;

    /// <summary>Initializes the resampler with the given weights and random source.</summary>
    public WildBootstrapResampler(WildBootstrapWeights weights, IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        _weights = weights;
        _rng = random;
        _gauss = weights == WildBootstrapWeights.Gaussian ? new GaussianSampler(random) : null;
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static WildBootstrapResampler<T> FromSeed(WildBootstrapWeights weights, int? seed = null)
        => new(
            weights,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<T>.GetEntropySeed()));

    /// <summary>Weight family used for resampling.</summary>
    public WildBootstrapWeights Weights => _weights;

    /// <summary>
    /// Resamples residuals by IID multiplication. The returned array has
    /// the same length as <paramref name="residuals"/>; element <c>t</c> is
    /// <c>residuals[t] · w_t</c> with <c>w_t</c> drawn from the chosen
    /// weight family.
    /// </summary>
    public T[] Resample(T[] residuals)
    {
        ArgumentNullException.ThrowIfNull(residuals);
        var n = residuals.Length;
        if (n == 0)
        {
            throw new ArgumentException("Residuals must contain at least one element.", nameof(residuals));
        }

        var result = new T[n];
        for (var i = 0; i < n; i++)
        {
            var w = NextWeight();
            result[i] = residuals[i] * T.CreateChecked(w);
        }

        return result;
    }

    private double NextWeight() => _weights switch
    {
        WildBootstrapWeights.Mammen => _rng.NextDouble() < s_mammenP ? s_mammenLow : s_mammenHigh,
        WildBootstrapWeights.Rademacher => _rng.NextDouble() < 0.5 ? -1.0 : 1.0,
        WildBootstrapWeights.Gaussian => _gauss!.Next(),
        _ => throw new InvalidOperationException($"Unknown weight family: {_weights}"),
    };
}

/// <summary>
/// Wild bootstrap (Wu 1986; Liu 1988; Mammen 1993) — heteroskedastic-consistent
/// bootstrap that multiplies each residual by an IID weight with mean 0 and
/// variance 1 (and, in Mammen's variant, third moment 1 to match higher-order
/// asymptotics).
/// </summary>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Mammen, E. (1993). "Bootstrap and Wild Bootstrap for High Dimensional Linear Models." Annals of Statistics, 21(1), 255–285.</description></item>
/// <item><description>Davidson, R. &amp; Flachaire, E. (2008). "The Wild Bootstrap, Tamed at Last." Journal of Econometrics, 146(1), 162–169.</description></item>
/// </list>
/// </para>
/// <para>
/// Use the wild bootstrap when residuals are heteroskedastic — the IID
/// pairs/block bootstraps assume homoskedasticity. The Rademacher variant
/// is the recommended default for robust inference per Davidson-Flachaire 2008;
/// Mammen's two-point is preferred when third-moment matching matters.
/// </para>
/// <para>
/// Tier A: Delegates to <see cref="WildBootstrapResampler{T}"/> with T = <see cref="decimal"/>.
/// </para>
/// </remarks>
public sealed class WildBootstrapResampler
{
    private readonly WildBootstrapResampler<decimal> _impl;

    /// <summary>Initializes the resampler with the given weights and random source.</summary>
    public WildBootstrapResampler(WildBootstrapWeights weights, IRandomSource random)
    {
        _impl = new WildBootstrapResampler<decimal>(weights, random);
    }

    /// <summary>Convenience factory that wraps a seeded <see cref="Pcg64RandomSource"/>.</summary>
    public static WildBootstrapResampler FromSeed(WildBootstrapWeights weights, int? seed = null)
        => new(
            weights,
            seed.HasValue
                ? new Pcg64RandomSource((ulong)seed.Value)
                : new Pcg64RandomSource(BootstrapResampler<decimal>.GetEntropySeed()));

    /// <summary>Weight family used for resampling.</summary>
    public WildBootstrapWeights Weights => _impl.Weights;

    /// <summary>
    /// Resamples residuals by IID multiplication. The returned array has
    /// the same length as <paramref name="residuals"/>; element <c>t</c> is
    /// <c>residuals[t] · w_t</c> with <c>w_t</c> drawn from the chosen
    /// weight family.
    /// </summary>
    public decimal[] Resample(decimal[] residuals) => _impl.Resample(residuals);
}
