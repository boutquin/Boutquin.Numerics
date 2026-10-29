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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Result of the GT-inspired library variant, exposing its loss and component diagnostics.
/// </summary>
/// <typeparam name="T">Floating-point type.</typeparam>
/// <param name="Score">
/// The GT-inspired library loss (the piecewise definition). Interpreted as an
/// optimization <em>loss</em>: lower is better. For an outperforming strategy
/// (<c>z &gt; 1</c>) it equals <c>−(µ · ln(z) · r² / (σ_d + ε))</c>, which is
/// negative when the composite utility <c>µ · ln(z) · r²</c> is positive; for
/// marginal (<c>0 &lt; z ≤ 1</c>) or underperforming (<c>z ≤ 0</c>) strategies it
/// is a positive penalty in <c>[0, 200]</c>. The inclusive minimum-count safeguard
/// overrides these branches with <c>999 − 899·N/minimumObservations</c>.
/// </param>
/// <param name="Mu">Mean strategy return per observation (<c>µ</c>).</param>
/// <param name="Z">
/// Significance Z-score using sample standard deviation, <c>z = (µ − µ_m) / (σ / √N)</c>, measuring
/// outperformance of the benchmark relative to sampling noise.
/// This diagnostic can be infinite when significance exceeds the working type's
/// range; score evaluation retains its finite logarithm internally.
/// </param>
/// <param name="RSquared">
/// R² of cumulative arithmetic returns against a linear time trend (Kestner 1996) —
/// the consistency term, in <c>[0, 1]</c>.
/// </param>
/// <param name="DownsideDeviation">
/// Downside deviation <c>σ_d = √(mean(min(0, rᵢ)²))</c> with a zero minimum
/// acceptable return (Sortino &amp; van der Meer 1991) — the denominator term.
/// </param>
public sealed record GtInspiredScoreResult<T>(
    T Score,
    T Mu,
    T Z,
    T RSquared,
    T DownsideDeviation)
    where T : IFloatingPointIeee754<T>;

/// <summary>
/// Computes a GT-inspired composite search loss with explicit library component definitions.
/// </summary>
/// <typeparam name="T">IEEE-754 floating-point type supporting logarithms and square roots.</typeparam>
/// <remarks>
/// <para>Tier B: requires <c>IFloatingPointIeee754&lt;T&gt;</c>; supports
/// <c>double</c>, <c>float</c> and <c>Half</c>.</para>
/// <para>Inspired by Sheppert (2026), "The GT-Score", arXiv:2602.00080.
/// This library variant uses sample standard deviation (N − 1 denominator),
/// squared correlation of cumulative arithmetic returns with time 1..N,
/// zero-target downside RMS over all N observations, and additive epsilon
/// protection in the denominator. No author parity or empirical generalization
/// guarantee is claimed: the authenticated author implementation uses population
/// standard deviation, individual-return regression and negative-only dispersion.</para>
/// <para>The piecewise score is a loss (lower is better). At or below the inclusive
/// minimum-observation threshold it is 999 − 899·N/minimumObservations. Above it,
/// z ≤ 0 gives 100 + 100·(1 − exp(−|z−1|)); 0 &lt; z ≤ 1 gives
/// 100·(1 − exp(−|z−1|)); z &gt; 1 gives −µ·ln(z)·r²/(σ_d + ε).
/// Constant returns define z = 0; constant cumulative returns define r² = 0.</para>
/// <para>Time and auxiliary storage are O(N). Reductions are scaled to limit
/// intermediate overflow. Half reductions widen to double and round the diagnostic
/// scalars back; results can vary with the working type and floating-point rounding.
/// Independent variant checks are in the statistics unit tests and scalar reference vectors.</para>
/// </remarks>
public static class GtInspiredScore<T>
    where T : IFloatingPointIeee754<T>
{
    /// <summary>
    /// Default smoothing parameter <c>ε = 10⁻⁶</c> guarding the <c>σ_d</c>
    /// denominator for the library variant.
    /// </summary>
    private static T DefaultEpsilon => T.CreateChecked(1e-6);

    /// <summary>
    /// The documented default <c>ε = 10⁻⁶</c> held in double. <c>T = Half</c> rounds 10⁻⁶ to a
    /// subnormal (1.0133e-6, +1.3%), but the Half score is evaluated in double, so the default
    /// overloads hand it this exact value and narrow only the completed score.
    /// </summary>
    private const double DefaultEpsilonWide = 1e-6;

    /// <summary>
    /// Computes the GT-inspired variant loss for a return series with the library's default
    /// smoothing parameter <c>ε = 10⁻⁶</c> and a zero benchmark mean
    /// (<c>µ_m = 0</c>, i.e. outperformance is measured against a no-return
    /// baseline). The inclusive minimum-observation threshold is 50.
    /// </summary>
    /// <param name="returns">Return observations (at least 2 required).</param>
    /// <returns>The decomposed GT-inspired library variant result.</returns>
    /// <exception cref="ArgumentException">Fewer than 2 returns provided.</exception>
    public static GtInspiredScoreResult<T> Compute(ReadOnlySpan<T> returns)
        => ComputeCore(returns, T.Zero, DefaultEpsilon, DefaultEpsilonWide, 50);

    /// <summary>
    /// Computes the GT-inspired variant loss for a return series against a benchmark mean, using
    /// the library's default smoothing parameter <c>ε = 10⁻⁶</c> and minimum-observation threshold 50.
    /// </summary>
    /// <param name="returns">Return observations (at least 2 required).</param>
    /// <param name="benchmarkMean">Mean benchmark (buy-and-hold) return per observation, <c>µ_m</c>.</param>
    /// <returns>The decomposed GT-inspired library variant result.</returns>
    /// <exception cref="ArgumentException">Fewer than 2 returns provided.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="benchmarkMean"/> is not finite.
    /// </exception>
    public static GtInspiredScoreResult<T> Compute(ReadOnlySpan<T> returns, T benchmarkMean)
        => ComputeCore(returns, benchmarkMean, DefaultEpsilon, DefaultEpsilonWide, 50);

    /// <summary>
    /// Computes the GT-inspired variant loss for a return series using the library variant branches, after the inclusive 50-observation minimum-count safeguard:
    /// <list type="number">
    ///   <item>if <c>z ≤ 0</c> (underperforms benchmark):
    ///   <c>score = 100 + 100·(1 − e^(−|z−1|))</c> — large penalty;</item>
    ///   <item>else if <c>z ≤ 1</c> (marginal outperformance):
    ///   <c>score = 100·(1 − e^(−|z−1|))</c> — smooth transition;</item>
    ///   <item>else (<c>z &gt; 1</c>, outperforms benchmark):
    ///   <c>score = −(µ · ln(z) · r²) / (σ_d + ε)</c> — the variant utility branch.</item>
    /// </list>
    /// The result is an optimization loss (lower is better): the standard branch
    /// negates the variant utility so that a high composite utility maps to a
    /// low (strongly negative) loss, while the penalty branches yield positive
    /// values in <c>[0, 200]</c>.
    /// </summary>
    /// <param name="returns">Return observations (at least 2 required).</param>
    /// <param name="benchmarkMean">Mean benchmark (buy-and-hold) return per observation, <c>µ_m</c>.</param>
    /// <param name="epsilon">Smoothing parameter guarding the <c>σ_d</c> denominator; defaults to <c>10⁻⁶</c>.</param>
    /// <returns>The decomposed GT-inspired library variant result.</returns>
    /// <exception cref="ArgumentException">Fewer than 2 returns provided.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="benchmarkMean"/> is not finite, or <paramref name="epsilon"/>
    /// is not finite and strictly positive.
    /// </exception>
    public static GtInspiredScoreResult<T> Compute(ReadOnlySpan<T> returns, T benchmarkMean, T epsilon)
        => Compute(returns, benchmarkMean, epsilon, 50);

    /// <summary>
    /// Computes the score with an explicit inclusive minimum-observation safeguard.
    /// At or below the threshold the score is 999 − 899·N/minimumObservations;
    /// the component diagnostics retain their usual definitions.
    /// </summary>
    /// <param name="returns">At least two finite return observations.</param>
    /// <param name="benchmarkMean">Finite mean benchmark return per observation.</param>
    /// <param name="epsilon">Finite positive downside smoothing parameter.</param>
    /// <param name="minimumObservations">Positive inclusive safeguard threshold. Choose it for the sampling frequency; the default overloads use the author's fixed threshold of 50.</param>
    /// <returns>The score and component diagnostics.</returns>
    /// <exception cref="ArgumentException">Fewer than two observations or a non-finite observation.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The benchmark, epsilon or minimum-observation threshold is invalid.</exception>
    public static GtInspiredScoreResult<T> Compute(ReadOnlySpan<T> returns, T benchmarkMean, T epsilon, int minimumObservations)
        => ComputeCore(returns, benchmarkMean, epsilon, double.CreateChecked(epsilon), minimumObservations);

    /// <summary>
    /// Shared implementation. <paramref name="epsilonWide"/> is the smoothing parameter as the
    /// double-evaluated Half score must see it: the exact documented default for the default
    /// overloads, otherwise the caller's <paramref name="epsilon"/> widened. Other working types
    /// ignore it and use <paramref name="epsilon"/> directly.
    /// </summary>
    private static GtInspiredScoreResult<T> ComputeCore(
        ReadOnlySpan<T> returns, T benchmarkMean, T epsilon, double epsilonWide, int minimumObservations)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(minimumObservations, 1);

        if (returns.Length < 2)
        {
            throw new ArgumentException("At least 2 return observations are required.", nameof(returns));
        }

        if (!T.IsFinite(benchmarkMean))
        {
            throw new ArgumentOutOfRangeException(
                nameof(benchmarkMean), benchmarkMean, "Benchmark mean must be finite.");
        }

        if (!T.IsFinite(epsilon) || epsilon <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(epsilon), epsilon, "Epsilon must be finite and strictly positive.");
        }

        for (var i = 0; i < returns.Length; i++)
        {
            if (!T.IsFinite(returns[i]))
            {
                throw new ArgumentException(
                    $"Return series entry at index {i} is not finite (value {returns[i]}).",
                    nameof(returns));
            }
        }

        var n = returns.Length;
        var (mu, scale, normalizedSigma, halfMean) = ComputeMeanAndStdDev(returns);

        // Equation (2): z = (mu - mu_m) / (sigma / sqrt(N)). The standard error
        // sigma/sqrt(N) is zero only for a constant series; then there is no
        // sampling spread and z is taken as zero (the marginal/penalty branch).
        // sigma = normalizedSigma * scale is never formed in T: the absolute sample
        // SD can exceed the type range while z itself is finite (F-L1-14), so the
        // quotient is taken in units of scale.
        // sqrt(n) is computed in double and converted once rather than via
        // T.Sqrt(T.CreateChecked(n)): for T = Half, T.CreateChecked(n) itself
        // silently becomes +Infinity once n exceeds 65,504, which zeroed the
        // standard error (and therefore z) for any longer series (F-L1-03).
        // sqrt(n) is always small enough to be representable even when n is
        // not.
        var sqrtN = T.CreateChecked(Math.Sqrt(n));
        var standardError = normalizedSigma / sqrtN;
        var difference = mu - benchmarkMean;
        // Only opposite-sign subtraction can overflow for finite means. In that
        // case divide first; its two same-sign contributions cannot cancel.
        var z = standardError > T.Zero
            ? T.IsFinite(difference)
                ? difference / scale / standardError
                : (mu / scale - benchmarkMean / scale) / standardError
            : T.Zero;

        var zWide = 0.0;
        if (typeof(T) == typeof(Half))
        {
            // Form z from the unrounded double mean. A positive drift below Half's least
            // subnormal half-step rounds mu to zero, which would move z to zero and the
            // score to the penalty branch, although z itself is representable.
            var standardErrorWide = double.CreateChecked(normalizedSigma) * double.CreateChecked(scale) / Math.Sqrt(n);
            zWide = standardErrorWide > 0
                ? (halfMean - double.CreateChecked(benchmarkMean)) / standardErrorWide
                : 0;
            z = T.CreateChecked(zWide);
        }

        var rSquared = ComputeEquityCurveRSquared(returns);
        var downsideDeviation = ComputeDownsideDeviation(returns);

        if (n <= minimumObservations)
        {
            // Author's inclusive small-sample safeguard. The bounded scalar ratio
            // is evaluated in double so large counts cannot overflow Half.
            var penalty = T.CreateChecked(999.0 - 899.0 * n / minimumObservations);
            return new GtInspiredScoreResult<T>(penalty, mu, z, rSquared, downsideDeviation);
        }

        if (typeof(T) == typeof(Half))
        {
            // Widen only the reduced scalars: a finite z can exceed Half's range,
            // although its logarithm and the smoothed score still fit in Half.
            // The denominator terms are not widened from their Half roundings: the completed
            // downside RMS and the documented default epsilon are both subnormal-scale, where the
            // Half grid is coarse (up to 1.3% on the default epsilon, 29% on a small RMS), so the
            // score takes them from double and only the completed score is narrowed (F-L8-R2-LIT-01/02).
            var scoreWide = GtInspiredScore<double>.ComputePiecewiseScore(
                halfMean, zWide, double.CreateChecked(rSquared),
                ComputeHalfDownsideDeviationWide(returns), epsilonWide, Math.Log(zWide));
            return new GtInspiredScoreResult<T>(T.CreateChecked(scoreWide), mu, T.CreateChecked(zWide), rSquared, downsideDeviation);
        }

        var logZ = T.Log(z);
        if (T.IsPositiveInfinity(z) && normalizedSigma > T.Zero)
        {
            // Scale before subtracting opposite-sign means, and take logs of
            // the components before the significance quotient can overflow.
            var meanMagnitude = T.Max(T.Abs(mu), T.Abs(benchmarkMean));
            logZ = T.Log(mu / meanMagnitude - benchmarkMean / meanMagnitude) + T.Log(meanMagnitude)
                - T.Log(normalizedSigma) - T.Log(scale) + T.Log(sqrtN);
        }

        var score = ComputePiecewiseScore(mu, z, rSquared, downsideDeviation, epsilon, logZ);

        return new GtInspiredScoreResult<T>(score, mu, z, rSquared, downsideDeviation);
    }

    /// <summary>
    /// The piecewise GT-inspired library loss for the library variant.
    /// </summary>
    private static T ComputePiecewiseScore(T mu, T z, T rSquared, T downsideDeviation, T epsilon, T logZ)
    {
        var one = T.One;
        var hundred = T.CreateChecked(100);

        if (z <= T.Zero)
        {
            // Underperforms benchmark: large penalty.
            return hundred + hundred * (one - T.Exp(-T.Abs(z - one)));
        }

        if (z <= one)
        {
            // Marginal outperformance: smooth transition.
            return hundred * (one - T.Exp(-T.Abs(z - one)));
        }

        // Outperforms benchmark (z > 1): standard GT-Score, negated to a loss.
        // Combine bounded significands before applying the net power of two. Neither
        // the numerator product nor the denominator sum needs to fit separately.
        var muExponent = mu == T.Zero ? 0 : T.ILogB(T.Abs(mu));
        var logExponent = T.IsFinite(logZ) ? T.ILogB(logZ) : 0;
        var rExponent = rSquared == T.Zero ? 0 : T.ILogB(rSquared);
        var denominatorExponent = T.ILogB(T.Max(downsideDeviation, epsilon));
        var numerator = T.ScaleB(mu, -muExponent) * T.ScaleB(logZ, -logExponent)
            * T.ScaleB(rSquared, -rExponent);
        var denominator = T.ScaleB(downsideDeviation, -denominatorExponent)
            + T.ScaleB(epsilon, -denominatorExponent);
        return -T.ScaleB(numerator / denominator, muExponent + logExponent + rExponent - denominatorExponent);
    }

    /// <summary>
    /// Mean of the series and its sample standard deviation expressed in units of a positive
    /// length scale (<c>σ = NormalizedStdDev · Scale</c>). The absolute deviation is not formed
    /// because it can exceed the type's range even when every statistic built from it is finite.
    /// A constant series has zero deviation (scale <c>1</c>).
    /// </summary>
    /// <remarks>
    /// <c>HalfMean</c> is the unrounded double mean for <c>T = Half</c> (zero otherwise). A positive
    /// drift below Half's least subnormal half-step rounds <c>Mean</c> to zero, so z is formed from it.
    /// </remarks>
    private static (T Mean, T Scale, T NormalizedStdDev, double HalfMean) ComputeMeanAndStdDev(ReadOnlySpan<T> values)
    {
        var minimum = values[0];
        var maximum = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            minimum = T.Min(minimum, values[i]);
            maximum = T.Max(maximum, values[i]);
        }

        // A constant series has exactly zero spread. Detect it exactly: the summed
        // mean below misses the constant by an ulp whenever 1/N is inexact, and the
        // residual deviations would give a ~1e-18 sigma and a ~1e16 z.
        if (minimum == maximum)
        {
            return (values[0], T.One, T.Zero, typeof(T) == typeof(Half) ? double.CreateChecked(values[0]) : 0.0);
        }

        if (typeof(T) == typeof(Half))
        {
            var halfScale = T.Max(T.Abs(minimum), T.Abs(maximum));
            var scaleDouble = double.CreateChecked(halfScale);
            var positive = 0.0;
            var negative = 0.0;
            for (var i = 0; i < values.Length; i++)
            {
                var normalized = double.CreateChecked(values[i]) / scaleDouble;
                if (normalized >= 0)
                {
                    positive += normalized;
                }
                else
                {
                    negative += normalized;
                }
            }

            var normalizedMeanDouble = (positive + negative) / values.Length;
            var normalizedSumSquares = 0.0;
            for (var i = 0; i < values.Length; i++)
            {
                var difference = double.CreateChecked(values[i]) / scaleDouble - normalizedMeanDouble;
                normalizedSumSquares += difference * difference;
            }

            var meanDouble = normalizedMeanDouble * scaleDouble;
            return (
                T.CreateChecked(meanDouble),
                halfScale,
                T.CreateChecked(Math.Sqrt(normalizedSumSquares / (values.Length - 1))),
                meanDouble);
        }

        // Remove a common location before scaling: dividing raw levels by their maximum
        // rounds a large shared level into a few ulps of 1 and erases the representable
        // spread (1e16 + {0, 2, 4, 6} in double). With one sign throughout, differences from
        // the first value cannot overflow. A mixed-sign series has a spread comparable to its
        // range, so it is normalized about zero, where opposite-sign differences cannot
        // overflow either.
        var anchor = minimum >= T.Zero || maximum <= T.Zero ? values[0] : T.Zero;
        var scale = T.Max(maximum - anchor, anchor - minimum);

        var inverseCount = T.CreateChecked(1.0 / values.Length);
        var positiveSum = T.Zero;
        var negativeSum = T.Zero;
        for (var i = 0; i < values.Length; i++)
        {
            var term = ((values[i] - anchor) / scale) * inverseCount;
            if (term >= T.Zero)
            {
                positiveSum += term;
            }
            else
            {
                negativeSum += term;
            }
        }

        var normalizedOffsetMean = positiveSum + negativeSum;
        var mean = anchor + normalizedOffsetMean * scale;

        var inverseDegreesOfFreedom = T.CreateChecked(1.0 / (values.Length - 1));
        var sumSq = T.Zero;
        for (var i = 0; i < values.Length; i++)
        {
            var diff = (values[i] - anchor) / scale - normalizedOffsetMean;
            sumSq += diff * diff * inverseDegreesOfFreedom;
        }

        return (mean, scale, T.Sqrt(sumSq), 0.0);
    }

    /// <summary>
    /// R² of cumulative arithmetic returns against a linear time trend (Kestner
    /// 1996): the squared correlation between the running cumulative sum of
    /// returns and the index sequence <c>1..N</c>. A perfectly linear equity curve
    /// yields <c>r² = 1</c>; an erratic one is penalized toward 0. Returns 0 when
    /// either the time axis or the equity axis has no variance.
    /// </summary>
    private static T ComputeEquityCurveRSquared(ReadOnlySpan<T> returns)
    {
        var n = returns.Length;

        // T = Half has no widening branch here (unlike ComputeMeanAndStdDev,
        // above), so the running equity and the centered sums drifted for
        // series of a few thousand points (F-L1-02), and T.CreateChecked(n)/
        // T.CreateChecked(i + 1) silently become +Infinity once n exceeds
        // 65,504 — zeroing inverseCount and every later term (F-L1-03).
        // Mirror ComputeMeanAndStdDev's approach: run the whole reduction in
        // double and round only the final scalar r² back to T.
        if (typeof(T) == typeof(Half))
        {
            var maxAbsReturnDouble = 0.0;
            for (var i = 0; i < n; i++)
            {
                maxAbsReturnDouble = Math.Max(maxAbsReturnDouble, Math.Abs(double.CreateChecked(returns[i])));
            }

            if (maxAbsReturnDouble == 0.0)
            {
                return T.Zero;
            }

            var inverseCountDouble = 1.0 / n;
            var cumulativeDouble = 0.0;
            var equityMeanDouble = 0.0;
            var equityDouble = new double[n];
            for (var i = 0; i < n; i++)
            {
                cumulativeDouble += (double.CreateChecked(returns[i]) / maxAbsReturnDouble) * inverseCountDouble;
                equityDouble[i] = cumulativeDouble;
                equityMeanDouble += (cumulativeDouble - equityMeanDouble) / (i + 1);
            }

            var xMeanDouble = (1.0 + inverseCountDouble) / 2.0;

            var sxxDouble = 0.0;
            var syyDouble = 0.0;
            var sxyDouble = 0.0;
            for (var i = 0; i < n; i++)
            {
                var dxDouble = (i + 1) * inverseCountDouble - xMeanDouble;
                var dyDouble = equityDouble[i] - equityMeanDouble;
                sxxDouble += dxDouble * dxDouble * inverseCountDouble;
                syyDouble += dyDouble * dyDouble * inverseCountDouble;
                sxyDouble += dxDouble * dyDouble * inverseCountDouble;
            }

            if (sxxDouble <= 0.0 || syyDouble <= 0.0)
            {
                return T.Zero;
            }

            var correlationDouble = (sxyDouble / Math.Sqrt(sxxDouble)) / Math.Sqrt(syyDouble);
            return T.CreateChecked(Math.Min(1.0, correlationDouble * correlationDouble));
        }

        // Correlation is invariant to positive scaling of either axis. Keep
        // the running equity and both centered sums within the type's range
        // by normalizing returns and time before accumulating.
        //
        // Measure the equity curve from its first value. The first return only shifts every
        // equity value by one constant, which leaves the correlation unchanged, but if it is
        // large it would also set the normalization scale and round every later increment
        // away (a first return of 1e16 turns the curve's steps of 2 into noise).
        var maxAbsReturn = T.Zero;
        for (var i = 1; i < n; i++)
        {
            maxAbsReturn = T.Max(maxAbsReturn, T.Abs(returns[i]));
        }

        if (maxAbsReturn == T.Zero)
        {
            return T.Zero;
        }

        var inverseCount = T.One / T.CreateChecked(n);
        var equityMean = T.Zero;
        var cumulative = T.Zero;
        var equity = new T[n];
        for (var i = 1; i < n; i++)
        {
            cumulative += (returns[i] / maxAbsReturn) * inverseCount;
            equity[i] = cumulative;
            equityMean += (cumulative - equityMean) / T.CreateChecked(i + 1);
        }

        // Normalized time axis x = (1..N)/N; its mean is (N + 1)/(2N).
        var xMean = (T.One + inverseCount) / T.CreateChecked(2);

        var sxx = T.Zero;
        var syy = T.Zero;
        var sxy = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var dx = T.CreateChecked(i + 1) * inverseCount - xMean;
            var dy = equity[i] - equityMean;
            sxx += dx * dx * inverseCount;
            syy += dy * dy * inverseCount;
            sxy += dx * dy * inverseCount;
        }

        if (sxx <= T.Zero || syy <= T.Zero)
        {
            return T.Zero;
        }

        var correlation = (sxy / T.Sqrt(sxx)) / T.Sqrt(syy);
        return T.Min(T.One, correlation * correlation);
    }

    /// <summary>
    /// The <c>T = Half</c> downside RMS as the unrounded double the reduction produces. The score
    /// quotient takes it from here; only the reported <c>DownsideDeviation</c> field is the value
    /// narrowed once to Half.
    /// </summary>
    private static double ComputeHalfDownsideDeviationWide(ReadOnlySpan<T> returns)
    {
        var scaleDouble = 0.0;
        for (var i = 0; i < returns.Length; i++)
        {
            scaleDouble = Math.Max(scaleDouble, -Math.Min(0.0, double.CreateChecked(returns[i])));
        }

        if (scaleDouble == 0.0)
        {
            return 0.0;
        }

        var inverseCountDouble = 1.0 / returns.Length;
        var sumSqDownsideDouble = 0.0;
        for (var i = 0; i < returns.Length; i++)
        {
            var downsideDouble = Math.Min(0.0, double.CreateChecked(returns[i]) / scaleDouble);
            sumSqDownsideDouble += downsideDouble * downsideDouble * inverseCountDouble;
        }

        return Math.Sqrt(sumSqDownsideDouble) * scaleDouble;
    }

    /// <summary>
    /// Downside deviation with a zero minimum acceptable return (Sortino &amp; van
    /// der Meer 1991): the root-mean-square of the negative-side returns,
    /// <c>√(mean(min(0, rᵢ)²))</c>. Returns 0 when every observation is
    /// non-negative — the <c>ε</c> guard in <see cref="ComputePiecewiseScore"/>
    /// then keeps the denominator finite.
    /// </summary>
    private static T ComputeDownsideDeviation(ReadOnlySpan<T> returns)
    {
        // Same widening rationale as ComputeEquityCurveRSquared above
        // (F-L1-02): the per-term Half accumulation drifts for series of a
        // few thousand points even though inverseCount itself is already
        // computed safely via a double reciprocal.
        if (typeof(T) == typeof(Half))
        {
            return T.CreateChecked(ComputeHalfDownsideDeviationWide(returns));
        }

        var scale = T.Zero;
        for (var i = 0; i < returns.Length; i++)
        {
            // Positive returns contribute zero to downside RMS and must not erase
            // small losses when choosing the normalization scale.
            scale = T.Max(scale, -T.Min(T.Zero, returns[i]));
        }

        if (scale == T.Zero)
        {
            return T.Zero;
        }

        var inverseCount = T.CreateChecked(1.0 / returns.Length);
        var sumSqDownside = T.Zero;
        for (var i = 0; i < returns.Length; i++)
        {
            var downside = T.Min(T.Zero, returns[i]) / scale;
            sumSqDownside += downside * downside * inverseCount;
        }

        return T.Sqrt(sumSqDownside) * scale;
    }
}

/// <summary>
/// Result of the GT-inspired library variant, exposing its loss and component diagnostics.
/// </summary>
/// <param name="Score">The GT-inspired library loss (the piecewise definition); an optimization loss (lower is better).</param>
/// <param name="Mu">Mean strategy return per observation (<c>µ</c>).</param>
/// <param name="Z">Significance Z-score using sample standard deviation, <c>z = (µ − µ_m) / (σ / √N)</c>.</param>
/// <param name="RSquared">R² of cumulative arithmetic returns against a linear time trend (Kestner 1996).</param>
/// <param name="DownsideDeviation">Downside deviation <c>σ_d = √(mean(min(0, rᵢ)²))</c> (Sortino &amp; van der Meer 1991).</param>
public sealed record GtInspiredScoreResult(
    double Score,
    double Mu,
    double Z,
    double RSquared,
    double DownsideDeviation);

/// <summary>
/// Concrete double facade for the GT-inspired library search loss.
/// </summary>
/// <remarks>
/// <para>Delegates to <see cref="GtInspiredScore{T}"/> at <c>T = double</c>.
/// The same sample-SD, cumulative-return R², zero-target downside RMS and additive
/// epsilon definitions apply. No author parity or generalization guarantee is claimed.</para>
/// </remarks>
[Obsolete(
    "Use GtInspiredScore<double> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public static class GtInspiredScore
{
    /// <summary>
    /// Computes the GT-inspired variant loss for a return series against a benchmark mean
    /// (default <c>µ_m = 0</c>), using the library's smoothing parameter
    /// <c>ε = 10⁻⁶</c>.
    /// </summary>
    /// <param name="returns">Return observations (at least 2 required).</param>
    /// <param name="benchmarkMean">Mean benchmark (buy-and-hold) return per observation, <c>µ_m</c>.</param>
    /// <returns>The decomposed GT-inspired library variant result.</returns>
    /// <exception cref="ArgumentException">Fewer than 2 returns provided.</exception>
    public static GtInspiredScoreResult Compute(
        ReadOnlySpan<double> returns,
        double benchmarkMean = 0.0)
    {
        var result = GtInspiredScore<double>.Compute(returns, benchmarkMean);
        return new GtInspiredScoreResult(
            result.Score,
            result.Mu,
            result.Z,
            result.RSquared,
            result.DownsideDeviation);
    }
}
