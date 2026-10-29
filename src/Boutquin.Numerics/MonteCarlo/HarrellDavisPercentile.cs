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

using System.Buffers;
using System.Numerics;

using Boutquin.Numerics.Internal;

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// Harrell-Davis percentile estimator. Each order statistic is given a
/// Beta-distribution weight and the percentile is the weighted average of
/// all order statistics. Its variance relative to the linear-interpolation
/// estimator depends on the sample distribution. Positive weights on extreme
/// observations can reduce robustness for contaminated or heavy-tailed samples.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Harrell, F. E. &amp; Davis, C. E. (1982). "A New Distribution-Free
/// Quantile Estimator." Biometrika, 69(3), 635–640.
/// </para>
/// <para>
/// Weight for the i-th order statistic (1-based): w_i = I(i/n; (n+1)·p, (n+1)·(1−p)) − I((i−1)/n; (n+1)·p, (n+1)·(1−p))
/// where I is the regularized incomplete beta function. This is computed
/// via a numerical incomplete beta evaluation; for stability the weights
/// are normalized to sum to one.
/// </para>
/// <para>
/// Tier C: Beta weights are computed via double-precision arithmetic (incomplete beta
/// function using transcendental Log/Exp). Double tail masses retain a logarithmic
/// representation below 1e-280 until their products with observations are formed.
/// The final accumulation into the caller's
/// <c>T</c> is bounded by double weight precision. Weight ratios are cast from <c>double</c>
/// to <c>T</c> during the accumulation pass, except for <see cref="Half"/>, whose range
/// cannot hold small ratios: that pass runs in <c>double</c> and only the completed
/// estimate is cast to <see cref="Half"/>.
/// </para>
/// </remarks>
public static class HarrellDavisPercentile
{
    private const int StackAllocThreshold = 512;

    /// <summary>
    /// Computes the Harrell-Davis percentile of a sorted array.
    /// </summary>
    /// <param name="sorted">Sorted (ascending) sample.</param>
    /// <param name="p">Percentile in [0, 1].</param>
    /// <remarks>
    /// <para>
    /// <b>Degenerate-weight fallback.</b> The Beta weights are accumulated in
    /// <c>double</c> from the regularized incomplete beta function. For extreme
    /// shape parameters — very large <paramref name="sorted"/> with <paramref name="p"/>
    /// pushed hard into a tail — every interval weight can underflow to zero,
    /// leaving the weight sum at or below zero. In that degenerate case the
    /// estimator cannot form a weighted average and silently falls back to the
    /// linear-interpolation quantile (<see cref="Percentile.Compute{T}(ReadOnlySpan{T}, double)"/>)
    /// rather than dividing by zero. The interior open-interval cases handled
    /// here (boundaries are short-circuited above) reach this fallback only
    /// under numerical underflow, never for well-conditioned inputs.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">The incomplete-beta continued fraction does not converge within its shape-dependent iteration limit.</exception>
    public static T Compute<T>(ReadOnlySpan<T> sorted, double p)
        where T : IFloatingPoint<T>
    {
        Percentile.ValidateArguments(sorted, p);

        if (sorted.Length == 0)
        {
            return T.Zero;
        }

        if (sorted.Length == 1 || sorted[0] == sorted[^1])
        {
            return sorted[0];
        }

        if (p == 0.0)
        {
            return sorted[0];
        }

        if (p == 1.0)
        {
            return sorted[^1];
        }

        // p is in the open interval (0, 1) here — the boundary cases are handled above.
        var n = sorted.Length;
        var canAmplifyTinyMass = typeof(T) == typeof(double)
            && T.Max(T.Abs(sorted[0]), T.Abs(sorted[^1])) > T.One;

        var alpha = (n + 1) * p;
        var beta = (n + 1) * (1 - p);

        var rented = n > StackAllocThreshold ? ArrayPool<double>.Shared.Rent(n) : null;
        try
        {
            Span<double> weights = n <= StackAllocThreshold ? stackalloc double[n] : rented.AsSpan(0, n);
            var weightSum = 0.0;
            var pivot = (alpha + 1) / (alpha + beta + 2);
            var previous = 0.0;
            var upperTail = false;
            var hasScaledWeights = false;
            for (var i = 1; i <= n; i++)
            {
                var x = (double)i / n;
                var useUpperTail = x > pivot;
                // Above the distribution's mode, subtract the small complementary
                // tails directly. Subtracting cumulative values near one erases mass
                // that can still contribute to a large order statistic.
                var current = useUpperTail
                    ? RegularizedIncompleteBeta((double)(n - i) / n, beta, alpha)
                    : RegularizedIncompleteBeta(x, alpha, beta);
                var mass = useUpperTail
                    ? upperTail ? previous - current : 1 - previous - current
                    : current - previous;
                weights[i - 1] = Math.Max(0.0, mass);
                weightSum += weights[i - 1];
                if (canAmplifyTinyMass && weights[i - 1] < 1e-280)
                {
                    // A subnormal (or zero) probability can still have a normal
                    // product with a large observation. Negative slots encode log
                    // masses; ordinary slots retain their existing direct weights.
                    weights[i - 1] = LogIntervalMass(i, n, alpha, beta, pivot);
                    hasScaledWeights = true;
                }
                previous = current;
                upperTail = useUpperTail;
            }

            if (weightSum <= 0)
            {
                // Degenerate weights (all interval masses underflowed to zero for an
                // extreme tail percentile): fall back to the linear quantile. See the
                // method <remarks> for the trigger condition.
                return Percentile.Compute(sorted, p);
            }

            if (hasScaledWeights)
            {
                return T.CreateChecked(ComputeScaledDouble(sorted, weights));
            }

            if (typeof(T) == typeof(Half))
            {
                // A nonzero weight ratio can sit far below Half's least subnormal
                // (about 6e-8) while its product with the sample gap is a normal Half.
                // Keep the ratios in double and round the completed estimate once.
                var estimate = double.CreateChecked(sorted[0]);
                var cumulative = 0.0;
                for (var i = 0; i < n; i++)
                {
                    if (weights[i] <= 0.0)
                    {
                        continue;
                    }

                    var previousWeight = cumulative;
                    cumulative += weights[i];
                    var value = double.CreateChecked(sorted[i]);
                    estimate = weights[i] <= previousWeight
                        ? estimate + weights[i] / cumulative * (value - estimate)
                        : value + previousWeight / cumulative * (estimate - value);
                }

                return T.CreateChecked(estimate);
            }

            // Incremental convex interpolation preserves constants even when individually
            // converted normalized weights would not sum to exactly one in T.
            T acc = sorted[0];
            var cumulativeWeight = 0.0;
            for (var i = 0; i < n; i++)
            {
                if (weights[i] <= 0.0)
                {
                    continue;
                }

                var previousWeight = cumulativeWeight;
                cumulativeWeight += weights[i];
                var newFraction = weights[i] / cumulativeWeight;
                var oldFraction = previousWeight / cumulativeWeight;
                var value = sorted[i];
                // Apply the smaller share to a difference. In particular, a tiny
                // old mass must not become (1 - rounded-to-one newFraction).
                acc = acc < T.Zero && value > T.Zero
                    ? WeightedValue(acc, oldFraction) + WeightedValue(value, newFraction)
                    : weights[i] <= previousWeight
                        ? acc + WeightedValue(value - acc, newFraction)
                        : value + WeightedValue(acc - value, oldFraction);
            }

            return acc;
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    // The scaled path is only needed by double: even decimal.MaxValue cannot
    // amplify a probability below 1e-280 to decimal's 1e-28 resolution.
    private static double ComputeScaledDouble<T>(ReadOnlySpan<T> sorted, ReadOnlySpan<double> weights)
        where T : IFloatingPoint<T>
    {
        var estimate = double.CreateChecked(sorted[0]);
        var cumulative = 0.0;
        long cumulativeExponent = 0;
        for (var i = 0; i < weights.Length; i++)
        {
            var weight = weights[i];
            if (weight == 0.0 || double.IsNegativeInfinity(weight))
            {
                continue;
            }

            long exponent;
            double significand;
            if (weight < 0.0)
            {
                exponent = (long)Math.Floor(weight / Math.Log(2.0));
                significand = Math.Exp(double.FusedMultiplyAdd(-exponent, Math.Log(2.0), weight));
            }
            else
            {
                exponent = Math.ILogB(weight);
                significand = Math.ScaleB(weight, -(int)exponent);
            }

            var previous = cumulative;
            var previousExponent = cumulativeExponent;
            var commonExponent = previous == 0.0 ? exponent : Math.Max(exponent, previousExponent);
            var sum = Math.ScaleB(significand, int.CreateSaturating(exponent - commonExponent))
                + Math.ScaleB(previous, int.CreateSaturating(previousExponent - commonExponent));
            var shift = Math.ILogB(sum);
            cumulative = Math.ScaleB(sum, -shift);
            cumulativeExponent = commonExponent + shift;

            var value = double.CreateChecked(sorted[i]);
            double Weighted(double argument, double share, long shareExponent) =>
                ScaledArithmetic<double>.ProductQuotient(
                    argument, share, 1.0, cumulative, shareExponent - cumulativeExponent);
            var newIsSmaller = exponent < previousExponent
                || (exponent == previousExponent && significand <= previous);
            estimate = estimate < 0.0 && value > 0.0
                ? Weighted(estimate, previous, previousExponent) + Weighted(value, significand, exponent)
                : previous != 0.0 && newIsSmaller
                    ? estimate + Weighted(value - estimate, significand, exponent)
                    : value + Weighted(estimate - value, previous, previousExponent);
        }

        return estimate;
    }

    private static double LogIntervalMass(int index, int count, double alpha, double beta, double pivot)
    {
        var lower = (double)(index - 1) / count;
        var upper = (double)index / count;
        double larger;
        double smaller;
        if (upper <= pivot)
        {
            larger = LogLowerBeta(upper, alpha, beta);
            smaller = LogLowerBeta(lower, alpha, beta);
        }
        else
        {
            larger = LogLowerBeta((double)(count - index + 1) / count, beta, alpha);
            smaller = LogLowerBeta((double)(count - index) / count, beta, alpha);
        }

        return larger + Math.Log(-double.ExpM1(smaller - larger));
    }

    private static double LogLowerBeta(double x, double a, double b) => x <= 0.0
        ? double.NegativeInfinity
        : LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x)
            + b * double.LogP1(-x) + Math.Log(BetaContinuedFraction(x, a, b)) - Math.Log(a);

    private static T WeightedValue<T>(T value, double fraction) where T : IFloatingPoint<T>
    {
        if (typeof(T) == typeof(decimal) && fraction > 0 && fraction < 1e-14)
        {
            return T.CreateChecked(Percentile.ScaleByFraction(decimal.CreateChecked(value), fraction));
        }
        return value * T.CreateChecked(fraction);
    }

    private static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0.0)
        {
            return 0.0;
        }

        if (x >= 1.0)
        {
            return 1.0;
        }

        if (a == b && x == 0.5)
        {
            return 0.5;
        }

        var bt = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b)
                          + a * Math.Log(x) + b * double.LogP1(-x));

        if (x < (a + 1.0) / (a + b + 2.0))
        {
            return bt * BetaContinuedFraction(x, a, b) / a;
        }

        return 1.0 - bt * BetaContinuedFraction(1.0 - x, b, a) / b;
    }

    private static double BetaContinuedFraction(double x, double a, double b)
    {
        var maxIter = Math.Max(200, (int)Math.Ceiling(8 * Math.Sqrt(a + b)));
        const double eps = 2e-15;
        const double fpmin = 1e-300;

        var qab = a + b;
        var qap = a + 1.0;
        var qam = a - 1.0;
        var c = 1.0;
        var d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < fpmin)
        {
            d = fpmin;
        }

        d = 1.0 / d;
        var h = d;

        for (var m = 1; m <= maxIter; m++)
        {
            var m2 = 2 * m;
            var aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < fpmin)
            {
                d = fpmin;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < fpmin)
            {
                c = fpmin;
            }

            d = 1.0 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < fpmin)
            {
                d = fpmin;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < fpmin)
            {
                c = fpmin;
            }

            d = 1.0 / d;
            var del = d * c;
            h *= del;
            if (Math.Abs(del - 1.0) < eps)
            {
                return h;
            }
        }

        throw new InvalidOperationException("Incomplete beta continued fraction did not converge.");
    }

    private static double LogGamma(double x)
    {
        // Gamma recurrence followed by Stirling's Bernoulli expansion
        // (NIST DLMF 5.11.1). At x >= 16 the first omitted term is below 3e-20.
        var correction = 0.0;
        while (x < 16.0)
        {
            correction -= Math.Log(x);
            x += 1.0;
        }

        var inverse = 1.0 / x;
        var square = inverse * inverse;
        var series = inverse * (1.0 / 12.0 + square * (-1.0 / 360.0
            + square * (1.0 / 1260.0 + square * (-1.0 / 1680.0
            + square * (1.0 / 1188.0 + square * (-691.0 / 360360.0 + square / 156.0))))));
        return (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2.0 * Math.PI) + series + correction;
    }
}
