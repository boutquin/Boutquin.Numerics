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

using Boutquin.Numerics.Internal;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Shared implementation primitives for covariance estimators. All estimators
/// in this namespace consume returns in T-by-N layout (rows = observations,
/// columns = assets) and produce an N-by-N symmetric covariance matrix.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
internal static class CovarianceHelpers<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Fits the MP noise variance to a Gaussian KDE (López de Prado 2020,
    /// snippet 2.4). Only model selection runs in double; no fitted value is
    /// used in covariance reconstruction.
    /// </summary>
    /// <remarks>
    /// Returns the deterministic global minimizer of the declared objective (1000-point MP density against
    /// a bandwidth-0.01 Gaussian KDE of the spectrum) over [1e-5, 1 - 1e-5]: a uniform scan, then golden-section
    /// refinement of every scan-local minimum, then the lowest refined objective. The objective varies with
    /// the variance on a scale of at most <c>bandwidth / upper-edge</c>, because the KDE is evaluated at
    /// <c>variance * x</c> for abscissae x up to the upper MP edge. The scan therefore uses a step of half that
    /// scale (never fewer than 256 intervals), which keeps every minimum whose basin is wider than about two
    /// steps from falling between scan points. The step is capped at <see cref="MaximumScanIntervals"/>
    /// intervals (a step of 6e-5): for an upper edge above about 80 (T / N below about 0.015) the cap is
    /// coarser than half the smallest feature, and a minimum narrower than two steps can still be missed.
    /// </remarks>
    internal static double FitNoiseVariance(T[] eigenvalues, double q)
    {
        const int points = 1000;
        const int minimumIntervals = 256;
        const double lower = 1e-5;
        const double upper = 1 - lower;
        const double bandwidth = 0.01;
        // exp(-z^2 / 2) is exactly zero in double beyond |z| = 38.6, so terms past 40 bandwidths are skipped.
        const double cutoff = 40 * bandwidth;
        var spectrum = new double[eigenvalues.Length];
        for (var i = 0; i < spectrum.Length; i++)
        {
            spectrum[i] = double.CreateChecked(eigenvalues[i]);
        }

        var root = Math.Sqrt(1 / q);
        var left = (1 - root) * (1 - root);
        var right = (1 + root) * (1 + root);
        // At q=1 the book's endpoint PDF is 0/0. Preserve its failed-fit fallback.
        if (left == 0)
        {
            return 1;
        }

        var abscissae = new double[points];
        var density = new double[points];
        for (var i = 0; i < points; i++)
        {
            var x = left + (right - left) * i / (points - 1);
            abscissae[i] = x;
            density[i] = q * Math.Sqrt(Math.Max(0, (right - x) * (x - left))) / (2 * Math.PI * x);
        }

        // The abscissae x_i * variance increase with i, so the eigenvalues within `cutoff` of the evaluation
        // point form a window of the ascending spectrum whose two ends only move forward.
        var ascending = (double[])spectrum.Clone();
        Array.Sort(ascending);
        var normalization = spectrum.Length * bandwidth * Math.Sqrt(2 * Math.PI);

        double Objective(double variance)
        {
            var error = 0.0;
            var windowStart = 0;
            var windowEnd = 0;
            for (var i = 0; i < points; i++)
            {
                var x = variance * abscissae[i];
                while (windowStart < ascending.Length && ascending[windowStart] < x - cutoff)
                {
                    windowStart++;
                }

                windowEnd = Math.Max(windowEnd, windowStart);
                while (windowEnd < ascending.Length && ascending[windowEnd] <= x + cutoff)
                {
                    windowEnd++;
                }

                // Descending eigenvalue order, the order a Jacobi spectrum is supplied in.
                var empirical = 0.0;
                for (var j = windowEnd - 1; j >= windowStart; j--)
                {
                    var z = (x - ascending[j]) / bandwidth;
                    empirical += Math.Exp(-0.5 * z * z);
                }

                empirical /= normalization;
                var difference = density[i] / variance - empirical;
                error += difference * difference;
            }

            return error;
        }

        var intervals = (int)Math.Clamp(
            Math.Ceiling((upper - lower) * 2 * right / bandwidth), minimumIntervals, MaximumScanIntervals);
        var scores = new double[intervals + 1];
        var step = (upper - lower) / intervals;
        var best = 1.0;
        var bestScore = double.PositiveInfinity;
        for (var i = 0; i <= intervals; i++)
        {
            var variance = lower + step * i;
            scores[i] = Objective(variance);
            if (scores[i] < bestScore)
            {
                bestScore = scores[i];
                best = variance;
            }
        }

        // Refine every grid-local minimum, including endpoint brackets. This
        // avoids choosing whichever basin a single initial guess happens to find.
        const double ratio = 0.6180339887498949;
        for (var i = 0; i <= intervals; i++)
        {
            if ((i > 0 && scores[i] > scores[i - 1]) ||
                (i < intervals && scores[i] > scores[i + 1]))
            {
                continue;
            }

            var a = lower + step * Math.Max(0, i - 1);
            var b = lower + step * Math.Min(intervals, i + 1);
            var c = b - ratio * (b - a);
            var d = a + ratio * (b - a);
            var fc = Objective(c);
            var fd = Objective(d);
            for (var iteration = 0; iteration < 64 && b - a > 1e-10; iteration++)
            {
                if (fc < fd)
                {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - ratio * (b - a);
                    fc = Objective(c);
                }
                else
                {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + ratio * (b - a);
                    fd = Objective(d);
                }
            }

            var candidate = fc < fd ? c : d;
            var score = Math.Min(fc, fd);
            if (score < bestScore)
            {
                best = candidate;
                bestScore = score;
            }
        }

        return double.IsFinite(bestScore) ? best : 1;
    }

    /// <summary>Upper bound on the scan resolution of <see cref="FitNoiseVariance"/>.</summary>
    private const int MaximumScanIntervals = 16384;

    /// <summary>
    /// Validates a T-by-N return matrix. Throws <see cref="ArgumentException"/>
    /// when the matrix has fewer than two observations or fewer than one asset.
    /// </summary>
    internal static void ValidateReturns(T[,] returns)
    {
        ArgumentNullException.ThrowIfNull(returns);
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        if (n < 1)
        {
            throw new ArgumentException("Returns matrix must contain at least one asset.", nameof(returns));
        }

        if (t < 2)
        {
            throw new ArgumentException("Returns matrix must contain at least two observations.", nameof(returns));
        }

        // Fail fast on non-finite inputs: a single NaN/Infinity otherwise propagates
        // silently through means, covariance, and any downstream eigendecomposition.
        // This is one extra O(t·n) pass, the same order as the mean computation.
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (!T.IsFinite(returns[i, j]))
                {
                    throw new ArgumentException(
                        $"Returns matrix must contain only finite values; found a non-finite entry at [{i}, {j}].",
                        nameof(returns));
                }
            }
        }
    }

    /// <summary>
    /// Smallest dominant fourth-order term for which the unnormalized path keeps the
    /// fourth-moment terms of the shrinkage intensities at working precision. Below it the
    /// terms sit near the type's absolute resolution (decimal's 1e-28 grid, the subnormal
    /// range of float and double), where their relative precision is lost silently.
    /// </summary>
    /// <remarks>
    /// IEEE types: the smallest normal value times 2^(p/2 + 1), with p the significand width
    /// (float 1e-34, double 1e-299). Decimal: 1e-14, the resolution 1e-28 times the square root of
    /// its 28-digit unit roundoff. Each floor leaves a dominant fourth-order term half of the
    /// significand's digits of headroom above the resolution.
    /// </remarks>
    private static readonly T s_fourthPowerFloor = typeof(T) == typeof(decimal)
        ? T.CreateTruncating(1e-14)
        : typeof(T) == typeof(float)
            ? T.CreateTruncating(1e-34)
            : T.CreateTruncating(1e-299);

    /// <summary>
    /// Keeps fourth-order shrinkage-intensity terms representable while leaving
    /// ordinary-scale inputs on their original arithmetic path.
    /// </summary>
    /// <remarks>
    /// The path is normalized when the aggregate fourth-order bound overflows or a dominant term falls below
    /// <see cref="s_fourthPowerFloor"/>: dividing by the largest magnitude is exact scaling for the
    /// scale-equivariant shrinkage estimators, and scaling up a tiny panel restores the digits the
    /// absolute-resolution types would otherwise round away.
    /// </remarks>
    internal static (T[,] Returns, T Scale) NormalizeForShrinkage(T[,] returns)
    {
        // Fourth moments and the market factor must use the same relative
        // coordinates as sample covariance, even when scaling is unnecessary.
        returns = ShiftToAnchor(returns);
        var scale = T.Zero;
        foreach (var value in returns)
        {
            scale = T.Max(scale, T.Abs(value));
        }

        if (scale == T.Zero)
        {
            return (returns, T.One);
        }

        try
        {
            var square = scale * scale;
            var fourthPower = square * square;
            // Individual fourth powers may fit while the time/asset reductions do not.
            var aggregateBound = fourthPower * T.CreateChecked(32)
                * T.CreateChecked(returns.GetLength(0))
                * T.CreateChecked(returns.GetLength(1))
                * T.CreateChecked(returns.GetLength(1));
            if (T.IsFinite(aggregateBound) && fourthPower >= s_fourthPowerFloor)
            {
                return (returns, T.One);
            }
        }
        catch (OverflowException)
        {
            // Decimal signals overflow with an exception rather than infinity.
        }

        for (var i = 0; i < returns.GetLength(0); i++)
        {
            for (var j = 0; j < returns.GetLength(1); j++)
            {
                returns[i, j] /= scale;
            }
        }

        return (returns, scale);
    }

    /// <summary>
    /// Three-factor product that stays exact where a left-to-right product would not: when the plain
    /// product and its first partial product are normal it is returned unchanged (bit-identical to
    /// <c>a * b * c</c>), otherwise the factors are multiplied in magnitude-balanced order.
    /// </summary>
    internal static T ScaleSafeProduct(T a, T b, T c)
    {
        try
        {
            var partial = a * b;
            if (T.IsNormal(partial))
            {
                var product = partial * c;
                if (T.IsNormal(product))
                {
                    return product;
                }
            }
        }
        catch (OverflowException)
        {
            // Decimal signals overflow with an exception; the balanced order below cannot overflow
            // unless the exact product does.
        }

        return ScaleSafeProductOrdered(a, b, c);
    }

    /// <summary>
    /// Multiplies the factors in an order that keeps the running product near one in magnitude:
    /// while the product is at least one it takes the smallest remaining factor, otherwise the
    /// largest. A left-to-right product can overflow or underflow in an intermediate even when the
    /// exact result and every factor are representable (for example <c>2^-700 . 2^-700 . 2^1000</c>);
    /// this order cannot, because the running product always moves back toward one.
    /// </summary>
    /// <remarks>At most 64 factors. A zero factor returns zero before any product is formed.</remarks>
    internal static T ScaleSafeProductOrdered(params ReadOnlySpan<T> factors)
    {
        foreach (var factor in factors)
        {
            if (factor == T.Zero)
            {
                return T.Zero;
            }
        }

        ulong taken = 0;
        var product = T.One;
        for (var step = 0; step < factors.Length; step++)
        {
            var wantLarge = T.Abs(product) < T.One;
            var pick = -1;
            var pickMagnitude = T.Zero;
            for (var i = 0; i < factors.Length; i++)
            {
                if ((taken & (1UL << i)) != 0)
                {
                    continue;
                }

                var magnitude = T.Abs(factors[i]);
                if (pick < 0 || (wantLarge ? magnitude > pickMagnitude : magnitude < pickMagnitude))
                {
                    pick = i;
                    pickMagnitude = magnitude;
                }
            }

            taken |= 1UL << pick;
            product *= factors[pick];
        }

        return product;
    }

    /// <summary>Removes per-column locations without reconstructing rounded absolute means.</summary>
    internal static T[,] ShiftToAnchor(T[,] returns)
    {
        var shifted = new T[returns.GetLength(0), returns.GetLength(1)];
        for (var j = 0; j < shifted.GetLength(1); j++)
        {
            var anchor = returns[0, j];
            for (var i = 1; i < shifted.GetLength(0); i++)
            {
                if (T.IsNegative(returns[i, j]) != T.IsNegative(anchor))
                {
                    // Opposite endpoint subtraction could overflow.
                    anchor = T.Zero;
                    break;
                }
            }

            for (var i = 0; i < shifted.GetLength(0); i++)
            {
                shifted[i, j] = returns[i, j] - anchor;
            }
        }

        return shifted;
    }

    /// <summary>
    /// Evaluates Half shrinkage with exact input widening, representable observation
    /// counts and fourth-moment reductions, rounding only the returned covariance.
    /// </summary>
    internal static T[,] EstimateHalfShrinkage(T[,] returns, ICovarianceEstimator<double> estimator)
    {
        var widened = new double[returns.GetLength(0), returns.GetLength(1)];
        for (var i = 0; i < widened.GetLength(0); i++)
        {
            for (var j = 0; j < widened.GetLength(1); j++)
            {
                widened[i, j] = double.CreateChecked(returns[i, j]);
            }
        }

        var covariance = estimator.Estimate(widened);
        var result = new T[covariance.GetLength(0), covariance.GetLength(1)];
        for (var i = 0; i < result.GetLength(0); i++)
        {
            for (var j = 0; j < result.GetLength(1); j++)
            {
                result[i, j] = T.CreateChecked(covariance[i, j]);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes per-asset arithmetic means down the time axis.
    /// </summary>
    internal static T[] ComputeMeans(T[,] returns)
    {
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var tCount = T.CreateChecked(t);
        var means = new T[n];
        if (typeof(T) == typeof(Half))
        {
            // Half cannot represent observation counts above 65,504, and an anchored
            // deviation sum overflows Half.MaxValue when one row sits far from the rest
            // even though every input and the mean are representable. Widen the
            // reduction (Half-to-double conversion is exact), then round only the final mean.
            for (var j = 0; j < n; j++)
            {
                var sum = 0.0;
                for (var i = 0; i < t; i++)
                {
                    sum += double.CreateChecked(returns[i, j]);
                }

                means[j] = T.CreateChecked(sum / t);
            }

            return means;
        }

        for (var j = 0; j < n; j++)
        {
            var anchor = returns[0, j];
            var deviationSum = T.Zero;
            for (var i = 1; i < t; i++)
            {
                deviationSum += returns[i, j] - anchor;
            }

            means[j] = anchor + deviationSum / tCount;
        }

        return means;
    }

    /// <summary>
    /// Computes the unbiased sample covariance matrix (observation-count-minus-one divisor).
    /// </summary>
    internal static T[,] ComputeSampleCovariance(T[,] returns, T[] means)
    {
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        if (typeof(T) == typeof(Half))
        {
            return EstimateHalfShrinkage(returns, new SampleCovarianceEstimator<double>());
        }

        // Keep the mean residual separate from the large absolute level. Adding
        // the anchor back would round away representable centered differences.
        var centered = new T[t, n];
        var shifted = false;
        for (var j = 0; j < n; j++)
        {
            var anchor = returns[0, j];
            for (var k = 1; k < t; k++)
            {
                if (T.IsNegative(returns[k, j]) != T.IsNegative(anchor))
                {
                    anchor = T.Zero;
                    break;
                }
            }

            shifted |= anchor != T.Zero;
            for (var k = 0; k < t; k++)
            {
                centered[k, j] = returns[k, j] - anchor;
            }
        }

        returns = centered;
        if (shifted)
        {
            means = ComputeMeans(centered);
        }
        var divisor = T.CreateChecked(t - 1);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var value = ComputeCovarianceEntry(returns, means, i, j, t, divisor);
                cov[i, j] = value;
                cov[j, i] = value;
            }
        }

        return cov;
    }

    private static T ComputeCovarianceEntry(T[,] returns, T[] means, int i, int j, int t, T divisor)
    {
        try
        {
            var sum = T.Zero;
            var underflow = false;
            for (var k = 0; k < t; k++)
            {
                var left = returns[k, i] - means[i];
                var right = returns[k, j] - means[j];
                var product = left * right;
                underflow |= T.IsSubnormal(product) || (product == T.Zero && left != T.Zero && right != T.Zero);
                sum += product;
            }

            if (T.IsFinite(sum))
            {
                if (underflow)
                {
                    var leftScale = T.Zero;
                    var rightScale = T.Zero;
                    for (var k = 0; k < t; k++)
                    {
                        leftScale = T.Max(leftScale, T.Abs(returns[k, i] - means[i]));
                        rightScale = T.Max(rightScale, T.Abs(returns[k, j] - means[j]));
                    }

                    var normalizedSum = T.Zero;
                    for (var k = 0; k < t; k++)
                    {
                        normalizedSum += ((returns[k, i] - means[i]) / leftScale)
                            * ((returns[k, j] - means[j]) / rightScale);
                    }

                    return (normalizedSum / divisor * leftScale) * rightScale;
                }

                return sum / divisor;
            }
        }
        catch (OverflowException)
        {
            // Fall back below: the final covariance can be representable even when
            // the pre-division centered-product sum is not.
        }

        var scaled = T.Zero;
        for (var k = 0; k < t; k++)
        {
            scaled += ((returns[k, i] - means[i]) / divisor) * (returns[k, j] - means[j]);
        }

        return scaled;
    }

    /// <summary>
    /// Converts a covariance matrix to a correlation matrix and returns the
    /// per-asset standard deviations as a side product.
    /// </summary>
    /// <remarks>
    /// Zero-variance assets (<c>sigma_i = 0</c>) are handled by writing <c>1</c> on the
    /// diagonal and <c>0</c> elsewhere for that asset's row/column — preserving
    /// the identity-like structure for a constant series rather than propagating
    /// <c>NaN</c> via <c>0 / 0</c>.
    /// </remarks>
    internal static (T[,] Correlation, T[] StdDevs) CovarianceToCorrelation(T[,] cov)
    {
        var n = cov.GetLength(0);
        var stdDevs = new T[n];
        for (var i = 0; i < n; i++)
        {
            stdDevs[i] = NumericPrecision<T>.Sqrt(cov[i, i]);
        }

        var corr = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (stdDevs[i] == T.Zero || stdDevs[j] == T.Zero)
                {
                    corr[i, j] = i == j ? T.One : T.Zero;
                }
                else
                {
                    corr[i, j] = T.Clamp(cov[i, j] / (stdDevs[i] * stdDevs[j]), -T.One, T.One);
                }
            }
        }

        return (corr, stdDevs);
    }

    /// <summary>
    /// Sample correlation matrix and per-asset standard deviations of a returns panel, formed
    /// column by column at a scale that keeps each column's variance inside the type's range.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="CovarianceToCorrelation"/> takes <c>sigma_i</c> from the sample-covariance
    /// diagonal, so a column whose variance underflows reads as a constant series and loses
    /// every correlation although <c>sigma_i</c>, its correlations and its off-diagonal
    /// covariances are representable. For binary IEEE types (<see cref="double"/>,
    /// <see cref="float"/>, <see cref="Half"/>) each column is first multiplied by the power of
    /// two that brings its largest magnitude into <c>[1, 2)</c>; the correlation is formed from
    /// that panel and each standard deviation is scaled back by the same power of two.
    /// </para>
    /// <para>
    /// Power-of-two scaling is exact, so whenever the unscaled computation stays in the normal
    /// range the correlation and the standard deviations are bit-identical to
    /// <see cref="CovarianceToCorrelation"/> applied to the sample covariance.
    /// </para>
    /// <para>
    /// <see cref="decimal"/> has an absolute <c>1e-28</c> grid, so a column of magnitude about
    /// <c>1e-14</c> has a variance that rounds to zero while its correlations are representable.
    /// Each decimal column is therefore first shifted by an anchor (its first value, or zero when
    /// the column changes sign) and then multiplied by the power of ten that brings its largest
    /// anchored magnitude into <c>[0.1, 1)</c> (<see cref="ScaledArithmetic{T}"/>); the correlation
    /// is formed from that panel, and each standard deviation is scaled back by the same power of
    /// ten. Taking the exponent from the anchored column keeps the scaling translation invariant: a
    /// column far above its own spread (for example <c>1e13 + 0.01 z</c>) is scaled by its spread,
    /// not its level, so its centred products stay above the grid. The scaling is exact unless a
    /// value has more than 28 fractional digits after it. Other types keep the unscaled computation.
    /// </para>
    /// </remarks>
    internal static (T[,] Correlation, T[] StdDevs) SampleCorrelation(T[,] returns)
    {
        if (typeof(T) == typeof(decimal))
        {
            return SampleCorrelationPerColumnScaled(returns);
        }

        if (typeof(T) != typeof(double) && typeof(T) != typeof(float) && typeof(T) != typeof(Half))
        {
            return CovarianceToCorrelation(ComputeSampleCovariance(returns, ComputeMeans(returns)));
        }

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var exponents = new int[n];
        var scaled = new T[t, n];
        for (var j = 0; j < n; j++)
        {
            var largest = T.Zero;
            for (var k = 0; k < t; k++)
            {
                largest = T.Max(largest, T.Abs(returns[k, j]));
            }

            exponents[j] = largest == T.Zero ? 0 : BinaryExponent(largest);
            for (var k = 0; k < t; k++)
            {
                scaled[k, j] = ScaleByPowerOfTwo(returns[k, j], -exponents[j]);
            }
        }

        var (corr, scaledStdDevs) = CovarianceToCorrelation(ComputeSampleCovariance(scaled, ComputeMeans(scaled)));
        var stdDevs = new T[n];
        for (var j = 0; j < n; j++)
        {
            stdDevs[j] = ScaleByPowerOfTwo(scaledStdDevs[j], exponents[j]);
        }

        return (corr, stdDevs);
    }

    // Radix-exponent column scaling through ScaledArithmetic (decimal callers): each column is first shifted by
    // the same anchor ComputeSampleCovariance uses (its first value, or zero for a mixed-sign column), so the
    // exponent follows the column's spread rather than its level; correlations are formed on anchored columns
    // whose largest magnitude lies in [0.1, 1), and each standard deviation is restored exactly once.
    private static (T[,] Correlation, T[] StdDevs) SampleCorrelationPerColumnScaled(T[,] returns)
    {
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var exponents = new int[n];
        var column = new T[t];
        var scaled = new T[t, n];
        for (var j = 0; j < n; j++)
        {
            var anchor = returns[0, j];
            for (var k = 1; k < t; k++)
            {
                if (T.IsNegative(returns[k, j]) != T.IsNegative(anchor))
                {
                    anchor = T.Zero;
                    break;
                }
            }

            for (var k = 0; k < t; k++)
            {
                column[k] = returns[k, j] - anchor;
            }

            exponents[j] = ScaledArithmetic<T>.ScaleExponent(column);
            for (var k = 0; k < t; k++)
            {
                scaled[k, j] = ScaledArithmetic<T>.Rescale(column[k], -exponents[j]);
            }
        }

        var (corr, scaledStdDevs) = CovarianceToCorrelation(ComputeSampleCovariance(scaled, ComputeMeans(scaled)));
        var stdDevs = new T[n];
        for (var j = 0; j < n; j++)
        {
            stdDevs[j] = ScaledArithmetic<T>.Rescale(scaledStdDevs[j], exponents[j]);
        }

        return (corr, stdDevs);
    }

    // Unbiased binary exponent of a finite, nonzero binary IEEE value (callers dispatch on T first).
    private static int BinaryExponent(T value)
    {
        if (typeof(T) == typeof(double))
        {
            return Math.ILogB(double.CreateTruncating(value));
        }

        return typeof(T) == typeof(float)
            ? MathF.ILogB(float.CreateTruncating(value))
            : Half.ILogB(Half.CreateTruncating(value));
    }

    // Exact multiplication by 2^exponent for binary IEEE T (rounded only where the result is subnormal).
    private static T ScaleByPowerOfTwo(T value, int exponent)
    {
        if (typeof(T) == typeof(double))
        {
            return T.CreateTruncating(Math.ScaleB(double.CreateTruncating(value), exponent));
        }

        return typeof(T) == typeof(float)
            ? T.CreateTruncating(MathF.ScaleB(float.CreateTruncating(value), exponent))
            : T.CreateTruncating(Half.ScaleB(Half.CreateTruncating(value), exponent));
    }

    /// <summary>
    /// Converts a correlation matrix back to covariance using per-asset standard deviations.
    /// </summary>
    internal static T[,] CorrelationToCovariance(T[,] corr, T[] stdDevs)
    {
        var n = corr.GetLength(0);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                cov[i, j] = corr[i, j] * stdDevs[i] * stdDevs[j];
                cov[j, i] = cov[i, j];
            }
        }

        return cov;
    }

    /// <summary>
    /// Marcenko-Pastur upper spectral edge for the empirical
    /// eigenvalue distribution of a sample correlation matrix under the null
    /// hypothesis of IID noise, where <c>q = T / N</c> is the inverse
    /// concentration ratio.
    /// </summary>
    /// <remarks>
    /// Eigenvalues below the upper bound are indistinguishable from noise in the asymptotic
    /// limit; eigenvalues above it carry signal. Used as the threshold in
    /// <see cref="DenoisedCovarianceEstimator"/> and
    /// <see cref="DetonedCovarianceEstimator"/>. For finite-sample refinement,
    /// <see cref="TracyWidomDenoisedCovarianceEstimator"/> adds an N^(-2/3)
    /// correction. Returns the sentinel T.CreateTruncating(1e15) when <c>q = 0</c>.
    /// This is 1e15 for double and decimal, and positive infinity for Half;
    /// the zero-observation threshold has no statistical interpretation.
    /// </remarks>
    internal static T MarcenkoPasturUpperBound(T q)
    {
        var sqrtQ = NumericPrecision<T>.Sqrt(q);
        if (sqrtQ == T.Zero)
        {
            // q = 0 means no observations — threshold is meaningless; return a sentinel
            // large enough that no eigenvalue of a finite correlation matrix exceeds it.
            // CreateTruncating (rather than CreateChecked) keeps this defensive path from
            // throwing on narrow-range T (e.g. Half, whose max is ~65504); the sentinel
            // saturates instead, which preserves the "exceeds any eigenvalue" intent.
            return T.CreateTruncating(1e15);
        }

        var bound = T.One + T.One / sqrtQ;
        return bound * bound;
    }

    /// <summary>
    /// Reconstructs a symmetric matrix from its eigendecomposition
    /// <c>A = V . diag(lambda) . V^T</c>, given eigenvalues and column-eigenvectors.
    /// </summary>
    /// <remarks>
    /// Symmetry is enforced by computing each entry in the upper triangle once
    /// and assigning the symmetric partner directly, rather than computing
    /// both entries independently and relying on floating-point
    /// associativity to produce identical values. No eigenvalue ordering is
    /// assumed by the reconstruction; the caller controls sign/magnitude
    /// modifications (clipping, shrinkage, detoning) before invoking.
    /// </remarks>
    internal static T[,] ReconstructFromEigen(T[] values, T[,] vectors)
    {
        var n = values.Length;
        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < n; k++)
                {
                    sum += values[k] * vectors[i, k] * vectors[j, k];
                }

                result[i, j] = sum;
                result[j, i] = sum;
            }
        }

        return result;
    }
}

/// <summary>
/// Legacy decimal facade for <see cref="CovarianceHelpers{T}"/>. Delegates all
/// operations to <see cref="CovarianceHelpers{T}"/> with <c>T = decimal</c>.
/// Existing estimators that reference <c>CovarianceHelpers.Method()</c> continue
/// to compile unchanged.
/// </summary>
internal static class CovarianceHelpers
{
    /// <inheritdoc cref="CovarianceHelpers{T}.ValidateReturns"/>
    internal static void ValidateReturns(decimal[,] returns)
        => CovarianceHelpers<decimal>.ValidateReturns(returns);

    /// <inheritdoc cref="CovarianceHelpers{T}.ComputeMeans"/>
    internal static decimal[] ComputeMeans(decimal[,] returns)
        => CovarianceHelpers<decimal>.ComputeMeans(returns);

    /// <inheritdoc cref="CovarianceHelpers{T}.ComputeSampleCovariance"/>
    internal static decimal[,] ComputeSampleCovariance(decimal[,] returns, decimal[] means)
        => CovarianceHelpers<decimal>.ComputeSampleCovariance(returns, means);

    /// <inheritdoc cref="CovarianceHelpers{T}.CovarianceToCorrelation"/>
    internal static (decimal[,] Correlation, decimal[] StdDevs) CovarianceToCorrelation(decimal[,] cov)
        => CovarianceHelpers<decimal>.CovarianceToCorrelation(cov);

    /// <inheritdoc cref="CovarianceHelpers{T}.CorrelationToCovariance"/>
    internal static decimal[,] CorrelationToCovariance(decimal[,] corr, decimal[] stdDevs)
        => CovarianceHelpers<decimal>.CorrelationToCovariance(corr, stdDevs);

    /// <inheritdoc cref="CovarianceHelpers{T}.MarcenkoPasturUpperBound"/>
    internal static decimal MarcenkoPasturUpperBound(decimal q)
        => CovarianceHelpers<decimal>.MarcenkoPasturUpperBound(q);

    /// <inheritdoc cref="CovarianceHelpers{T}.ReconstructFromEigen"/>
    internal static decimal[,] ReconstructFromEigen(decimal[] values, decimal[,] vectors)
        => CovarianceHelpers<decimal>.ReconstructFromEigen(values, vectors);
}
