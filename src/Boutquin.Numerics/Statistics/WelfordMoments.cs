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
/// Single-pass numerically stable computation of mean, variance, and
/// covariance using Welford's algorithm with the parallel-merge extension
/// of Chan-Golub-LeVeque (1979). Supports both univariate and bivariate
/// (Pearson covariance) statistics in O(n) time and O(1) memory.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics, 4(3), 419-420.</description></item>
/// <item><description>Chan, T. F., Golub, G. H. and LeVeque, R. J. (1979). "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances." Technical Report, Stanford University.</description></item>
/// </list>
/// </para>
/// <para>
/// Use this in place of two-pass algorithms when streaming data, when
/// memory is constrained, or when intermediate sums could overflow. The
/// resulting variance is the unbiased (N-1 divisor) sample variance.
/// </para>
/// <para>
/// Running and combined observation counts must convert to a finite
/// <typeparamref name="T"/>. For <see cref="Half"/>, the maximum count is
/// 65,519; attempting to reach 65,520 throws <see cref="InvalidOperationException"/>.
/// This count limit does not guarantee representability of every moment.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class WelfordMoments<T>
    where T : IFloatingPoint<T>
{
    private long _count;
    private T _anchor = T.Zero;
    private T _mean = T.Zero;
    private T _meanCompensation = T.Zero;
    private T _m2 = T.Zero;
    private T _m2Compensation = T.Zero;
    private T _m2Scale = T.One;
    private bool _m2Scaled;

    /// <summary>Number of observations seen so far.</summary>
    public long Count => _count;

    /// <summary>Sample mean.</summary>
    public T Mean => _count == 0 ? T.Zero : _anchor + _mean;

    /// <summary>Unbiased sample variance (N-1 divisor). Returns 0 for fewer than two observations.</summary>
    public T Variance => _count < 2 ? T.Zero : GetVariance(_count - 1);

    /// <summary>Population variance (N divisor). Returns 0 for an empty sample.</summary>
    public T PopulationVariance => _count < 1 ? T.Zero : GetVariance(_count);

    private T GetVariance(long divisor)
    {
        var value = _m2 / T.CreateChecked(divisor);
        return _m2Scaled ? (value * _m2Scale) * _m2Scale : value;
    }

    /// <summary>Sample standard deviation. Returns 0 for fewer than two observations.</summary>
    /// <remarks>Restores the observation scale after taking the square root, so a
    /// representable standard deviation does not require a representable variance.</remarks>
    public T StdDev
    {
        get
        {
            if (_count < 2)
            {
                return T.Zero;
            }

            var root = NumericPrecision<T>.Sqrt(_m2 / T.CreateChecked(_count - 1));
            return _m2Scaled ? root * _m2Scale : root;
        }
    }

    /// <summary>Adds a single observation, updating the running mean and variance.</summary>
    /// <exception cref="InvalidOperationException">The new count cannot convert to a finite
    /// <typeparamref name="T"/>; <see cref="Half"/> permits at most 65,519 observations.</exception>
    public void Add(T value)
    {
        if (!T.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(nameof(value), value, "Observation must be finite.");
        }

        var newCount = checked(_count + 1);
        var countT = T.CreateChecked(newCount);
        if (!T.IsFinite(countT))
        {
            throw new InvalidOperationException(
                $"The observation count {newCount} cannot be represented as a finite {typeof(T).Name} value.");
        }

        if (_count == 0)
        {
            _anchor = value;
            _mean = T.Zero;
            _m2 = T.Zero;
            _m2Scale = T.One;
            _m2Scaled = false;
            _count = newCount;
            return;
        }

        // The running mean is a weighted average of finite observations, so it is
        // always representable even when the anchor-relative subtraction below is not
        // (e.g. an opposite-sign pair whose magnitudes each approach T.MaxValue, whose
        // raw difference exceeds T's range). Try the direct, anchor-relative path first
        // for its numerical-precision benefit; compute into locals so a mid-way
        // OverflowException (decimal) never leaves the accumulator partially mutated.
        T residual, delta, delta2, updatedMean;
        var meanCompensation = _meanCompensation;
        bool direct;
        try
        {
            residual = value - _anchor;
            delta = residual - _mean;
            updatedMean = _mean;
            AddCompensated(ref updatedMean, ref meanCompensation, delta / countT);
            delta2 = residual - updatedMean;
            direct = T.IsFinite(residual) && T.IsFinite(delta) && T.IsFinite(updatedMean) && T.IsFinite(delta2);
        }
        catch (OverflowException)
        {
            direct = false;
            delta = delta2 = updatedMean = default!;
        }

        if (!direct)
        {
            AddViaSafeMeanUpdate(value, countT, newCount);
            return;
        }

        _mean = updatedMean;
        _meanCompensation = meanCompensation;
        if (!_m2Scaled)
        {
            try
            {
                var increment = delta * delta2;
                // Normalize small nonzero deviations before their product loses
                // precision or underflows (including decimal's fixed scale).
                var smallDeviations = delta != T.Zero && delta2 != T.Zero
                    && T.Abs(delta) < T.One && T.Abs(delta2) < T.One;
                if (!smallDeviations && T.IsFinite(increment) && T.IsFinite(_m2 + increment))
                {
                    AddCompensated(ref _m2, ref _m2Compensation, increment);
                    _count = newCount;
                    return;
                }
            }
            catch (OverflowException)
            {
                // A representable variance may have an unrepresentable M2.
            }

            _m2Scale = T.Max(NumericPrecision<T>.Sqrt(_m2), T.Max(T.Abs(delta), T.Abs(delta2)));
            _m2 = (_m2 / _m2Scale) / _m2Scale;
            _m2Compensation = T.Zero;
            _m2Scaled = true;
        }

        var neededScale = T.Max(_m2Scale, T.Max(T.Abs(delta), T.Abs(delta2)));
        if (neededScale > _m2Scale)
        {
            var ratio = _m2Scale / neededScale;
            _m2 *= ratio * ratio;
            _m2Scale = neededScale;
        }

        AddCompensated(ref _m2, ref _m2Compensation,
            (delta / _m2Scale) * (delta2 / _m2Scale));
        _count = newCount;
    }

    /// <summary>
    /// Updates the mean via a division-first recurrence that cannot overflow, used when
    /// the anchor-relative subtraction in <see cref="Add"/> is not representable in
    /// <typeparamref name="T"/>. Dividing by <paramref name="countT"/> (&gt;= 2 in this
    /// path) before subtracting bounds the intermediate result by
    /// max(|value|, |currentMean|), which is always representable because both operands
    /// are. Re-anchors to the updated mean so later observations are not computed
    /// relative to an anchor whose offset is no longer representable. The observation's
    /// Welford contribution <c>delta^2 . (n-1)/n</c> (with <c>delta = value - mean</c>)
    /// is still added, in scaled form: <c>delta / scale</c> is formed as
    /// <c>value / scale - mean / scale</c> when the raw difference overflows, so M2 never
    /// silently drops the observation.
    /// </summary>
    private void AddViaSafeMeanUpdate(T value, T countT, long newCount)
    {
        var currentMean = _anchor + _mean;
        var newMean = SafeWeightedMean(currentMean, countT - T.One, value, T.One, countT, out var spread);
        _anchor = newMean;
        _mean = T.Zero;
        _meanCompensation = T.Zero;

        var deltaFinite = TryDifference(value, currentMean, out var delta);
        var magnitude = deltaFinite ? T.Abs(delta) : T.Max(T.Abs(value), T.Abs(currentMean));
        var scale = T.Max(T.One, T.Max(spread, T.Max(magnitude, _m2Scale)));
        _m2 = _m2Scaled
            ? (T.IsFinite(_m2) ? _m2 * (_m2Scale / scale) * (_m2Scale / scale) : T.Zero)
            : (T.IsFinite(_m2) ? (_m2 / scale) / scale : T.Zero);
        _m2Compensation = T.Zero;
        _m2Scale = scale;
        _m2Scaled = true;

        var normalizedDelta = deltaFinite ? delta / scale : (value / scale) - (currentMean / scale);
        AddCompensated(ref _m2, ref _m2Compensation,
            normalizedDelta * (normalizedDelta * ((countT - T.One) / countT)));
        _count = newCount;
    }

    /// <summary>
    /// Computes <c>a - b</c>, reporting <see langword="false"/> instead of throwing
    /// (decimal) or returning a non-finite value (IEEE) when the difference is not
    /// representable in <typeparamref name="T"/>.
    /// </summary>
    private static bool TryDifference(T a, T b, out T difference)
    {
        try
        {
            difference = a - b;
            return T.IsFinite(difference);
        }
        catch (OverflowException)
        {
            difference = T.Zero;
            return false;
        }
    }

    /// <summary>Resets the accumulator to its empty state.</summary>
    public void Reset()
    {
        _count = 0;
        _anchor = T.Zero;
        _mean = T.Zero;
        _meanCompensation = T.Zero;
        _m2 = T.Zero;
        _m2Compensation = T.Zero;
        _m2Scale = T.One;
        _m2Scaled = false;
    }

    /// <summary>
    /// Convenience: compute mean and variance over an entire span in one
    /// pass without persistent state.
    /// </summary>
    /// <exception cref="InvalidOperationException">The count cannot convert to a finite
    /// <typeparamref name="T"/>; <see cref="Half"/> permits at most 65,519 observations.</exception>
    public static (T Mean, T Variance) Compute(ReadOnlySpan<T> values)
    {
        var w = new WelfordMoments<T>();
        for (var i = 0; i < values.Length; i++)
        {
            w.Add(values[i]);
        }

        return (w.Mean, w.Variance);
    }

    /// <summary>
    /// Merges another <see cref="WelfordMoments{T}"/> into this one using the
    /// Chan-Golub-LeVeque (1979) parallel-combination formula, enabling
    /// map-reduce-style computation of variance across partitioned data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Given partition <c>A</c> (this) with count <c>n_a</c>, mean <c>mu_a</c>,
    /// and sum-of-squared-deviations <c>M2_a</c>, and partition <c>B</c>
    /// (other) with <c>n_b</c>, <c>mu_b</c>, <c>M2_b</c>, the combined
    /// moments are:
    /// <list type="bullet">
    /// <item><description>n = n_a + n_b</description></item>
    /// <item><description>delta = mu_b - mu_a</description></item>
    /// <item><description>mu = mu_a + delta . n_b / n</description></item>
    /// <item><description>M2 = M2_a + M2_b + delta^2 . n_a . n_b / n</description></item>
    /// </list>
    /// This is numerically equivalent to serially streaming all B observations
    /// through A, but permits truly parallel aggregation.
    /// </para>
    /// </remarks>
    /// <param name="other">The accumulator to merge into this one.</param>
    /// <exception cref="InvalidOperationException">The combined count cannot convert to a finite
    /// <typeparamref name="T"/>; <see cref="Half"/> permits at most 65,519 observations.</exception>
    public void Merge(WelfordMoments<T> other)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (other._count == 0)
        {
            return;
        }

        if (_count == 0)
        {
            _count = other._count;
            _anchor = other._anchor;
            _mean = other._mean;
            _meanCompensation = other._meanCompensation;
            _m2 = other._m2;
            _m2Compensation = other._m2Compensation;
            _m2Scale = other._m2Scale;
            _m2Scaled = other._m2Scaled;
            return;
        }

        var combined = checked(_count + other._count);
        var combinedT = T.CreateChecked(combined);
        var countT = T.CreateChecked(_count);
        var otherCountT = T.CreateChecked(other._count);
        if (!T.IsFinite(combinedT) || !T.IsFinite(countT) || !T.IsFinite(otherCountT))
        {
            throw new InvalidOperationException(
                $"The combined observation count {combined} cannot be represented as a finite {typeof(T).Name} value.");
        }

        var mean = Mean;
        var otherMean = other.Mean;
        var weightSelf = countT / combinedT;

        // The combined mean is a weighted average of two representable means and is
        // always representable, even when the raw difference (otherMean - mean) is not
        // (e.g. two partitions whose means sit at opposite extremes near T.MaxValue).
        // SafeWeightedMean sums the pre-weighted terms (exact, avoids a division-then-
        // subtract rounding loss at a type's extreme boundary) or falls back to
        // dividing by the total weight first, rather than via the delta-based
        // recurrence, so the combined mean never depends on that overflow-prone
        // difference.
        var combinedMean = SafeWeightedMean(mean, countT, otherMean, otherCountT, combinedT, out _);

        T delta;
        bool deltaFinite;
        try
        {
            // Keep the anchor and residual separate: adding them first can
            // round away a partition mean's sub-ULP displacement.
            delta = (other._anchor - _anchor) + (other._mean - _mean);
            deltaFinite = T.IsFinite(delta);
        }
        catch (OverflowException)
        {
            deltaFinite = false;
            delta = default!;
        }

        var mergedAnchor = combinedMean;
        var mergedResidual = T.Zero;
        if (deltaFinite)
        {
            try
            {
                var residual = _mean + delta * (otherCountT / combinedT);
                if (T.IsFinite(residual))
                {
                    mergedAnchor = _anchor;
                    mergedResidual = residual;
                }
            }
            catch (OverflowException)
            {
                // The representable combined mean remains the overflow fallback.
            }
        }

        if (deltaFinite && !_m2Scaled && !other._m2Scaled
            && (delta == T.Zero || T.Abs(delta) >= T.One))
        {
            try
            {
                var correction = delta * (delta * weightSelf) * otherCountT;
                if (T.IsFinite(correction) && T.IsFinite(_m2 + other._m2 + correction))
                {
                    if (typeof(T) == typeof(Half))
                    {
                        AddCompensated(ref _m2, ref _m2Compensation, other._m2 - other._m2Compensation);
                        AddCompensated(ref _m2, ref _m2Compensation, correction);
                    }
                    else
                    {
                        _m2 += other._m2 + correction;
                    }

                    _count = combined;
                    _anchor = mergedAnchor;
                    _mean = mergedResidual;
                    _meanCompensation = T.Zero;
                    return;
                }
            }
            catch (OverflowException)
            {
                // Merge the moments in scaled form below.
            }
        }

        // Preserve both small and large moments in scaled form. When the raw
        // difference itself is unrepresentable, the
        // normalized difference is still formed as otherMean/scale - mean/scale with a
        // scale covering both means, so the cross term (usually the dominant part of
        // the merged M2) is never dropped.
        var deltaMagnitude = deltaFinite ? T.Abs(delta) : T.Max(T.Abs(mean), T.Abs(otherMean));
        var selfScale = _m2Scaled ? _m2Scale : NumericPrecision<T>.Sqrt(_m2);
        var otherScale = other._m2Scaled ? other._m2Scale : NumericPrecision<T>.Sqrt(other._m2);
        var scale = T.Max(deltaMagnitude, T.Max(selfScale, otherScale));
        _m2 = ScaledM2(scale) + other.ScaledM2(scale);
        var normalizedDelta = deltaFinite ? delta / scale : (otherMean / scale) - (mean / scale);
        var crossTerm = normalizedDelta * normalizedDelta * weightSelf * otherCountT;
        if (T.IsFinite(crossTerm))
        {
            _m2 += crossTerm;
        }

        _m2Compensation = T.Zero;
        _m2Scale = scale;
        _m2Scaled = true;
        _count = combined;
        _anchor = mergedAnchor;
        _mean = mergedResidual;
        _meanCompensation = T.Zero;
    }

    private T ScaledM2(T scale)
    {
        if (!_m2Scaled)
        {
            return (_m2 / scale) / scale;
        }

        var ratio = _m2Scale / scale;
        return _m2 * ratio * ratio;
    }

    /// <summary>
    /// Returns a new <see cref="WelfordMoments{T}"/> that is the parallel
    /// combination of <paramref name="left"/> and <paramref name="right"/>.
    /// Neither input is modified.
    /// </summary>
    /// <exception cref="InvalidOperationException">The combined count cannot convert to a finite
    /// <typeparamref name="T"/>; <see cref="Half"/> permits at most 65,519 observations.</exception>
    public static WelfordMoments<T> Combine(WelfordMoments<T> left, WelfordMoments<T> right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        var merged = new WelfordMoments<T>
        {
            _count = left._count,
            _anchor = left._anchor,
            _mean = left._mean,
            _meanCompensation = left._meanCompensation,
            _m2 = left._m2,
            _m2Compensation = left._m2Compensation,
            _m2Scale = left._m2Scale,
            _m2Scaled = left._m2Scaled,
        };
        merged.Merge(right);
        return merged;
    }

    /// <summary>
    /// Computes the weighted average of two finite, representable values without ever
    /// forming their raw (potentially overflowing) difference. Used wherever an online
    /// recurrence would otherwise compute <c>a - b</c> for two quantities that can each
    /// independently approach <typeparamref name="T"/>'s representable range.
    /// </summary>
    /// <remarks>
    /// Tier 1 sums the two pre-weighted terms and divides once; it is exact (no
    /// division-then-subtract rounding loss) but can itself overflow when
    /// <paramref name="currentWeight"/> or <paramref name="incomingWeight"/> is large
    /// (e.g. a running count in the millions). Tier 2 divides each operand by the total
    /// weight first, which is always finite (each term is bounded by the corresponding
    /// input's magnitude) but can lose a representable digit at a type's extreme
    /// boundary (e.g. <c>decimal.MinValue</c>/<c>decimal.MaxValue</c> at n=2, where
    /// halving forces a rounding that tier 1 avoids). Trying tier 1 first covers the
    /// small-count/extreme-magnitude case; falling back to tier 2 covers the
    /// large-count case where tier 1's multiply overflows.
    /// </remarks>
    private static T SafeWeightedMean(T current, T currentWeight, T incoming, T incomingWeight, T totalWeight, out T spread)
    {
        try
        {
            var weighted = current * currentWeight + incoming * incomingWeight;
            if (T.IsFinite(weighted))
            {
                var candidate = weighted / totalWeight;
                if (T.IsFinite(candidate))
                {
                    spread = DifferenceScale(candidate, current);
                    return candidate;
                }
            }
        }
        catch (OverflowException)
        {
            // Fall through to the division-first tier below.
        }

        try
        {
            var newMean = current * (currentWeight / totalWeight) + incoming * (incomingWeight / totalWeight);
            if (T.IsFinite(newMean))
            {
                spread = DifferenceScale(newMean, current);
                return newMean;
            }
        }
        catch (OverflowException)
        {
            // Both tiers overflowed (a doubly-extreme, large-weight regime). Fall back
            // to the incoming value: it is representable by construction (validated
            // finite at the call site), and the true weighted mean lies between it and
            // the current mean, so this is the closer of the two known-representable
            // endpoints when incomingWeight approaches currentWeight in significance,
            // and never worse than an unrepresentable/thrown result.
        }

        spread = DifferenceScale(incoming, current);
        return incoming;
    }

    // Spread only selects a normalization scale. If the displacement cannot be
    // represented, scaling by the larger endpoint still bounds the normalized
    // difference by two, without discarding a valid weighted mean.
    private static T DifferenceScale(T left, T right)
        => TryDifference(left, right, out var difference)
            ? T.Abs(difference)
            : T.Max(T.Abs(left), T.Abs(right));

    private static void AddCompensated(ref T sum, ref T compensation, T increment)
    {
        if (typeof(T) != typeof(Half))
        {
            sum += increment;
            return;
        }

        var adjusted = increment - compensation;
        var next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
    }

    /// <summary>
    /// Online Pearson correlation (mean, variance, covariance updated jointly).
    /// </summary>
    public sealed class Pearson
    {
        private long _count;
        private T _anchorX = T.Zero;
        private T _anchorY = T.Zero;
        private T _meanX = T.Zero;
        private T _meanY = T.Zero;
        private T _meanXCompensation = T.Zero;
        private T _meanYCompensation = T.Zero;
        private T _m2X = T.Zero;
        private T _m2Y = T.Zero;
        private T _coMoment = T.Zero;
        private T _m2XCompensation = T.Zero;
        private T _m2YCompensation = T.Zero;
        private T _coMomentCompensation = T.Zero;
        // Once scaled, M2X is stored divided by _scaleX^2, M2Y by _scaleY^2 and the
        // co-moment by _scaleX * _scaleY. Each variable is normalized by its own
        // magnitude: a shared scale set by the larger series would underflow the
        // smaller series' squared deviations and collapse the correlation.
        private T _scaleX = T.One;
        private T _scaleY = T.One;
        private bool _scaled;

        /// <summary>Number of observations seen so far.</summary>
        public long Count => _count;

        /// <summary>Adds a paired observation.</summary>
        /// <exception cref="InvalidOperationException">The new count cannot convert to a finite
        /// <typeparamref name="T"/>; <see cref="Half"/> permits at most 65,519 observations.</exception>
        public void Add(T x, T y)
        {
            if (!T.IsFinite(x))
            {
                throw new ArgumentOutOfRangeException(nameof(x), x, "Observation must be finite.");
            }

            if (!T.IsFinite(y))
            {
                throw new ArgumentOutOfRangeException(nameof(y), y, "Observation must be finite.");
            }

            var newCount = checked(_count + 1);
            var countT = T.CreateChecked(newCount);
            if (!T.IsFinite(countT))
            {
                throw new InvalidOperationException(
                    $"The observation count {newCount} cannot be represented as a finite {typeof(T).Name} value.");
            }

            if (_count == 0)
            {
                _anchorX = x;
                _anchorY = y;
                _count = newCount;
                return;
            }

            if (_scaled)
            {
                AddScaled(x, y, countT, newCount);
                return;
            }

            // The running means are weighted averages of finite observations, so they
            // are always representable even when the anchor-relative subtraction below
            // is not (e.g. an opposite-sign pair whose magnitudes each approach
            // T.MaxValue). Compute into locals first so a mid-way OverflowException
            // (decimal) never leaves the accumulator partially mutated.
            // Compute the mean update first, in isolation from the M2/co-moment
            // products below: a finite mean with an unrepresentable *product* of
            // deviations (e.g. two deviations of order 1e200, whose product overflows
            // double) is a different, already-handled failure mode (the scaled M2 path
            // below) from an unrepresentable *mean* (this class's overflow fallback).
            // Conflating them would abandon a perfectly good mean update whenever only
            // the squared-deviation product overflows.
            T residualX, residualY, deltaX, deltaY, newMeanX, newMeanY;
            var meanXCompensation = _meanXCompensation;
            var meanYCompensation = _meanYCompensation;
            bool direct;
            try
            {
                residualX = x - _anchorX;
                residualY = y - _anchorY;
                deltaX = residualX - _meanX;
                deltaY = residualY - _meanY;
                newMeanX = _meanX;
                newMeanY = _meanY;
                AddCompensated(ref newMeanX, ref meanXCompensation, deltaX / countT);
                AddCompensated(ref newMeanY, ref meanYCompensation, deltaY / countT);

                direct = T.IsFinite(residualX) && T.IsFinite(residualY)
                    && T.IsFinite(deltaX) && T.IsFinite(deltaY)
                    && T.IsFinite(newMeanX) && T.IsFinite(newMeanY);
            }
            catch (OverflowException)
            {
                direct = false;
                residualX = residualY = deltaX = deltaY = newMeanX = newMeanY = default!;
            }

            if (!direct)
            {
                AddViaSafeMeanUpdate(x, y, countT, newCount);
                return;
            }

            if (!_scaled)
            {
                T m2XIncrement, m2YIncrement, coMomentIncrement;
                bool incrementsOk;
                try
                {
                    m2XIncrement = deltaX * (residualX - newMeanX);
                    m2YIncrement = deltaY * (residualY - newMeanY);
                    coMomentIncrement = deltaX * (residualY - newMeanY);
                    incrementsOk = T.IsFinite(m2XIncrement) && T.IsFinite(m2YIncrement) && T.IsFinite(coMomentIncrement)
                        && T.IsFinite(_m2X + m2XIncrement) && T.IsFinite(_m2Y + m2YIncrement) && T.IsFinite(_coMoment + coMomentIncrement);
                }
                catch (OverflowException)
                {
                    incrementsOk = false;
                    m2XIncrement = m2YIncrement = coMomentIncrement = default!;
                }

                // Normalize a newly varying small series before products can round
                // to zero. Correlation can be representable when covariance is not.
                var magnitudeX = T.Max(T.Abs(deltaX), T.Abs(residualX - newMeanX));
                var magnitudeY = T.Max(T.Abs(deltaY), T.Abs(residualY - newMeanY));
                var smallX = _m2X == T.Zero && magnitudeX > T.Zero && magnitudeX < T.One;
                var smallY = _m2Y == T.Zero && magnitudeY > T.Zero && magnitudeY < T.One;
                if (incrementsOk && !smallX && !smallY)
                {
                    _meanX = newMeanX;
                    _meanY = newMeanY;
                    _meanXCompensation = meanXCompensation;
                    _meanYCompensation = meanYCompensation;
                    AddCompensated(ref _m2X, ref _m2XCompensation, m2XIncrement);
                    AddCompensated(ref _m2Y, ref _m2YCompensation, m2YIncrement);
                    AddCompensated(ref _coMoment, ref _coMomentCompensation, coMomentIncrement);
                    _count = newCount;
                    return;
                }

                _scaleX = smallX ? magnitudeX : T.Max(T.One, magnitudeX);
                _scaleY = smallY ? magnitudeY : T.Max(T.One, magnitudeY);
                _m2X = (_m2X / _scaleX) / _scaleX;
                _m2Y = (_m2Y / _scaleY) / _scaleY;
                _coMoment = (_coMoment / _scaleX) / _scaleY;
                _meanX /= _scaleX;
                _meanY /= _scaleY;
                _meanXCompensation /= _scaleX;
                _meanYCompensation /= _scaleY;
                _m2XCompensation = T.Zero;
                _m2YCompensation = T.Zero;
                _coMomentCompensation = T.Zero;
                _scaled = true;
            }

            AddScaled(x, y, countT, newCount);
        }

        /// <summary>
        /// Enters normalized coordinates when the anchor-relative subtraction in
        /// <see cref="Add"/> is not representable. The initial scales are one;
        /// <see cref="AddScaled"/> grows them before forming either deviation.
        /// </summary>
        private void AddViaSafeMeanUpdate(T x, T y, T countT, long newCount)
        {
            _m2XCompensation = T.Zero;
            _m2YCompensation = T.Zero;
            _coMomentCompensation = T.Zero;
            _scaled = true;
            AddScaled(x, y, countT, newCount);
        }

        // In scaled mode the anchor-relative means share the moments' scales.
        // Normalize before centering, so a subnormal mean increment is retained.
        private void AddScaled(T x, T y, T countT, long newCount)
        {
            var finiteX = TryDifference(x, _anchorX, out var residualX);
            var finiteY = TryDifference(y, _anchorY, out var residualY);
            GrowScales(
                finiteX ? T.Abs(residualX) : T.Max(T.Abs(x), T.Abs(_anchorX)),
                finiteY ? T.Abs(residualY) : T.Max(T.Abs(y), T.Abs(_anchorY)));
            var normalizedX = finiteX ? residualX / _scaleX : x / _scaleX - _anchorX / _scaleX;
            var normalizedY = finiteY ? residualY / _scaleY : y / _scaleY - _anchorY / _scaleY;
            var deltaX = normalizedX - _meanX;
            var deltaY = normalizedY - _meanY;
            AddCompensated(ref _meanX, ref _meanXCompensation, deltaX / countT);
            AddCompensated(ref _meanY, ref _meanYCompensation, deltaY / countT);
            AddCompensated(ref _m2X, ref _m2XCompensation, deltaX * (normalizedX - _meanX));
            AddCompensated(ref _m2Y, ref _m2YCompensation, deltaY * (normalizedY - _meanY));
            AddCompensated(ref _coMoment, ref _coMomentCompensation, deltaX * (normalizedY - _meanY));
            _count = newCount;
        }

        /// <summary>
        /// Rescales each variable independently. A constant series may acquire a small
        /// first nonzero scale; otherwise scales only grow. Ratios bounded by one
        /// avoid overflow when the magnitude grows across the type's dynamic range.
        /// </summary>
        private void GrowScales(T magnitudeX, T magnitudeY)
        {
            var neededX = _m2X == T.Zero && magnitudeX > T.Zero ? magnitudeX : T.Max(_scaleX, magnitudeX);
            if (neededX != _scaleX)
            {
                if (_meanX != T.Zero || _meanXCompensation != T.Zero)
                {
                    var meanRatio = _scaleX / neededX;
                    _meanX *= meanRatio;
                    _meanXCompensation *= meanRatio;
                }
                if (_m2X != T.Zero)
                {
                    var ratio = _scaleX / neededX;
                    _m2X = (_m2X * ratio) * ratio;
                    _coMoment *= ratio;
                }
                _scaleX = neededX;
            }

            var neededY = _m2Y == T.Zero && magnitudeY > T.Zero ? magnitudeY : T.Max(_scaleY, magnitudeY);
            if (neededY != _scaleY)
            {
                if (_meanY != T.Zero || _meanYCompensation != T.Zero)
                {
                    var meanRatio = _scaleY / neededY;
                    _meanY *= meanRatio;
                    _meanYCompensation *= meanRatio;
                }
                if (_m2Y != T.Zero)
                {
                    var ratio = _scaleY / neededY;
                    _m2Y = (_m2Y * ratio) * ratio;
                    _coMoment *= ratio;
                }
                _scaleY = neededY;
            }
        }

        /// <summary>Pearson correlation coefficient over the data seen so far.</summary>
        public T Correlation
        {
            get
            {
                if (_count < 2 || _m2X == T.Zero || _m2Y == T.Zero)
                {
                    return T.Zero;
                }

                var scaleX = NumericPrecision<T>.Sqrt(_m2X);
                var scaleY = NumericPrecision<T>.Sqrt(_m2Y);
                var corr = (_coMoment / scaleX) / scaleY;
                return T.Clamp(corr, -T.One, T.One);
            }
        }

        /// <summary>Sample covariance (N-1 divisor).</summary>
        public T Covariance
        {
            get
            {
                if (_count < 2 || _coMoment == T.Zero)
                {
                    return T.Zero;
                }

                var divisor = T.CreateChecked(_count - 1);
                if (!_scaled)
                {
                    return _coMoment / divisor;
                }

                if (typeof(T) == typeof(double) || typeof(T) == typeof(float) || typeof(T) == typeof(Half))
                {
                    // Combine exponents before restoring either scale. Multiplying
                    // by the tiny axis first can underflow a representable result.
                    var moment = double.CreateChecked(_coMoment);
                    var scaleX = double.CreateChecked(_scaleX);
                    var scaleY = double.CreateChecked(_scaleY);
                    var count = double.CreateChecked(divisor);
                    var em = Math.ILogB(Math.Abs(moment));
                    var ex = Math.ILogB(scaleX);
                    var ey = Math.ILogB(scaleY);
                    var en = Math.ILogB(count);
                    var mantissa = Math.ScaleB(moment, -em) * Math.ScaleB(scaleX, -ex)
                        * Math.ScaleB(scaleY, -ey) / Math.ScaleB(count, -en);
                    return T.CreateChecked(Math.ScaleB(mantissa, em + ex + ey - en));
                }

                if ((_scaleX < T.One && _scaleY > T.One) || (_scaleY < T.One && _scaleX > T.One))
                {
                    // Opposing scales have a product between their endpoints.
                    // Combine them first to avoid decimal's fixed-scale underflow.
                    return (_coMoment / divisor) * (_scaleX * _scaleY);
                }

                return ((_coMoment / divisor) * _scaleX) * _scaleY;
            }
        }

        /// <summary>Resets the accumulator.</summary>
        public void Reset()
        {
            _count = 0;
            _anchorX = T.Zero;
            _anchorY = T.Zero;
            _meanX = T.Zero;
            _meanY = T.Zero;
            _meanXCompensation = T.Zero;
            _meanYCompensation = T.Zero;
            _m2X = T.Zero;
            _m2Y = T.Zero;
            _coMoment = T.Zero;
            _m2XCompensation = T.Zero;
            _m2YCompensation = T.Zero;
            _coMomentCompensation = T.Zero;
            _scaleX = T.One;
            _scaleY = T.One;
            _scaled = false;
        }
    }
}

/// <summary>
/// Single-pass numerically stable computation of mean, variance, and
/// covariance using Welford's algorithm with the parallel-merge extension
/// of Chan-Golub-LeVeque (1979). Supports both univariate and bivariate
/// (Pearson covariance) statistics in O(n) time and O(1) memory.
/// </summary>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics, 4(3), 419-420.</description></item>
/// <item><description>Chan, T. F., Golub, G. H. and LeVeque, R. J. (1979). "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances." Technical Report, Stanford University.</description></item>
/// </list>
/// </para>
/// <para>
/// Use this in place of two-pass algorithms when streaming data, when
/// memory is constrained, or when intermediate sums could overflow. The
/// resulting variance is the unbiased (N-1 divisor) sample variance.
/// </para>
/// </remarks>
[Obsolete(
    "Use WelfordMoments<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class WelfordMoments
{
    private readonly WelfordMoments<decimal> _inner = new();

    /// <summary>Number of observations seen so far.</summary>
    public long Count => _inner.Count;

    /// <summary>Sample mean.</summary>
    public decimal Mean => _inner.Mean;

    /// <summary>Unbiased sample variance (N-1 divisor). Returns 0 for fewer than two observations.</summary>
    public decimal Variance => _inner.Variance;

    /// <summary>Population variance (N divisor). Returns 0 for an empty sample.</summary>
    public decimal PopulationVariance => _inner.PopulationVariance;

    /// <summary>Sample standard deviation.</summary>
    public decimal StdDev => _inner.StdDev;

    /// <summary>Adds a single observation, updating the running mean and variance.</summary>
    public void Add(decimal value) => _inner.Add(value);

    /// <summary>Resets the accumulator to its empty state.</summary>
    public void Reset() => _inner.Reset();

    /// <summary>
    /// Convenience: compute mean and variance over an entire span in one
    /// pass without persistent state.
    /// </summary>
    public static (decimal Mean, decimal Variance) Compute(ReadOnlySpan<decimal> values)
        => WelfordMoments<decimal>.Compute(values);

    /// <summary>
    /// Merges another <see cref="WelfordMoments"/> into this one using the
    /// Chan-Golub-LeVeque (1979) parallel-combination formula, enabling
    /// map-reduce-style computation of variance across partitioned data.
    /// </summary>
    /// <param name="other">The accumulator to merge into this one.</param>
    public void Merge(WelfordMoments other)
    {
        ArgumentNullException.ThrowIfNull(other);
        _inner.Merge(other._inner);
    }

    /// <summary>
    /// Returns a new <see cref="WelfordMoments"/> that is the parallel
    /// combination of <paramref name="left"/> and <paramref name="right"/>.
    /// Neither input is modified.
    /// </summary>
    public static WelfordMoments Combine(WelfordMoments left, WelfordMoments right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

#pragma warning disable BNU0001
        var merged = new WelfordMoments();
#pragma warning restore BNU0001
        // Merge left then right into the fresh accumulator.
        merged._inner.Merge(left._inner);
        merged._inner.Merge(right._inner);
        return merged;
    }

    /// <summary>
    /// Online Pearson correlation (mean, variance, covariance updated jointly).
    /// </summary>
    public sealed class Pearson
    {
        private readonly WelfordMoments<decimal>.Pearson _inner = new();

        /// <summary>Number of observations seen so far.</summary>
        public long Count => _inner.Count;

        /// <summary>Adds a paired observation.</summary>
        public void Add(decimal x, decimal y) => _inner.Add(x, y);

        /// <summary>Pearson correlation coefficient over the data seen so far.</summary>
        public decimal Correlation => _inner.Correlation;

        /// <summary>Sample covariance (N-1 divisor).</summary>
        public decimal Covariance => _inner.Covariance;

        /// <summary>Resets the accumulator.</summary>
        public void Reset() => _inner.Reset();
    }
}
