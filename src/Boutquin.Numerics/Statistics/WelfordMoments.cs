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
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class WelfordMoments<T>
    where T : IFloatingPoint<T>
{
    private long _count;
    private T _mean = T.Zero;
    private T _m2 = T.Zero;

    /// <summary>Number of observations seen so far.</summary>
    public long Count => _count;

    /// <summary>Sample mean.</summary>
    public T Mean => _count == 0 ? T.Zero : _mean;

    /// <summary>Unbiased sample variance (N-1 divisor). Returns 0 for fewer than two observations.</summary>
    public T Variance => _count < 2 ? T.Zero : _m2 / T.CreateChecked(_count - 1);

    /// <summary>Population variance (N divisor). Returns 0 for an empty sample.</summary>
    public T PopulationVariance => _count < 1 ? T.Zero : _m2 / T.CreateChecked(_count);

    /// <summary>Sample standard deviation.</summary>
    public T StdDev => NumericPrecision<T>.Sqrt(Variance);

    /// <summary>Adds a single observation, updating the running mean and variance.</summary>
    public void Add(T value)
    {
        _count++;
        var countT = T.CreateChecked(_count);
        var delta = value - _mean;
        _mean += delta / countT;
        var delta2 = value - _mean;
        _m2 += delta * delta2;
    }

    /// <summary>Resets the accumulator to its empty state.</summary>
    public void Reset()
    {
        _count = 0;
        _mean = T.Zero;
        _m2 = T.Zero;
    }

    /// <summary>
    /// Convenience: compute mean and variance over an entire span in one
    /// pass without persistent state.
    /// </summary>
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
            _mean = other._mean;
            _m2 = other._m2;
            return;
        }

        var combined = _count + other._count;
        var combinedT = T.CreateChecked(combined);
        var countT = T.CreateChecked(_count);
        var otherCountT = T.CreateChecked(other._count);
        var delta = other._mean - _mean;
        _mean += delta * otherCountT / combinedT;
        _m2 += other._m2 + delta * delta * countT * otherCountT / combinedT;
        _count = combined;
    }

    /// <summary>
    /// Returns a new <see cref="WelfordMoments{T}"/> that is the parallel
    /// combination of <paramref name="left"/> and <paramref name="right"/>.
    /// Neither input is modified.
    /// </summary>
    public static WelfordMoments<T> Combine(WelfordMoments<T> left, WelfordMoments<T> right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        var merged = new WelfordMoments<T>
        {
            _count = left._count,
            _mean = left._mean,
            _m2 = left._m2,
        };
        merged.Merge(right);
        return merged;
    }

    /// <summary>
    /// Online Pearson correlation (mean, variance, covariance updated jointly).
    /// </summary>
    public sealed class Pearson
    {
        private long _count;
        private T _meanX = T.Zero;
        private T _meanY = T.Zero;
        private T _m2X = T.Zero;
        private T _m2Y = T.Zero;
        private T _coMoment = T.Zero;

        /// <summary>Number of observations seen so far.</summary>
        public long Count => _count;

        /// <summary>Adds a paired observation.</summary>
        public void Add(T x, T y)
        {
            _count++;
            var countT = T.CreateChecked(_count);
            var deltaX = x - _meanX;
            _meanX += deltaX / countT;
            var deltaY = y - _meanY;
            _meanY += deltaY / countT;

            // M_2 updates use post-mean delta.
            _m2X += deltaX * (x - _meanX);
            _m2Y += deltaY * (y - _meanY);
            // Covariance (Welford bivariate, Bennett 2009 form).
            _coMoment += deltaX * (y - _meanY);
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

                var denom = NumericPrecision<T>.Sqrt(_m2X * _m2Y);
                var corr = _coMoment / denom;
                return T.Clamp(corr, -T.One, T.One);
            }
        }

        /// <summary>Sample covariance (N-1 divisor).</summary>
        public T Covariance => _count < 2 ? T.Zero : _coMoment / T.CreateChecked(_count - 1);

        /// <summary>Resets the accumulator.</summary>
        public void Reset()
        {
            _count = 0;
            _meanX = T.Zero;
            _meanY = T.Zero;
            _m2X = T.Zero;
            _m2Y = T.Zero;
            _coMoment = T.Zero;
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

        var merged = new WelfordMoments();
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
