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

namespace Boutquin.Numerics.MonteCarlo;

/// <summary>
/// P² (P-square) online percentile estimator. Tracks a single percentile
/// of an unbounded stream with constant memory (five marker positions and
/// five marker heights). Uses parabolic interpolation to keep the markers
/// at their ideal quantile positions as new observations arrive.
/// </summary>
/// <typeparam name="T">Floating-point type for the observation values.</typeparam>
/// <remarks>
/// <para>
/// Reference: Jain, R. &amp; Chlamtac, I. (1985). "The P² Algorithm for Dynamic
/// Calculation of Quantiles and Histograms Without Storing Observations."
/// Communications of the ACM, 28(10), 1076–1085.
/// </para>
/// <para>
/// Use this estimator for streaming bootstrap, very large simulation counts,
/// or any setting where materializing the full sample is infeasible. Accuracy
/// depends on the distribution, target percentile, and observation count;
/// tail percentiles of discrete distributions can differ substantially from
/// the exact sample percentile even after thousands of observations.
/// </para>
/// <para>
/// This implementation uses the five-marker variant that tracks a single
/// percentile <c>p</c>. The markers are positioned at <c>0, p/2, p, (1+p)/2, 1</c>
/// of the cumulative distribution. Until five observations have been seen
/// the estimator simply stores the raw samples and reports the interpolated
/// percentile over that sorted buffer.
/// </para>
/// <para>
/// Tier A: Observation values and estimates use T; internal position tracking uses double.
/// </para>
/// <para>
/// Not thread-safe: holds mutable advancement state. Instantiate one instance
/// per thread; do not share across concurrent consumers.
/// </para>
/// </remarks>
public sealed class PSquareEstimator<T>
    where T : IFloatingPoint<T>
{
    private readonly double _p;
    private readonly T[] _q = new T[5];
    private readonly long[] _n = new long[5];
    private long _count;

    /// <summary>
    /// Creates an estimator for the given percentile.
    /// </summary>
    /// <param name="percentile">Percentile in (0, 1).</param>
    public PSquareEstimator(double percentile)
    {
        if (!double.IsFinite(percentile) || percentile <= 0.0 || percentile >= 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(percentile),
                percentile,
                "Percentile must lie strictly between 0 and 1.");
        }

        _p = percentile;
    }

    /// <summary>Target percentile (same value passed to the constructor).</summary>
    public double Percentile => _p;

    /// <summary>Number of observations seen so far.</summary>
    public long Count => _count;

    /// <summary>
    /// Current estimate of the target percentile. Returns 0 when no
    /// observations have been added. For fewer than five observations
    /// the estimate is obtained by linear interpolation over the sorted
    /// partial sample.
    /// </summary>
    public T Estimate
    {
        get
        {
            if (_count == 0)
            {
                return T.Zero;
            }

            if (_count < 5)
            {
                var buffer = new T[(int)_count];
                Array.Copy(_q, buffer, buffer.Length);
                Array.Sort(buffer);

                return MonteCarlo.Percentile.Compute<T>(buffer, _p);
            }

            return _q[2];
        }
    }

    /// <summary>
    /// Adds an observation to the stream and updates the percentile estimate.
    /// </summary>
    public void Add(T observation)
    {
        if (!T.IsFinite(observation))
        {
            throw new ArgumentOutOfRangeException(
                nameof(observation),
                observation,
                "Observation must be a finite number.");
        }

        if (_count < 5)
        {
            _q[_count] = observation;
            _count++;

            if (_count == 5)
            {
                Array.Sort(_q);
                for (var i = 0; i < 5; i++)
                {
                    _n[i] = i + 1;
                }
            }

            return;
        }

        int k;
        if (observation < _q[0])
        {
            _q[0] = observation;
            k = 0;
        }
        else if (observation < _q[1])
        {
            k = 0;
        }
        else if (observation < _q[2])
        {
            k = 1;
        }
        else if (observation < _q[3])
        {
            k = 2;
        }
        else if (observation < _q[4])
        {
            // Interior top cell [q4, q5): strict upper bound per the canonical
            // Jain-Chlamtac (1985) cell-finding loop. An observation exactly
            // equal to q5 is NOT interior — it is handled by the max-extension
            // branch below (a no-op height update; the marker count still advances).
            k = 3;
        }
        else
        {
            // Max-extension: observation >= q5. The canonical algorithm widens
            // the top marker to the new maximum; when observation == q5 this is
            // an idempotent assignment, but k = 3 is reached the same way so the
            // marker counters advance correctly either way.
            _q[4] = observation;
            k = 3;
        }

        for (var i = k + 1; i < 5; i++)
        {
            _n[i]++;
        }

        for (var i = 1; i <= 3; i++)
        {
            // Compare count*p directly with the integer boundary. Forming
            // desired-n first can round a value just inside a +/-1 boundary
            // onto equality; repeated desired-position addition also drifts.
            var boundary = i == 2 ? (double)_n[i] : 2.0 * _n[i] - (i == 3 ? _count : 0);
            var step = i == 2 ? 2.0 : 4.0;
            var moveUp = double.FusedMultiplyAdd(_count, _p, -boundary) >= 0.0 && _n[i + 1] - _n[i] > 1;
            var moveDown = double.FusedMultiplyAdd(_count, _p, -(boundary - step)) <= 0.0 && _n[i - 1] - _n[i] < -1;
            if (moveUp || moveDown)
            {
                var sign = moveUp ? 1 : -1;
                _q[i] = TryParabolic(i, sign, out var qp) ? qp : Linear(i, sign);

                _n[i] += sign;
            }
        }

        _count++;
    }

    // Piecewise-parabolic (P²) prediction of marker i's new height. Returns false
    // when the prediction does not lie strictly between the neighbouring heights,
    // in which case the caller falls back to the linear formula.
    //
    // Evaluated at quarter scale so that finite markers can never produce a
    // non-finite (double) or overflowing (decimal) intermediate: each quarter gap
    // is at most MaxValue/2, and the combined coefficients
    //   cPost = d·(n_i − n_{i−1} + d) / ((n_{i+1} − n_{i−1})·(n_{i+1} − n_i))
    //   cPre  = d·(n_{i+1} − n_i − d) / ((n_{i+1} − n_{i−1})·(n_i − n_{i−1}))
    // have magnitude at most 1 under the marker-move precondition, so the
    // quarter-scale step is at most MaxValue in magnitude. An accepted step keeps
    // the height inside (q_{i−1}, q_{i+1}) in exact arithmetic. Rounding the
    // addition can still reach a quarter-scaled endpoint, especially in decimal;
    // check the rounded candidate before rescaling and the final height afterward.
    private bool TryParabolic(int i, int d, out T value)
    {
        var ni = _n[i];
        var npre = _n[i - 1];
        var npost = _n[i + 1];

        var denom = (double)(npost - npre);
        var postCoefficient = d * (ni - npre + d) / (denom * (npost - ni));
        var preCoefficient = d * (npost - ni - d) / (denom * (ni - npre));
        if (typeof(T) == typeof(Half))
        {
            // The coefficients shrink with the marker spacing and fall below Half's
            // least subnormal while their products with the height gaps stay
            // representable. As in Linear, keep them in double and round the
            // candidate height once; Half gaps cannot overflow a double.
            var height = double.CreateChecked(_q[i]);
            var gapAbove = double.CreateChecked(_q[i + 1]) - height;
            var gapBelow = height - double.CreateChecked(_q[i - 1]);
            var move = postCoefficient * gapAbove + preCoefficient * gapBelow;
            if (-gapBelow < move && move < gapAbove)
            {
                value = T.CreateChecked(height + move);
                return _q[i - 1] < value && value < _q[i + 1];
            }

            value = T.Zero;
            return false;
        }

        var cPost = T.CreateChecked(postCoefficient);
        var cPre = T.CreateChecked(preCoefficient);

        var four = T.CreateChecked(4);
        var q = _q[i] / four;
        var gapPost = _q[i + 1] / four - q;
        var gapPre = q - _q[i - 1] / four;

        var step = cPost * gapPost + cPre * gapPre;
        if (-gapPre < step && step < gapPost)
        {
            var candidate = q + step;
            if (_q[i - 1] / four < candidate && candidate < _q[i + 1] / four)
            {
                value = candidate * four;
                return _q[i - 1] < value && value < _q[i + 1];
            }
        }

        value = T.Zero;
        return false;
    }

    // Linear move of marker i one position toward neighbour i + d. A rounded
    // parabolic prediction equal to a neighbour also reaches this fallback, so
    // opposite-sign heights can have a gap larger than T can represent.
    private T Linear(int i, int d)
    {
        var direction = i + d;
        if (typeof(T) == typeof(Half))
        {
            // Long position gaps exceed Half's range; a tiny ratio can still
            // produce a representable height increment. Round only the result.
            var height = double.CreateChecked(_q[i]);
            var neighbour = double.CreateChecked(_q[direction]);
            return T.CreateChecked(height + (neighbour - height) / Math.Abs(_n[direction] - _n[i]));
        }

        if ((_q[i] < T.Zero && _q[direction] > T.Zero) ||
            (_q[i] > T.Zero && _q[direction] < T.Zero))
        {
            var fraction = T.One / T.CreateChecked(Math.Abs(_n[direction] - _n[i]));
            return (T.One - fraction) * _q[i] + fraction * _q[direction];
        }

        return _q[i] + T.CreateChecked(d) * (_q[direction] - _q[i]) / T.CreateChecked(_n[direction] - _n[i]);
    }
}

/// <summary>
/// P² (P-square) online percentile estimator. Tracks a single percentile
/// of an unbounded stream with constant memory (five marker positions and
/// five marker heights). Uses parabolic interpolation to keep the markers
/// at their ideal quantile positions as new observations arrive.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Jain, R. &amp; Chlamtac, I. (1985). "The P² Algorithm for Dynamic
/// Calculation of Quantiles and Histograms Without Storing Observations."
/// Communications of the ACM, 28(10), 1076–1085.
/// </para>
/// <para>
/// Use this estimator for streaming bootstrap, very large simulation counts,
/// or any setting where materializing the full sample is infeasible. Accuracy
/// depends on the distribution, target percentile, and observation count;
/// tail percentiles of discrete distributions can differ substantially from
/// the exact sample percentile even after thousands of observations.
/// </para>
/// <para>
/// This implementation uses the five-marker variant that tracks a single
/// percentile <c>p</c>. The markers are positioned at <c>0, p/2, p, (1+p)/2, 1</c>
/// of the cumulative distribution. Until five observations have been seen
/// the estimator simply stores the raw samples and reports the interpolated
/// percentile over that sorted buffer.
/// </para>
/// <para>
/// Tier A: Delegates to <see cref="PSquareEstimator{T}"/> with T = <see cref="double"/>.
/// </para>
/// <para>
/// Not thread-safe: holds mutable advancement state. Instantiate one instance
/// per thread; do not share across concurrent consumers.
/// </para>
/// </remarks>
public sealed class PSquareEstimator
{
    private readonly PSquareEstimator<double> _impl;

    /// <summary>
    /// Creates an estimator for the given percentile.
    /// </summary>
    /// <param name="percentile">Percentile in (0, 1).</param>
    public PSquareEstimator(double percentile)
    {
        _impl = new PSquareEstimator<double>(percentile);
    }

    /// <summary>Target percentile (same value passed to the constructor).</summary>
    public double Percentile => _impl.Percentile;

    /// <summary>Number of observations seen so far.</summary>
    public long Count => _impl.Count;

    /// <summary>
    /// Current estimate of the target percentile. Returns 0 when no
    /// observations have been added. For fewer than five observations
    /// the estimate is obtained by linear interpolation over the sorted
    /// partial sample.
    /// </summary>
    public double Estimate => _impl.Estimate;

    /// <summary>
    /// Adds an observation to the stream and updates the percentile estimate.
    /// </summary>
    public void Add(double observation) => _impl.Add(observation);
}
