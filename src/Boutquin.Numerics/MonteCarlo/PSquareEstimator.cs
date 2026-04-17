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
/// on stationary streams converges to within ~1% of the true percentile after
/// a few thousand observations.
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
/// </remarks>
public sealed class PSquareEstimator<T>
    where T : IFloatingPoint<T>
{
    private readonly double _p;
    private readonly T[] _q = new T[5];
    private readonly int[] _n = new int[5];
    private readonly double[] _nprime = new double[5];
    private readonly double[] _dn = new double[5];
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
        _dn[0] = 0.0;
        _dn[1] = _p / 2.0;
        _dn[2] = _p;
        _dn[3] = (1.0 + _p) / 2.0;
        _dn[4] = 1.0;
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

                var rank = _p * (buffer.Length - 1);
                var lo = (int)Math.Floor(rank);
                var hi = (int)Math.Ceiling(rank);
                if (lo == hi)
                {
                    return buffer[lo];
                }

                var frac = T.CreateChecked(rank - lo);
                return buffer[lo] + frac * (buffer[hi] - buffer[lo]);
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
                    _nprime[i] = 1.0 + _dn[i] * 4.0;
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
        else if (observation <= _q[4])
        {
            k = 3;
        }
        else
        {
            _q[4] = observation;
            k = 3;
        }

        for (var i = k + 1; i < 5; i++)
        {
            _n[i]++;
        }

        for (var i = 0; i < 5; i++)
        {
            _nprime[i] += _dn[i];
        }

        for (var i = 1; i <= 3; i++)
        {
            var d = _nprime[i] - _n[i];
            if ((d >= 1.0 && _n[i + 1] - _n[i] > 1) ||
                (d <= -1.0 && _n[i - 1] - _n[i] < -1))
            {
                var sign = Math.Sign(d);
                var qp = Parabolic(i, sign);
                if (_q[i - 1] < qp && qp < _q[i + 1])
                {
                    _q[i] = qp;
                }
                else
                {
                    _q[i] = Linear(i, sign);
                }

                _n[i] += sign;
            }
        }

        _count++;
    }

    private T Parabolic(int i, int d)
    {
        var ni = _n[i];
        var npre = _n[i - 1];
        var npost = _n[i + 1];
        var q = _q[i];
        var qpre = _q[i - 1];
        var qpost = _q[i + 1];

        var denom = npost - npre;
        var lhs = T.CreateChecked((double)d / denom);
        var leftFactor = T.CreateChecked((ni - npre + d) / (double)(npost - ni)) * (qpost - q);
        var rightFactor = T.CreateChecked((npost - ni - d) / (double)(ni - npre)) * (q - qpre);
        return q + lhs * (leftFactor + rightFactor);
    }

    private T Linear(int i, int d)
    {
        var direction = i + d;
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
/// on stationary streams converges to within ~1% of the true percentile after
/// a few thousand observations.
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
