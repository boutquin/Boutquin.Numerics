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
/// Generic exponentially weighted moving-average (EWMA) covariance estimator
/// (RiskMetrics, J.P. Morgan 1996). Recent observations carry geometrically
/// higher weight than old ones, parameterized by the decay factor <c>lambda in (0, 1)</c>.
/// PSD by construction whenever all weights are non-negative, which is
/// guaranteed for <c>lambda in (0, 1)</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Weights are assigned by <c>w_k proportional to lambda^(T-1-k)</c> for <c>k in [0, T)</c>, then
/// normalized to sum to 1 so the result is an honest weighted covariance.
/// </para>
/// <para>
/// Numerical note: the <c>lambda^(T-1-k)</c> power series is computed in
/// <c>double</c> via <c>Math.Pow</c> to avoid the accumulated drift of
/// repeated multiplication, then cast to <typeparamref name="T"/> for
/// the weight-normalization and cross-product pass.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class ExponentiallyWeightedCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private readonly T _lambda;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExponentiallyWeightedCovarianceEstimator{T}"/> type.
    /// </summary>
    /// <param name="lambda">Decay factor in (0, 1). Default 0.94 (RiskMetrics daily).</param>
    /// <exception cref="ArgumentOutOfRangeException">Lambda is not strictly between 0 and 1.</exception>
    public ExponentiallyWeightedCovarianceEstimator(T lambda)
    {
        if (lambda <= T.Zero || lambda >= T.One)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lambda), lambda, "Lambda must be strictly between 0 and 1.");
        }

        _lambda = lambda;
    }

    /// <summary>
    /// Initializes a new instance with the default RiskMetrics daily decay factor (0.94).
    /// </summary>
    public ExponentiallyWeightedCovarianceEstimator()
        : this(T.CreateChecked(0.94))
    {
    }

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var means = CovarianceHelpers<T>.ComputeMeans(returns);

        // Compute unnormalized weights lambda^(T-1-k), then normalize to sum to 1.
        var weights = new T[t];
        var weightSum = T.Zero;
        var lambdaDouble = double.CreateChecked(_lambda);
        for (var k = 0; k < t; k++)
        {
            weights[k] = T.CreateChecked(Math.Pow(lambdaDouble, t - 1 - k));
            weightSum += weights[k];
        }

        for (var k = 0; k < t; k++)
        {
            weights[k] /= weightSum;
        }

        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    sum += weights[k] * (returns[k, i] - means[i]) * (returns[k, j] - means[j]);
                }

                cov[i, j] = sum;
                cov[j, i] = sum;
            }
        }

        return cov;
    }
}

/// <summary>
/// Exponentially weighted moving-average (EWMA) covariance estimator
/// (RiskMetrics, J.P. Morgan 1996). Recent observations carry geometrically
/// higher weight than old ones, parameterized by the decay factor <c>lambda in (0, 1)</c>.
/// PSD by construction whenever all weights are non-negative, which is
/// guaranteed for <c>lambda in (0, 1)</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>lambda = 0.94</c> is the RiskMetrics default for daily equity returns
/// (effective half-life ~ 11 days); <c>lambda = 0.97</c> is the default for
/// monthly data (half-life ~ 23 months).
/// </para>
/// </remarks>
public sealed class ExponentiallyWeightedCovarianceEstimator : ICovarianceEstimator
{
    private readonly ExponentiallyWeightedCovarianceEstimator<decimal> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExponentiallyWeightedCovarianceEstimator"/> type.
    /// </summary>
    /// <param name="lambda">Decay factor in (0, 1). Default 0.94 (RiskMetrics daily).</param>
    /// <exception cref="ArgumentOutOfRangeException">Lambda is not strictly between 0 and 1.</exception>
    public ExponentiallyWeightedCovarianceEstimator(decimal lambda = 0.94m)
    {
        _inner = new ExponentiallyWeightedCovarianceEstimator<decimal>(lambda);
    }

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
