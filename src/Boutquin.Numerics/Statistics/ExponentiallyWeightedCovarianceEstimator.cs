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
/// Numerical note: geometric weights retain a separate radix exponent while their
/// significands and moment products run in <typeparamref name="T"/>. A weight below the
/// type's resolution can therefore still contribute a representable covariance.
/// Half inputs are widened exactly to
/// double for the weight and moment reductions, then rounded once at the output.
/// Each column is measured from its newest observation (the heaviest weight), so a remote
/// low-weight outlier cannot corrupt the weighted center, and every weighted cross-product
/// is multiplied in magnitude order so that a representable covariance entry never depends
/// on the order of the two columns.
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
        if (!T.IsFinite(lambda) || lambda <= T.Zero || lambda >= T.One)
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
        if (typeof(T) == typeof(Half))
        {
            return CovarianceHelpers<T>.EstimateHalfShrinkage(
                returns, new ExponentiallyWeightedCovarianceEstimator<double>(double.CreateChecked(_lambda)));
        }

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        // Retain each geometric weight's exponent until after its moment product:
        // lambda^k can underflow even when lambda^k * deviation_i * deviation_j fits.
        var weights = new T[t];
        var weightExponents = new long[t];
        weights[^1] = T.One;
        var weightSum = T.One;
        for (var k = t - 2; k >= 0; k--)
        {
            (weights[k], weightExponents[k]) = ScaledArithmetic<T>.MultiplyScaled(
                weights[k + 1], weightExponents[k + 1], _lambda);
            weightSum += ScaledArithmetic<T>.Rescale(weights[k], int.CreateSaturating(weightExponents[k]));
        }

        // Center each column on its newest observation before weighting. The newest row
        // carries the largest weight, so every other observation is measured from the
        // heaviest anchor: the weighted center's rounding error stays proportional to the
        // weighted spread. A first-row anchor has the smallest weight, and when it sits far
        // from the bulk its rounded center leaves a spurious residual that dwarfs the
        // negligible variance the anchor itself contributes. Anchored differences also keep
        // small decimal deviations from being rounded away at a large offset.
        var meanResiduals = new T[n];
        var meanExponents = new long[n];
        var deviationExponents = new long[t, n];
        var scales = new T[n];
        var centered = new T[t, n];
        for (var i = 0; i < n; i++)
        {
            // Halving mixed-sign columns before subtraction keeps their range
            // representable. Same-sign columns retain the unscaled anchor,
            // preserving small decimal deviations at large common offsets.
            scales[i] = T.One;
            var hasPositive = false;
            var hasNegative = false;
            var magnitude = T.Zero;
            for (var k = 0; k < t; k++)
            {
                hasPositive |= returns[k, i] > T.Zero;
                hasNegative |= returns[k, i] < T.Zero;
                magnitude = T.Max(magnitude, T.Abs(returns[k, i]));
            }

            // Small columns cannot overflow subtraction; halving them could
            // instead erase a representable subnormal covariance product.
            if (hasPositive && hasNegative && magnitude > T.One)
            {
                scales[i] = T.CreateChecked(2);
            }

            var anchor = returns[t - 1, i] / scales[i];
            for (var k = 0; k < t; k++)
            {
                centered[k, i] = returns[k, i] / scales[i] - anchor;
                var term = ScaledArithmetic<T>.ProductScaled(weights[k], centered[k, i], T.One, T.One);
                ScaledArithmetic<T>.AddScaled(ref meanResiduals[i], ref meanExponents[i],
                    term.Significand, term.Exponent + weightExponents[k]);
            }

            (meanResiduals[i], meanExponents[i]) = ScaledArithmetic<T>.DivideScaled(
                meanResiduals[i], meanExponents[i], weightSum);
            for (var k = 0; k < t; k++)
            {
                var deviation = T.Zero;
                long exponent = 0;
                ScaledArithmetic<T>.AddScaled(ref deviation, ref exponent, centered[k, i], 0);
                ScaledArithmetic<T>.AddScaled(ref deviation, ref exponent, -meanResiduals[i], meanExponents[i]);
                centered[k, i] = deviation;
                deviationExponents[k, i] = exponent;
            }
        }

        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                long exponent = 0;
                for (var k = 0; k < t; k++)
                {
                    var term = ScaledArithmetic<T>.ProductScaled(weights[k], centered[k, i], centered[k, j], T.One);
                    ScaledArithmetic<T>.AddScaled(ref sum, ref exponent, term.Significand,
                        term.Exponent + weightExponents[k] + deviationExponents[k, i] + deviationExponents[k, j]);
                }

                var value = ScaledArithmetic<T>.ProductQuotient(sum, scales[i], scales[j], weightSum, exponent);
                cov[i, j] = value;
                cov[j, i] = value;
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
/// (effective half-life ~ 11 days); <c>lambda = 0.97</c> is the RiskMetrics
/// decay for one-month forecasts built from daily returns (half-life ~ 23 days).
/// </para>
/// </remarks>
[Obsolete(
    "Use ExponentiallyWeightedCovarianceEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
