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
/// Generic unbiased sample covariance estimator <c>S = (T - 1)^-1 . (X - mu)^T(X - mu)</c>
/// using Bessel's <c>T - 1</c> divisor.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// PSD by construction for any returns
/// matrix with at least <c>N + 1</c> observations; rank-deficient (still PSD,
/// but singular) when <c>T &lt;= N</c>.
/// </para>
/// <para>
/// Sample regime: this is the baseline consistent estimator as <c>T -> infinity</c>
/// with <c>N</c> fixed. In the high-dimensional regime where the concentration
/// ratio <c>c = N / T</c> is non-negligible, the sample covariance is known to
/// have poorly-behaved extreme eigenvalues.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class SampleCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        var means = CovarianceHelpers<T>.ComputeMeans(returns);
        return CovarianceHelpers<T>.ComputeSampleCovariance(returns, means);
    }
}

/// <summary>
/// Unbiased sample covariance estimator <c>S = (T - 1)^-1 . (X - mu)^T(X - mu)</c>
/// using Bessel's <c>T - 1</c> divisor. PSD by construction for any returns
/// matrix with at least <c>N + 1</c> observations; rank-deficient (still PSD,
/// but singular) when <c>T &lt;= N</c>.
/// </summary>
/// <remarks>
/// <para>
/// Sample regime: this is the baseline consistent estimator as <c>T -> infinity</c>
/// with <c>N</c> fixed. In the high-dimensional regime where the concentration
/// ratio <c>c = N / T</c> is non-negligible, the sample covariance is known to
/// have poorly-behaved extreme eigenvalues (Marcenko and Pastur 1967): the
/// largest eigenvalue is biased upward and the smallest downward, inflating
/// portfolio-weight magnitudes in mean-variance optimization.
/// </para>
/// <para>
/// For small or high-dimensional samples prefer a regularized alternative:
/// <list type="bullet">
///   <item><see cref="LedoitWolfShrinkageEstimator"/> — linear shrinkage to scaled identity.</item>
///   <item><see cref="LedoitWolfConstantCorrelationEstimator"/> / <see cref="LedoitWolfSingleFactorEstimator"/> — structured targets.</item>
///   <item><see cref="QuadraticInverseShrinkageEstimator"/> — nonlinear eigenvalue shrinkage.</item>
///   <item><see cref="DenoisedCovarianceEstimator"/> / <see cref="TracyWidomDenoisedCovarianceEstimator"/> — eigenvalue cleaning.</item>
/// </list>
/// </para>
/// <para>
/// Numerical convention: <c>decimal[,]</c> in, <c>decimal[,]</c> out. Bessel's
/// divisor removes the O(1/T) bias under the independence assumption; for
/// dependent observations consider <see cref="NeweyWestVariance"/> on individual
/// series.
/// </para>
/// </remarks>
public sealed class SampleCovarianceEstimator : ICovarianceEstimator
{
    private readonly SampleCovarianceEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
