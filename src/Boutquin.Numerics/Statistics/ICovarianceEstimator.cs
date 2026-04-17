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
/// Generic contract for estimators that produce an N-by-N symmetric covariance matrix
/// from a T-by-N return matrix (T observations, N assets).
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// <strong>Universal contract (all implementations):</strong> the returned
/// matrix is square N-by-N and numerically symmetric, where N is the column count
/// of the input. Implementations do not mutate the input matrix.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public interface ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Estimates the N-by-N covariance matrix from a T-by-N return series.
    /// </summary>
    /// <param name="returns">
    /// T-by-N matrix of return observations (rows = time, columns = assets). Must have
    /// at least the number of observations required by the concrete estimator
    /// (typically T >= 2; higher for shrinkage / denoising methods).
    /// </param>
    /// <returns>
    /// N-by-N symmetric covariance matrix in <c>T[,]</c> layout.
    /// </returns>
    T[,] Estimate(T[,] returns);
}

/// <summary>
/// Contract for estimators that produce an N-by-N symmetric covariance matrix
/// from a T-by-N return matrix (T observations, N assets). Implementations
/// range from the classical sample estimator to shrinkage, factor-model, and
/// eigenvalue-denoising variants; consumers dispatch on the concrete type
/// based on sample regime and structural assumptions.
/// </summary>
/// <remarks>
/// <para>
/// <strong>Universal contract (all implementations):</strong> the returned
/// matrix is square N-by-N and numerically symmetric (<c>|Sigma_ij - Sigma_ji| &lt;= roundoff</c>),
/// where N is the column count of the input. Implementations do not mutate
/// the input matrix.
/// </para>
/// <para>
/// <strong>Per-implementation PSD contract (NOT universal).</strong> The
/// interface does not promise positive-semidefiniteness; each estimator states
/// its own guarantee:
/// </para>
/// <list type="bullet">
///   <item><see cref="SampleCovarianceEstimator"/>, <see cref="LedoitWolfShrinkageEstimator"/>,
///         <see cref="LedoitWolfConstantCorrelationEstimator"/>, <see cref="LedoitWolfSingleFactorEstimator"/>,
///         <see cref="OracleApproximatingShrinkageEstimator"/>, <see cref="QuadraticInverseShrinkageEstimator"/>,
///         <see cref="DenoisedCovarianceEstimator"/>, <see cref="TracyWidomDenoisedCovarianceEstimator"/>,
///         <see cref="NercomeCovarianceEstimator"/>, <see cref="DoublySparseEstimator"/> — PSD by construction.</item>
///   <item><see cref="ExponentiallyWeightedCovarianceEstimator"/> — PSD when weights are non-negative,
///         which is ensured for lambda in (0, 1); outside this range positivity of weights can break.</item>
///   <item><see cref="DetonedCovarianceEstimator"/> — <em>conditionally</em> PSD. PC1 shrinkage can
///         invert eigenvalue ordering and yield a non-PSD result; callers must project if PSD
///         is required downstream.</item>
///   <item><see cref="PoetCovarianceEstimator"/> — no PSD guarantee. Soft-thresholding of the
///         residual covariance can break positivity when the threshold multiplier is aggressive.</item>
/// </list>
/// <para>
/// When a hard PSD guarantee is required (e.g., before a Cholesky factorization
/// or drawing from the implied Gaussian), wrap the result with
/// <see cref="LinearAlgebra.NearestPsdProjection"/>.
/// </para>
/// <para>
/// The canonical input layout is T-by-N. All downstream estimators operate on that
/// layout; the jagged overload exists to bridge Trading-shaped inputs
/// without forcing callers to allocate.
/// </para>
/// </remarks>
public interface ICovarianceEstimator : ICovarianceEstimator<decimal>
{
    /// <summary>
    /// Estimates from a <see cref="ReturnsMatrix"/> input accepting either T-by-N
    /// or asset-major layouts. The default implementation materializes a T-by-N
    /// view via <see cref="ReturnsMatrix.AsTimeByAsset"/> and delegates to
    /// <see cref="ICovarianceEstimator{T}.Estimate"/>; implementations that prefer
    /// asset-major traversal may override to avoid the copy.
    /// </summary>
    /// <param name="returns">Returns matrix with either layout.</param>
    /// <returns>N-by-N covariance matrix.</returns>
    decimal[,] Estimate(ReturnsMatrix returns) => ((ICovarianceEstimator<decimal>)this).Estimate(returns.AsTimeByAsset());
}
