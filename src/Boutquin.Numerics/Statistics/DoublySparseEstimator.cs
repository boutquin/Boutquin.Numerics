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
using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Generic doubly-sparse covariance estimator (DSCE) — decomposes the sample covariance
/// into a signal component with sparsified eigenvectors (hard-thresholded
/// entries, re-normalized to unit norm) and a noise component with eigenvalues
/// replaced by their average.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/>.
/// </para>
/// <para>
/// Reference: <em>Econometrics and Statistics</em> (2024) — "Doubly Sparse
/// Estimation of High-Dimensional Covariance Matrices".
/// </para>
/// <para>
/// Algorithm:
/// <list type="number">
///   <item>Compute sample covariance and convert to correlation.</item>
///   <item>Eigendecompose the correlation matrix via Jacobi rotations.</item>
///   <item>Partition eigenvalues into signal and noise using the Marcenko-Pastur upper edge.</item>
///   <item>Hard-threshold each signal eigenvector entry; re-normalize to unit norm.</item>
///   <item>Replace noise eigenvalues with their arithmetic mean.</item>
///   <item>Reconstruct and convert back to covariance.</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class DoublySparseEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private readonly T _eigenvectorThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="DoublySparseEstimator{T}"/> type.
    /// </summary>
    /// <param name="eigenvectorThreshold">
    /// Absolute threshold for eigenvector entries. Values below this threshold are
    /// set to zero. Default is 0.1.
    /// </param>
    public DoublySparseEstimator(T? eigenvectorThreshold = default)
    {
        _eigenvectorThreshold = eigenvectorThreshold ?? T.CreateChecked(0.1);
    }

    /// <inheritdoc/>
    public T[,] Estimate(T[,] returns)
    {
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        if (t < 2)
        {
            throw new ArgumentException("At least 2 observations are required.", nameof(returns));
        }

        // Step 1: Compute sample covariance.
        var means = CovarianceHelpers<T>.ComputeMeans(returns);
        var sampleCov = CovarianceHelpers<T>.ComputeSampleCovariance(returns, means);

        // Step 2: Convert to correlation.
        var (corr, stdDevs) = CovarianceHelpers<T>.CovarianceToCorrelation(sampleCov);

        // Step 3: Eigendecompose.
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);

        // Step 4: Marcenko-Pastur bound.
        var q = T.CreateChecked(t) / T.CreateChecked(n);
        var lambdaPlus = CovarianceHelpers<T>.MarcenkoPasturUpperBound(q);

        // Step 5: Separate signal and noise, threshold eigenvectors.
        var cleanValues = new T[n];
        var sparseVectors = new T[n, n];

        // Count noise eigenvalues and compute their average.
        var noiseSum = T.Zero;
        var noiseCount = 0;
        for (var i = 0; i < n; i++)
        {
            if (eigen.Values[i] <= lambdaPlus)
            {
                noiseSum += eigen.Values[i];
                noiseCount++;
            }
        }

        var noiseAvg = noiseCount > 0 ? noiseSum / T.CreateChecked(noiseCount) : T.Zero;

        for (var i = 0; i < n; i++)
        {
            if (eigen.Values[i] > lambdaPlus)
            {
                // Signal: keep eigenvalue, threshold eigenvector.
                cleanValues[i] = eigen.Values[i];
                for (var j = 0; j < n; j++)
                {
                    var entry = eigen.Vectors[j, i];
                    sparseVectors[j, i] = T.Abs(entry) >= _eigenvectorThreshold ? entry : T.Zero;
                }

                // Re-normalize the thresholded eigenvector.
                var norm = T.Zero;
                for (var j = 0; j < n; j++)
                {
                    norm += sparseVectors[j, i] * sparseVectors[j, i];
                }

                if (norm > T.Zero)
                {
                    var invNorm = T.One / NumericPrecision<T>.Sqrt(norm);
                    for (var j = 0; j < n; j++)
                    {
                        sparseVectors[j, i] *= invNorm;
                    }
                }
            }
            else
            {
                // Noise: replace eigenvalue with average, keep eigenvector as-is.
                cleanValues[i] = noiseAvg;
                for (var j = 0; j < n; j++)
                {
                    sparseVectors[j, i] = eigen.Vectors[j, i];
                }
            }
        }

        // Step 6: Reconstruct correlation = V . diag(lambda) . V^T.
        var cleanCorr = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < n; k++)
                {
                    sum += sparseVectors[i, k] * cleanValues[k] * sparseVectors[j, k];
                }

                cleanCorr[i, j] = sum;
                cleanCorr[j, i] = sum;
            }
        }

        // Step 7: Convert back to covariance.
        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                result[i, j] = cleanCorr[i, j] * stdDevs[i] * stdDevs[j];
            }
        }

        return result;
    }
}

/// <summary>
/// Doubly-sparse covariance estimator (DSCE) — decomposes the sample covariance
/// into a signal component with sparsified eigenvectors (hard-thresholded
/// entries, re-normalized to unit norm) and a noise component with eigenvalues
/// replaced by their average. PSD by construction.
/// </summary>
/// <remarks>
/// <para>
/// Reference: <em>Econometrics and Statistics</em> (2024) — "Doubly Sparse
/// Estimation of High-Dimensional Covariance Matrices".
/// </para>
/// </remarks>
public sealed class DoublySparseEstimator : ICovarianceEstimator
{
    private readonly DoublySparseEstimator<decimal> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="DoublySparseEstimator"/> type.
    /// </summary>
    /// <param name="eigenvectorThreshold">
    /// Absolute threshold for eigenvector entries. Values below this threshold are
    /// set to zero. Default is 0.1.
    /// </param>
    public DoublySparseEstimator(decimal eigenvectorThreshold = 0.1m)
    {
        _inner = new DoublySparseEstimator<decimal>(eigenvectorThreshold);
    }

    /// <inheritdoc/>
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
