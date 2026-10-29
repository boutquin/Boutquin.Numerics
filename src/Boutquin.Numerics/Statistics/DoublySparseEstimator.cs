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
/// This is a declared repository variant: MP eigenvalue clipping followed by a
/// one-shot absolute threshold on signal-eigenvector entries and renormalisation.
/// It does not implement the Kutateladze–Seregina estimator: there is no truncated
/// power iteration, rank or support-size input, least-squares eigenvalue
/// re-estimation, or residual thresholding.
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
/// <para>
/// <b>PSD guarantee.</b> The reconstructed correlation matrix has the form
/// <c>Σᵢ∈signal λᵢ uᵢuᵢᵀ + λ̄noise Σⱼ∈noise vⱼvⱼᵀ</c>, where <c>uᵢ</c> are the
/// re-normalized (post-threshold) signal eigenvectors and <c>vⱼ</c> are the original
/// noise eigenvectors. Every rank-1 term <c>wwᵀ</c> is positive semi-definite for any
/// non-zero vector <c>w</c>, and both <c>λᵢ</c> (signal eigenvalues, positive by
/// Marcenko-Pastur selection) and <c>λ̄noise</c> (arithmetic mean of noise eigenvalues,
/// non-negative) are non-negative scalars. A non-negative linear combination of PSD
/// matrices is PSD, so the result is PSD by construction — regardless of how aggressively
/// the eigenvectors were thresholded. The fallback path (triggered when all entries of a
/// signal eigenvector are thresholded to zero) substitutes the original eigenvector,
/// preserving the same guarantee.
/// </para>
/// <para>Half inputs widen exactly to double for estimation; only the final covariance is rounded back.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class DoublySparseEstimator<T> : ICovarianceEstimator<T>
    where T : struct, IFloatingPoint<T>
{
    private readonly T _eigenvectorThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="DoublySparseEstimator{T}"/> type
    /// with the default 0.1 eigenvector threshold.
    /// </summary>
    public DoublySparseEstimator()
        : this(T.CreateChecked(0.1))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DoublySparseEstimator{T}"/> type.
    /// </summary>
    /// <param name="eigenvectorThreshold">
    /// Absolute threshold for eigenvector entries. Values below this threshold are
    /// set to zero. <see langword="null"/> (including the <see langword="default"/> literal) selects 0.1.
    /// </param>
    public DoublySparseEstimator(T? eigenvectorThreshold = null)
    {
        var threshold = eigenvectorThreshold ?? T.CreateChecked(0.1);

        if (!T.IsFinite(threshold) || threshold < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(eigenvectorThreshold),
                threshold,
                "Eigenvector threshold must be a non-negative number.");
        }

        _eigenvectorThreshold = threshold;
    }

    /// <inheritdoc/>
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            // Keep the correlation eigensolver and observation ratio out of Half arithmetic.
            return CovarianceHelpers<T>.EstimateHalfShrinkage(
                returns, new DoublySparseEstimator<double>(double.CreateChecked(_eigenvectorThreshold)));
        }

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        // Steps 1-2: Sample covariance converted to correlation, formed per column so a column
        // whose variance leaves the type's range keeps its correlations.
        var (corr, stdDevs) = CovarianceHelpers<T>.SampleCorrelation(returns);

        // Step 3: Eigendecompose.
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);
        var eigenvalues = eigen.Values;
        var eigenvectors = eigen.Vectors;

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
            if (eigenvalues[i] <= lambdaPlus)
            {
                noiseSum += eigenvalues[i];
                noiseCount++;
            }
        }

        var noiseAvg = noiseCount > 0 ? noiseSum / T.CreateChecked(noiseCount) : T.Zero;

        for (var i = 0; i < n; i++)
        {
            if (eigenvalues[i] > lambdaPlus)
            {
                // Signal: keep eigenvalue, threshold eigenvector.
                cleanValues[i] = eigenvalues[i];
                for (var j = 0; j < n; j++)
                {
                    var entry = eigenvectors[j, i];
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
                else
                {
                    // Every entry was thresholded away — fall back to the original eigenvector
                    // so the component and its information are preserved rather than silently zeroed.
                    // Singularity alone does not violate PSD.
                    for (var j = 0; j < n; j++)
                    {
                        sparseVectors[j, i] = eigenvectors[j, i];
                    }
                }
            }
            else
            {
                // Noise: replace eigenvalue with average, keep eigenvector as-is.
                cleanValues[i] = noiseAvg;
                for (var j = 0; j < n; j++)
                {
                    sparseVectors[j, i] = eigenvectors[j, i];
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
            for (var j = i; j < n; j++)
            {
                result[i, j] = cleanCorr[i, j] * stdDevs[i] * stdDevs[j];
                result[j, i] = result[i, j];
            }
        }

        return result;
    }
}

/// <summary>
/// Repository-variant doubly-sparse covariance estimator (DSCE) — decomposes the sample covariance
/// into a signal component with sparsified eigenvectors (hard-thresholded
/// entries, re-normalized to unit norm) and a noise component with eigenvalues
/// replaced by their average. PSD by construction.
/// </summary>
/// <remarks>
/// <para>
/// This is a declared repository variant: MP eigenvalue clipping followed by a
/// one-shot absolute threshold on signal-eigenvector entries and renormalisation.
/// It does not implement the Kutateladze–Seregina estimator: there is no truncated
/// power iteration, rank or support-size input, least-squares eigenvalue
/// re-estimation, or residual thresholding.
/// </para>
/// </remarks>
[Obsolete(
    "Use DoublySparseEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
