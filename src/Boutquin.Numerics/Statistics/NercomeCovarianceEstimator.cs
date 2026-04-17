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

using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Generic NERCOME — Nonparametric Eigenvalue-Regularized COvariance Matrix Estimator.
/// Splits the observation window into two disjoint halves: eigenvectors are
/// computed on the first half, then rotated eigenvalues are estimated on the
/// second half.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Reference: Abadir, K. M., Distaso, W. &amp; Zikes, F. (2014). "Design-free
/// estimation of variance matrices." Journal of Econometrics, 181(2), 165-180.
/// </para>
/// <para>
/// Algorithm (T x N returns, split fraction <c>f in (0, 1)</c>):
/// <list type="number">
/// <item><description>Partition rows into S1 and S2.</description></item>
/// <item><description>Compute sample covariance on S1; extract eigenvectors <c>V</c>.</description></item>
/// <item><description>Compute sample covariance on S2; call it <c>Sigma_2</c>.</description></item>
/// <item><description>Form rotated matrix <c>D = V^T . Sigma_2 . V</c>; extract diagonal <c>d</c>.</description></item>
/// <item><description>Return <c>Sigma_hat = V . diag(d) . V^T</c>.</description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class NercomeCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);

    private readonly T _splitFraction;

    /// <summary>
    /// Creates a NERCOME estimator with the specified split fraction.
    /// </summary>
    /// <param name="splitFraction">
    /// Fraction of rows allocated to the first half (eigenvector estimation).
    /// Must lie in (0, 1); defaults to 0.5.
    /// </param>
    public NercomeCovarianceEstimator(T? splitFraction = default)
    {
        splitFraction ??= T.CreateChecked(0.5);

        if (splitFraction <= T.Zero || splitFraction >= T.One)
        {
            throw new ArgumentOutOfRangeException(
                nameof(splitFraction),
                splitFraction,
                "Split fraction must lie strictly between 0 and 1.");
        }

        _splitFraction = splitFraction;
    }

    /// <summary>Split fraction used to partition observations.</summary>
    public T SplitFraction => _splitFraction;

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        var firstSize = int.CreateChecked(_splitFraction * T.CreateChecked(t));
        if (firstSize < 2)
        {
            firstSize = 2;
        }

        if (firstSize > t - 2)
        {
            firstSize = t - 2;
        }

        if (firstSize < 2 || (t - firstSize) < 2)
        {
            // Too few rows to split; fall back to sample covariance.
            return new SampleCovarianceEstimator<T>().Estimate(returns);
        }

        var first = new T[firstSize, n];
        for (var i = 0; i < firstSize; i++)
        {
            for (var j = 0; j < n; j++)
            {
                first[i, j] = returns[i, j];
            }
        }

        var secondSize = t - firstSize;
        var second = new T[secondSize, n];
        for (var i = 0; i < secondSize; i++)
        {
            for (var j = 0; j < n; j++)
            {
                second[i, j] = returns[firstSize + i, j];
            }
        }

        var cov1 = new SampleCovarianceEstimator<T>().Estimate(first);
        var cov2 = new SampleCovarianceEstimator<T>().Estimate(second);

        // Eigendecomposition of cov1 -> V.
        var eigen = JacobiEigenDecomposition<T>.Decompose(cov1);
        var v = eigen.Vectors;

        // D = V^T . cov2 . V.
        var vtCov2 = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < n; k++)
                {
                    sum += v[k, i] * cov2[k, j];
                }

                vtCov2[i, j] = sum;
            }
        }

        var diag = new T[n];
        for (var i = 0; i < n; i++)
        {
            var sum = T.Zero;
            for (var k = 0; k < n; k++)
            {
                sum += vtCov2[i, k] * v[k, i];
            }

            // Regularize: rotated eigenvalues can occasionally be slightly
            // negative for small samples due to sign-flipping; floor at 0.
            diag[i] = sum < T.Zero ? T.Zero : sum;
        }

        var reconstructed = CovarianceHelpers<T>.ReconstructFromEigen(diag, v);

        // Enforce exact symmetry.
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var avg = (reconstructed[i, j] + reconstructed[j, i]) / s_two;
                reconstructed[i, j] = avg;
                reconstructed[j, i] = avg;
            }
        }

        return reconstructed;
    }
}

/// <summary>
/// NERCOME — Nonparametric Eigenvalue-Regularized COvariance Matrix Estimator.
/// Splits the observation window into two disjoint halves: eigenvectors are
/// computed on the first half, then rotated eigenvalues are estimated on the
/// second half. Breaking the in-sample correlation between eigenvectors and
/// eigenvalues yields a design-free covariance estimator with smaller
/// eigenvalue-spread bias than the raw sample covariance.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Abadir, K. M., Distaso, W. &amp; Zikes, F. (2014). "Design-free
/// estimation of variance matrices." Journal of Econometrics, 181(2), 165-180.
/// </para>
/// </remarks>
public sealed class NercomeCovarianceEstimator : ICovarianceEstimator
{
    private readonly NercomeCovarianceEstimator<decimal> _inner;

    /// <summary>
    /// Creates a NERCOME estimator with the specified split fraction.
    /// </summary>
    /// <param name="splitFraction">
    /// Fraction of rows allocated to the first half (eigenvector estimation).
    /// Must lie in (0, 1); defaults to 0.5.
    /// </param>
    public NercomeCovarianceEstimator(decimal splitFraction = 0.5m)
    {
        _inner = new NercomeCovarianceEstimator<decimal>(splitFraction);
    }

    /// <summary>Split fraction used to partition observations.</summary>
    public decimal SplitFraction => _inner.SplitFraction;

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
