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
/// Shared implementation primitives for covariance estimators. All estimators
/// in this namespace consume returns in T-by-N layout (rows = observations,
/// columns = assets) and produce an N-by-N symmetric covariance matrix.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
internal static class CovarianceHelpers<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Validates a T-by-N return matrix. Throws <see cref="ArgumentException"/>
    /// when the matrix has fewer than two observations or fewer than one asset.
    /// </summary>
    internal static void ValidateReturns(T[,] returns)
    {
        ArgumentNullException.ThrowIfNull(returns);
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        if (n < 1)
        {
            throw new ArgumentException("Returns matrix must contain at least one asset.", nameof(returns));
        }

        if (t < 2)
        {
            throw new ArgumentException("Returns matrix must contain at least two observations.", nameof(returns));
        }
    }

    /// <summary>
    /// Computes per-asset arithmetic means down the time axis.
    /// </summary>
    internal static T[] ComputeMeans(T[,] returns)
    {
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var tCount = T.CreateChecked(t);
        var means = new T[n];
        for (var j = 0; j < n; j++)
        {
            var sum = T.Zero;
            for (var i = 0; i < t; i++)
            {
                sum += returns[i, j];
            }

            means[j] = sum / tCount;
        }

        return means;
    }

    /// <summary>
    /// Computes the unbiased sample covariance matrix (N-1 divisor).
    /// </summary>
    internal static T[,] ComputeSampleCovariance(T[,] returns, T[] means)
    {
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var divisor = T.CreateChecked(t - 1);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    sum += (returns[k, i] - means[i]) * (returns[k, j] - means[j]);
                }

                var value = sum / divisor;
                cov[i, j] = value;
                cov[j, i] = value;
            }
        }

        return cov;
    }

    /// <summary>
    /// Converts a covariance matrix to a correlation matrix and returns the
    /// per-asset standard deviations as a side product.
    /// </summary>
    /// <remarks>
    /// Zero-variance assets (<c>sigma_i = 0</c>) are handled by writing <c>1</c> on the
    /// diagonal and <c>0</c> elsewhere for that asset's row/column — preserving
    /// the identity-like structure for a constant series rather than propagating
    /// <c>NaN</c> via <c>0 / 0</c>.
    /// </remarks>
    internal static (T[,] Correlation, T[] StdDevs) CovarianceToCorrelation(T[,] cov)
    {
        var n = cov.GetLength(0);
        var stdDevs = new T[n];
        for (var i = 0; i < n; i++)
        {
            stdDevs[i] = NumericPrecision<T>.Sqrt(cov[i, i]);
        }

        var corr = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (stdDevs[i] == T.Zero || stdDevs[j] == T.Zero)
                {
                    corr[i, j] = i == j ? T.One : T.Zero;
                }
                else
                {
                    corr[i, j] = cov[i, j] / (stdDevs[i] * stdDevs[j]);
                }
            }
        }

        return (corr, stdDevs);
    }

    /// <summary>
    /// Converts a correlation matrix back to covariance using per-asset standard deviations.
    /// </summary>
    internal static T[,] CorrelationToCovariance(T[,] corr, T[] stdDevs)
    {
        var n = corr.GetLength(0);
        var cov = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                cov[i, j] = corr[i, j] * stdDevs[i] * stdDevs[j];
            }
        }

        return cov;
    }

    /// <summary>
    /// Marcenko-Pastur upper spectral edge for the empirical
    /// eigenvalue distribution of a sample correlation matrix under the null
    /// hypothesis of IID noise, where <c>q = T / N</c> is the inverse
    /// concentration ratio.
    /// </summary>
    /// <remarks>
    /// Eigenvalues below the upper bound are indistinguishable from noise in the asymptotic
    /// limit; eigenvalues above it carry signal. Used as the threshold in
    /// <see cref="DenoisedCovarianceEstimator"/> and
    /// <see cref="DetonedCovarianceEstimator"/>. For finite-sample refinement,
    /// <see cref="TracyWidomDenoisedCovarianceEstimator"/> adds an N^(-2/3)
    /// correction. Returns the maximum value of <typeparamref name="T"/> when <c>q = 0</c>
    /// (no observations — threshold is meaningless and no eigenvalue will
    /// exceed it).
    /// </remarks>
    internal static T MarcenkoPasturUpperBound(T q)
    {
        var sqrtQ = NumericPrecision<T>.Sqrt(q);
        if (sqrtQ == T.Zero)
        {
            // For decimal: decimal.MaxValue; for double: double.MaxValue; etc.
            // Use a very large sentinel. T doesn't have MaxValue directly,
            // but we can replicate the original behaviour by returning a
            // sufficiently large value. The original returned decimal.MaxValue.
            // We use CreateChecked from the double representation.
            return T.CreateChecked(decimal.MaxValue);
        }

        var bound = T.One + T.One / sqrtQ;
        return bound * bound;
    }

    /// <summary>
    /// Reconstructs a symmetric matrix from its eigendecomposition
    /// <c>A = V . diag(lambda) . V^T</c>, given eigenvalues and column-eigenvectors.
    /// </summary>
    /// <remarks>
    /// Symmetry is enforced by computing each entry in the upper triangle once
    /// and assigning the symmetric partner directly, rather than computing
    /// both entries independently and relying on floating-point
    /// associativity to produce identical values. No eigenvalue ordering is
    /// assumed by the reconstruction; the caller controls sign/magnitude
    /// modifications (clipping, shrinkage, detoning) before invoking.
    /// </remarks>
    internal static T[,] ReconstructFromEigen(T[] values, T[,] vectors)
    {
        var n = values.Length;
        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < n; k++)
                {
                    sum += values[k] * vectors[i, k] * vectors[j, k];
                }

                result[i, j] = sum;
                result[j, i] = sum;
            }
        }

        return result;
    }
}

/// <summary>
/// Legacy decimal facade for <see cref="CovarianceHelpers{T}"/>. Delegates all
/// operations to <see cref="CovarianceHelpers{T}"/> with <c>T = decimal</c>.
/// Existing estimators that reference <c>CovarianceHelpers.Method()</c> continue
/// to compile unchanged.
/// </summary>
internal static class CovarianceHelpers
{
    /// <inheritdoc cref="CovarianceHelpers{T}.ValidateReturns"/>
    internal static void ValidateReturns(decimal[,] returns)
        => CovarianceHelpers<decimal>.ValidateReturns(returns);

    /// <inheritdoc cref="CovarianceHelpers{T}.ComputeMeans"/>
    internal static decimal[] ComputeMeans(decimal[,] returns)
        => CovarianceHelpers<decimal>.ComputeMeans(returns);

    /// <inheritdoc cref="CovarianceHelpers{T}.ComputeSampleCovariance"/>
    internal static decimal[,] ComputeSampleCovariance(decimal[,] returns, decimal[] means)
        => CovarianceHelpers<decimal>.ComputeSampleCovariance(returns, means);

    /// <inheritdoc cref="CovarianceHelpers{T}.CovarianceToCorrelation"/>
    internal static (decimal[,] Correlation, decimal[] StdDevs) CovarianceToCorrelation(decimal[,] cov)
        => CovarianceHelpers<decimal>.CovarianceToCorrelation(cov);

    /// <inheritdoc cref="CovarianceHelpers{T}.CorrelationToCovariance"/>
    internal static decimal[,] CorrelationToCovariance(decimal[,] corr, decimal[] stdDevs)
        => CovarianceHelpers<decimal>.CorrelationToCovariance(corr, stdDevs);

    /// <inheritdoc cref="CovarianceHelpers{T}.MarcenkoPasturUpperBound"/>
    internal static decimal MarcenkoPasturUpperBound(decimal q)
        => CovarianceHelpers<decimal>.MarcenkoPasturUpperBound(q);

    /// <inheritdoc cref="CovarianceHelpers{T}.ReconstructFromEigen"/>
    internal static decimal[,] ReconstructFromEigen(decimal[] values, decimal[,] vectors)
        => CovarianceHelpers<decimal>.ReconstructFromEigen(values, vectors);
}
