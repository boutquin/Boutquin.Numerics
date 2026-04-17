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
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.LinearAlgebra;

/// <summary>
/// Principal component analysis — eigen-decomposition of a covariance matrix with
/// factor-model semantics (explained-variance vectors, loadings, scores). Built on
/// top of <see cref="JacobiEigenDecomposition{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Entry point: <see cref="Decompose"/> — takes a pre-computed covariance matrix.
/// Useful when the caller has applied a specialized estimator (Ledoit-Wolf shrinkage,
/// Newey-West, POET) and wants PCA on top of it. <see cref="PcaResult{T}.Mean"/> is
/// empty because no observation data was supplied.
/// </para>
/// <para>
/// Sign convention: for each eigenvector column, the sign is flipped so the component with
/// the largest absolute value is positive. Deterministic across runs on identical data.
/// </para>
/// <para>
/// Sorting convention: eigenvalues descending, eigenvectors reordered to match, so that
/// PC1 = largest eigenvalue. This matches the Hull Ch 33.6 level/slope/curvature convention
/// for yield curves.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class PrincipalComponentAnalysis<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Runs PCA on a pre-computed symmetric covariance matrix.
    /// </summary>
    /// <param name="covarianceMatrix">Symmetric N×N covariance matrix.</param>
    /// <returns>Eigenvalues (descending), eigenvectors (columns), and explained-variance vectors. <see cref="PcaResult{T}.Mean"/> is empty.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="covarianceMatrix"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">The matrix is not square or is not symmetric.</exception>
    public static PcaResult<T> Decompose(T[,] covarianceMatrix)
    {
        ArgumentNullException.ThrowIfNull(covarianceMatrix);

        ValidateSymmetric(covarianceMatrix);

        var eigen = JacobiEigenDecomposition<T>.Decompose(covarianceMatrix);
        ApplySignConvention(eigen.Vectors);

        var explained = ComputeExplainedVariance(eigen.Values);

        return new PcaResult<T>
        {
            Eigenvalues = eigen.Values,
            Eigenvectors = eigen.Vectors,
            ExplainedVarianceRatio = explained.Ratio,
            CumulativeExplainedVariance = explained.Cumulative,
            Mean = [],
        };
    }

    private static void ValidateSymmetric(T[,] matrix)
    {
        var n = matrix.GetLength(0);
        if (matrix.GetLength(1) != n)
        {
            throw new ArgumentException("Covariance matrix must be square.", nameof(matrix));
        }

        // Strict symmetry check — covariance matrices are symmetric by construction,
        // so any asymmetry is a caller bug. Exact equality is appropriate for decimal
        // inputs (no floating-point noise).
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                if (matrix[i, j] != matrix[j, i])
                {
                    throw new ArgumentException(
                        $"Covariance matrix is not symmetric at ({i}, {j}): " +
                        $"{matrix[i, j]} vs {matrix[j, i]}.",
                        nameof(matrix));
                }
            }
        }
    }

    internal static void ApplySignConvention(T[,] eigenvectors)
    {
        var rows = eigenvectors.GetLength(0);
        var cols = eigenvectors.GetLength(1);

        for (var k = 0; k < cols; k++)
        {
            var maxAbs = T.Zero;
            var maxAbsSign = 1;
            for (var i = 0; i < rows; i++)
            {
                var abs = T.Abs(eigenvectors[i, k]);
                if (abs > maxAbs)
                {
                    maxAbs = abs;
                    maxAbsSign = T.IsNegative(eigenvectors[i, k]) ? -1 : 1;
                }
            }

            if (maxAbsSign < 0)
            {
                for (var i = 0; i < rows; i++)
                {
                    eigenvectors[i, k] = -eigenvectors[i, k];
                }
            }
        }
    }

    internal static (T[] Ratio, T[] Cumulative) ComputeExplainedVariance(T[] eigenvalues)
    {
        var n = eigenvalues.Length;
        var total = T.Zero;
        for (var i = 0; i < n; i++)
        {
            // Numerical noise in near-zero eigenvalues can go slightly negative; clamp to zero for the ratio.
            if (eigenvalues[i] > T.Zero)
            {
                total += eigenvalues[i];
            }
        }

        var ratio = new T[n];
        var cumulative = new T[n];

        if (total == T.Zero)
        {
            return (ratio, cumulative);
        }

        var running = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var positive = eigenvalues[i] > T.Zero ? eigenvalues[i] : T.Zero;
            ratio[i] = positive / total;
            running += ratio[i];
            cumulative[i] = running;
        }

        return (ratio, cumulative);
    }
}

/// <summary>
/// Legacy facade delegating to <see cref="PrincipalComponentAnalysis{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// Also retains the <see cref="FromReturns"/> entry point that accepts
/// <see cref="ReturnsMatrix"/> (a decimal-typed type not yet migrated to generic math).
/// </summary>
public static class PrincipalComponentAnalysis
{
    /// <inheritdoc cref="PrincipalComponentAnalysis{T}.Decompose"/>
    public static PcaResult Decompose(decimal[,] covarianceMatrix)
    {
        var result = PrincipalComponentAnalysis<decimal>.Decompose(covarianceMatrix);
        return new PcaResult
        {
            Eigenvalues = result.Eigenvalues,
            Eigenvectors = result.Eigenvectors,
            ExplainedVarianceRatio = result.ExplainedVarianceRatio,
            CumulativeExplainedVariance = result.CumulativeExplainedVariance,
            Mean = result.Mean,
        };
    }

    /// <summary>
    /// Runs PCA on a returns matrix. Computes the sample covariance internally and
    /// populates <see cref="PcaResult.Mean"/> with per-column means.
    /// </summary>
    /// <param name="returns">
    /// Returns matrix — rows are observations, columns are variables. Accepts either a
    /// T×N <c>decimal[,]</c> or an asset-major <c>decimal[][]</c> via
    /// <see cref="ReturnsMatrix"/>'s implicit conversions.
    /// </param>
    /// <param name="standardize">
    /// When <see langword="true"/>, each column is divided by its sample standard deviation
    /// before covariance estimation (running PCA on the correlation matrix). Useful when
    /// variables have heterogeneous scales. Defaults to <see langword="false"/>.
    /// </param>
    /// <returns>PCA result including per-column <see cref="PcaResult.Mean"/>.</returns>
    /// <exception cref="ArgumentException"><paramref name="returns"/> has fewer than two observations, or <paramref name="standardize"/> is <see langword="true"/> and any column has zero variance.</exception>
    public static PcaResult FromReturns(ReturnsMatrix returns, bool standardize = false)
    {
        var t = returns.Observations;
        var n = returns.Assets;

        if (t < 2)
        {
            throw new ArgumentException(
                "Returns matrix must contain at least two observations.", nameof(returns));
        }

        // Per-column mean.
        var mean = new decimal[n];
        for (var j = 0; j < n; j++)
        {
            var sum = 0m;
            for (var i = 0; i < t; i++)
            {
                sum += returns[i, j];
            }

            mean[j] = sum / t;
        }

        // Per-column std (sample, divisor t − 1) — only needed when standardizing.
        decimal[]? std = null;
        if (standardize)
        {
            std = new decimal[n];
            for (var j = 0; j < n; j++)
            {
                var ss = 0m;
                for (var i = 0; i < t; i++)
                {
                    var dev = returns[i, j] - mean[j];
                    ss += dev * dev;
                }

                var variance = ss / (t - 1);
                std[j] = NumericPrecision<decimal>.Sqrt(variance);

                if (std[j] == 0m)
                {
                    throw new ArgumentException(
                        $"Cannot standardize: column {j} has zero variance.", nameof(returns));
                }
            }
        }

        // Sample covariance (or correlation when standardizing).
        var covariance = new decimal[n, n];
        for (var j1 = 0; j1 < n; j1++)
        {
            for (var j2 = j1; j2 < n; j2++)
            {
                var sum = 0m;
                for (var i = 0; i < t; i++)
                {
                    var dev1 = returns[i, j1] - mean[j1];
                    var dev2 = returns[i, j2] - mean[j2];
                    if (std is not null)
                    {
                        dev1 /= std[j1];
                        dev2 /= std[j2];
                    }

                    sum += dev1 * dev2;
                }

                var value = sum / (t - 1);
                covariance[j1, j2] = value;
                covariance[j2, j1] = value;
            }
        }

        var eigen = JacobiEigenDecomposition<decimal>.Decompose(covariance);
        PrincipalComponentAnalysis<decimal>.ApplySignConvention(eigen.Vectors);

        var explained = PrincipalComponentAnalysis<decimal>.ComputeExplainedVariance(eigen.Values);

        return new PcaResult
        {
            Eigenvalues = eigen.Values,
            Eigenvectors = eigen.Vectors,
            ExplainedVarianceRatio = explained.Ratio,
            CumulativeExplainedVariance = explained.Cumulative,
            Mean = mean,
        };
    }
}
