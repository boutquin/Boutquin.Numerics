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

namespace Boutquin.Numerics.LinearAlgebra;

/// <summary>
/// Result of a principal component analysis — eigenvalues sorted descending,
/// matching eigenvectors as columns, explained-variance vectors, and (when
/// applicable) the per-variable mean used for centering.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+√. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// The factor-model semantics are the standard form consumers expect:
/// <c>Eigenvectors[:, k]</c> are the <em>loadings</em> for principal component
/// <c>k</c>; projecting a centered observation vector <c>x − Mean</c> onto those
/// loadings yields the scalar <em>score</em> of the observation on that PC.
/// </para>
/// <para>
/// <see cref="Mean"/> is populated only when <c>PrincipalComponentAnalysis.FromReturns</c>
/// is used (it computes per-column means as part of covariance estimation).
/// The <c>Decompose(covarianceMatrix)</c> entry point does not have observation data,
/// so <see cref="Mean"/> is returned empty there.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed record PcaResult<T>
    where T : IFloatingPoint<T>
{
    /// <summary>Eigenvalues of the covariance matrix, sorted in descending order.</summary>
    public required T[] Eigenvalues { get; init; }

    /// <summary>Eigenvector matrix — column <c>k</c> is the loading vector for principal component <c>k</c>.</summary>
    public required T[,] Eigenvectors { get; init; }

    /// <summary>Per-component fraction of total variance, <c>λₖ / Σⱼ λⱼ</c>.</summary>
    public required T[] ExplainedVarianceRatio { get; init; }

    /// <summary>
    /// Cumulative sum of <see cref="ExplainedVarianceRatio"/>. The last element equals 1
    /// (up to rounding) for any full-rank covariance matrix.
    /// </summary>
    public required T[] CumulativeExplainedVariance { get; init; }

    /// <summary>
    /// Per-variable mean used to center observations before projection. Empty when the
    /// result came from <c>Decompose(covarianceMatrix)</c> (no observation data available).
    /// </summary>
    public required T[] Mean { get; init; }

    /// <summary>
    /// Smallest <c>k</c> such that the cumulative explained variance reaches <paramref name="threshold"/>.
    /// </summary>
    /// <param name="threshold">Target cumulative variance, in <c>(0, 1]</c>.</param>
    /// <returns>
    /// <c>k ∈ [1, N]</c> where <c>N = Eigenvalues.Length</c>. Always at least 1 for a
    /// non-empty result.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is not in <c>(0, 1]</c>.</exception>
    public int NumComponentsForExplainedVariance(T threshold)
    {
        if (threshold <= T.Zero || threshold > T.One)
        {
            throw new ArgumentOutOfRangeException(
                nameof(threshold),
                threshold,
                "Threshold must be in the interval (0, 1].");
        }

        for (var k = 0; k < CumulativeExplainedVariance.Length; k++)
        {
            if (CumulativeExplainedVariance[k] >= threshold)
            {
                return k + 1;
            }
        }

        return CumulativeExplainedVariance.Length;
    }

    /// <summary>
    /// Projects observations onto the top <paramref name="numComponents"/> principal components.
    /// </summary>
    /// <param name="data">
    /// Observations in rows-by-variables layout. Rows are observations, columns are
    /// variables, in the same order as the variables that produced this PCA result.
    /// When <see cref="Mean"/> is populated, each row is centered by subtracting
    /// <see cref="Mean"/> before projection; when <see cref="Mean"/> is empty, the caller
    /// must pass pre-centered data.
    /// </param>
    /// <param name="numComponents">
    /// Number of principal components <c>k</c> onto which to project. Must be in <c>[1, N]</c>
    /// where <c>N</c> is the variable count.
    /// </param>
    /// <returns>
    /// Projected scores with shape <c>T × k</c>, where row <c>t</c> contains the scores of
    /// observation <paramref name="data"/>'s row <c>t</c> on the first <paramref name="numComponents"/>
    /// components.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="data"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Column count of <paramref name="data"/> does not match the PCA variable count.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="numComponents"/> is outside <c>[1, N]</c>.</exception>
    public T[,] Project(T[,] data, int numComponents)
    {
        ArgumentNullException.ThrowIfNull(data);

        var variables = Eigenvectors.GetLength(0);
        if (data.GetLength(1) != variables)
        {
            throw new ArgumentException(
                $"Data has {data.GetLength(1)} columns but PCA was fit on {variables} variables.",
                nameof(data));
        }

        if (numComponents < 1 || numComponents > variables)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numComponents),
                numComponents,
                $"Number of components must be in [1, {variables}].");
        }

        var rows = data.GetLength(0);
        var scores = new T[rows, numComponents];

        // Center (when Mean is populated) and project.
        var centering = Mean.Length == variables;

        for (var t = 0; t < rows; t++)
        {
            for (var k = 0; k < numComponents; k++)
            {
                var score = T.Zero;
                for (var j = 0; j < variables; j++)
                {
                    var value = centering ? data[t, j] - Mean[j] : data[t, j];
                    score += value * Eigenvectors[j, k];
                }

                scores[t, k] = score;
            }
        }

        return scores;
    }
}

/// <summary>
/// Legacy facade for <see cref="PcaResult{T}"/> at <c>T = decimal</c>.
/// Preserves source compatibility for existing callers.
/// </summary>
public sealed record PcaResult
{
    /// <summary>Eigenvalues of the covariance matrix, sorted in descending order.</summary>
    public required decimal[] Eigenvalues { get; init; }

    /// <summary>Eigenvector matrix — column <c>k</c> is the loading vector for principal component <c>k</c>.</summary>
    public required decimal[,] Eigenvectors { get; init; }

    /// <summary>Per-component fraction of total variance, <c>λₖ / Σⱼ λⱼ</c>.</summary>
    public required decimal[] ExplainedVarianceRatio { get; init; }

    /// <summary>
    /// Cumulative sum of <see cref="ExplainedVarianceRatio"/>. The last element equals 1
    /// (up to rounding) for any full-rank covariance matrix.
    /// </summary>
    public required decimal[] CumulativeExplainedVariance { get; init; }

    /// <summary>
    /// Per-variable mean used to center observations before projection. Empty when the
    /// result came from <c>Decompose(covarianceMatrix)</c> (no observation data available).
    /// </summary>
    public required decimal[] Mean { get; init; }

    /// <summary>
    /// Smallest <c>k</c> such that the cumulative explained variance reaches <paramref name="threshold"/>.
    /// </summary>
    /// <param name="threshold">Target cumulative variance, in <c>(0, 1]</c>.</param>
    /// <returns>
    /// <c>k ∈ [1, N]</c> where <c>N = Eigenvalues.Length</c>. Always at least 1 for a
    /// non-empty result.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is not in <c>(0, 1]</c>.</exception>
    public int NumComponentsForExplainedVariance(decimal threshold)
    {
        if (threshold <= 0m || threshold > 1m)
        {
            throw new ArgumentOutOfRangeException(
                nameof(threshold),
                threshold,
                "Threshold must be in the interval (0, 1].");
        }

        for (var k = 0; k < CumulativeExplainedVariance.Length; k++)
        {
            if (CumulativeExplainedVariance[k] >= threshold)
            {
                return k + 1;
            }
        }

        return CumulativeExplainedVariance.Length;
    }

    /// <summary>
    /// Projects observations onto the top <paramref name="numComponents"/> principal components.
    /// </summary>
    /// <param name="data">
    /// Observations in rows-by-variables layout. Rows are observations, columns are
    /// variables, in the same order as the variables that produced this PCA result.
    /// When <see cref="Mean"/> is populated, each row is centered by subtracting
    /// <see cref="Mean"/> before projection; when <see cref="Mean"/> is empty, the caller
    /// must pass pre-centered data.
    /// </param>
    /// <param name="numComponents">
    /// Number of principal components <c>k</c> onto which to project. Must be in <c>[1, N]</c>
    /// where <c>N</c> is the variable count.
    /// </param>
    /// <returns>
    /// Projected scores with shape <c>T × k</c>, where row <c>t</c> contains the scores of
    /// observation <paramref name="data"/>'s row <c>t</c> on the first <paramref name="numComponents"/>
    /// components.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="data"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Column count of <paramref name="data"/> does not match the PCA variable count.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="numComponents"/> is outside <c>[1, N]</c>.</exception>
    public decimal[,] Project(decimal[,] data, int numComponents)
    {
        ArgumentNullException.ThrowIfNull(data);

        var variables = Eigenvectors.GetLength(0);
        if (data.GetLength(1) != variables)
        {
            throw new ArgumentException(
                $"Data has {data.GetLength(1)} columns but PCA was fit on {variables} variables.",
                nameof(data));
        }

        if (numComponents < 1 || numComponents > variables)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numComponents),
                numComponents,
                $"Number of components must be in [1, {variables}].");
        }

        var rows = data.GetLength(0);
        var scores = new decimal[rows, numComponents];

        // Center (when Mean is populated) and project.
        var centering = Mean.Length == variables;

        for (var t = 0; t < rows; t++)
        {
            for (var k = 0; k < numComponents; k++)
            {
                var score = 0m;
                for (var j = 0; j < variables; j++)
                {
                    var value = centering ? data[t, j] - Mean[j] : data[t, j];
                    score += value * Eigenvectors[j, k];
                }

                scores[t, k] = score;
            }
        }

        return scores;
    }
}
