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
/// PC1 = largest eigenvalue. This matches Hull (2021, 11th Global Edition, §22.9)'s level/slope/curvature convention
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
        var eigenvalues = eigen.Values;
        var eigenvectors = eigen.Vectors;
        ApplySignConvention(eigenvectors);

        var explained = ComputeExplainedVariance(eigenvalues);

        return new PcaResult<T>
        {
            Eigenvalues = eigenvalues,
            Eigenvectors = eigenvectors,
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
        var scale = T.Zero;
        for (var i = 0; i < n; i++)
        {
            // Numerical noise in near-zero eigenvalues can go slightly negative; clamp to zero for the ratio.
            if (eigenvalues[i] > T.Zero)
            {
                scale = T.Max(scale, eigenvalues[i]);
            }
        }

        var ratio = new T[n];
        var cumulative = new T[n];

        if (scale == T.Zero)
        {
            return (ratio, cumulative);
        }

        var scaledTotal = T.Zero;
        var totalCompensation = T.Zero;
        for (var i = 0; i < n; i++)
        {
            if (eigenvalues[i] > T.Zero)
            {
                AddCompensated(ref scaledTotal, ref totalCompensation, eigenvalues[i] / scale);
            }
        }

        var running = T.Zero;
        var runningCompensation = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var positive = eigenvalues[i] > T.Zero ? eigenvalues[i] : T.Zero;
            ratio[i] = (positive / scale) / scaledTotal;
            // Sum the scaled spectrum, not the already-rounded ratios. Compensation
            // prevents narrow types from losing small contributions to a large sum.
            AddCompensated(ref running, ref runningCompensation, positive / scale);
            cumulative[i] = running / scaledTotal;
        }

        return (ratio, cumulative);
    }

    private static void AddCompensated(ref T sum, ref T compensation, T increment)
    {
        var adjusted = increment - compensation;
        var next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
    }
}

/// <summary>
/// Legacy facade delegating to <see cref="PrincipalComponentAnalysis{T}"/> instantiated
/// at <c>T = decimal</c>. Preserves source compatibility for existing callers.
/// Also retains the <see cref="FromReturns"/> entry point that accepts
/// <see cref="ReturnsMatrix"/> (a decimal-typed type not yet migrated to generic math).
/// </summary>
[Obsolete(
    "Use PrincipalComponentAnalysis<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
#pragma warning disable BNU0001
    public static PcaResult FromReturns(ReturnsMatrix returns, bool standardize = false)
#pragma warning restore BNU0001
    {
        var t = returns.Observations;
        var n = returns.Assets;

        if (t < 2)
        {
            throw new ArgumentException(
                "Returns matrix must contain at least two observations.", nameof(returns));
        }

        // Center in coordinates relative to an observed value. The absolute mean
        // can round away a fractional displacement at a large decimal offset.
        var origins = new decimal[n];
        var shifted = new decimal[t, n];
        for (var j = 0; j < n; j++)
        {
            origins[j] = returns[0, j];
            try
            {
                for (var i = 0; i < t; i++)
                {
                    shifted[i, j] = returns[i, j] - origins[j];
                }
            }
            catch (OverflowException)
            {
                // Opposite extremes have no representable shifted range. Retain
                // the existing scaled deviation path for that column.
                origins[j] = 0m;
                for (var i = 0; i < t; i++)
                {
                    shifted[i, j] = returns[i, j];
                }
            }
        }

        returns = shifted;

        // Per-column mean in shifted coordinates.
        var mean = new decimal[n];
        for (var j = 0; j < n; j++)
        {
            var sum = 0m;
            var scaled = false;
            for (var i = 0; i < t; i++)
            {
                try
                {
                    sum += returns[i, j];
                }
                catch (OverflowException)
                {
                    scaled = true;
                    break;
                }
            }

            if (scaled)
            {
                sum = 0m;
                for (var i = 0; i < t; i++)
                {
                    sum += returns[i, j] / t;
                }
            }

            mean[j] = scaled ? sum : sum / t;
        }

        // Per-column std (sample, divisor t − 1) — only needed when standardizing.
        decimal[]? std = null;
        decimal[]? scales = null;
        decimal[]? units = null;
        if (standardize)
        {
            std = new decimal[n];
            scales = new decimal[n];
            units = new decimal[n];
            for (var j = 0; j < n; j++)
            {
                // Sum the squares in units of the column's magnitude. A raw square rounds to
                // decimal's absolute 1e-28 grid before the standard deviation normalizes it
                // ((1.5e-14)^2 = 2.25e-28 becomes 2e-28, a 12.5% error in the scale), and a raw
                // square can also overflow. Every normalized square is at most four.
                var scale = ColumnMagnitude(returns, j);
                var ss = 0m;
                for (var i = 0; scale != 0m && i < t; i++)
                {
                    var dev = PcaResult.StandardizedDeviation(returns[i, j], mean[j], scale);
                    ss += dev * dev;
                }

                // The standard deviation in units of the column magnitude is of order one. The
                // product below is the stored standard deviation, which for a small column lies on
                // decimal's absolute grid (a 2e-14 scale keeps 14 digits); the fitted correlation is
                // standardized by the unit pair so its diagonal stays one to decimal precision.
                scales[j] = scale;
                units[j] = NumericPrecision<decimal>.Sqrt(ss / (t - 1));
                std[j] = units[j] * scale;

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
                decimal value;
                try
                {
                    var sum = 0m;
                    for (var i = 0; i < t; i++)
                    {
                        var dev1 = std is null ? returns[i, j1] - mean[j1]
                            : PcaResult.StandardizedDeviation(returns[i, j1], mean[j1], scales![j1]) / units![j1];
                        var dev2 = std is null ? returns[i, j2] - mean[j2]
                            : PcaResult.StandardizedDeviation(returns[i, j2], mean[j2], scales![j2]) / units![j2];

                        sum += dev1 * dev2;
                    }

                    value = sum / (t - 1);
                }
                catch (OverflowException) when (std is null)
                {
                    var scale1 = ColumnMagnitude(returns, j1);
                    var scale2 = ColumnMagnitude(returns, j2);
                    var sum = 0m;
                    for (var i = 0; i < t; i++)
                    {
                        var dev1 = returns[i, j1] / scale1 - mean[j1] / scale1;
                        var dev2 = returns[i, j2] / scale2 - mean[j2] / scale2;
                        sum += dev1 * dev2;
                    }

                    value = ((sum / (t - 1)) * scale1) * scale2;
                }

                covariance[j1, j2] = value;
                covariance[j2, j1] = value;
            }
        }

        var eigen = JacobiEigenDecomposition<decimal>.Decompose(covariance);
        var eigenvalues = eigen.Values;
        var eigenvectors = eigen.Vectors;
        PrincipalComponentAnalysis<decimal>.ApplySignConvention(eigenvectors);

        var explained = PrincipalComponentAnalysis<decimal>.ComputeExplainedVariance(eigenvalues);

        // The absolute mean is the nearest decimal to origin + shifted mean; at a large origin it can
        // round away the fractional part of the shifted mean (1e28 + 0.5 is not a decimal). The
        // exact pair is retained so projection centers with it instead of the rounded sum.
        var shiftedMean = (decimal[])mean.Clone();
        for (var j = 0; j < n; j++)
        {
            mean[j] += origins[j];
        }

        return new PcaResult
        {
            Eigenvalues = eigenvalues,
            Eigenvectors = eigenvectors,
            ExplainedVarianceRatio = explained.Ratio,
            CumulativeExplainedVariance = explained.Cumulative,
            Mean = mean,
            StandardDeviations = std ?? [],
            FittedOrigins = origins,
            FittedShiftedMean = shiftedMean,
        };
    }

#pragma warning disable BNU0001
    private static decimal ColumnMagnitude(ReturnsMatrix returns, int column)
#pragma warning restore BNU0001
    {
        var magnitude = 0m;
        for (var i = 0; i < returns.Observations; i++)
        {
            magnitude = Math.Max(magnitude, Math.Abs(returns[i, column]));
        }

        return magnitude;
    }
}
