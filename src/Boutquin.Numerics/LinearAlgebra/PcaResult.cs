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
    private T[] _eigenvalues = null!;
    private T[,] _eigenvectors = null!;
    private T[] _explainedVarianceRatio = null!;
    private T[] _cumulativeExplainedVariance = null!;
    private T[] _mean = null!;

    /// <summary>Eigenvalues of the covariance matrix, sorted in descending order.</summary>
    public required T[] Eigenvalues
    {
        get => (T[])_eigenvalues.Clone();
        init => _eigenvalues = (T[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Eigenvector matrix — column <c>k</c> is the loading vector for principal component <c>k</c>.</summary>
    public required T[,] Eigenvectors
    {
        get => (T[,])_eigenvectors.Clone();
        init => _eigenvectors = (T[,])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Per-component fraction of total variance, <c>λₖ / Σⱼ λⱼ</c>.</summary>
    public required T[] ExplainedVarianceRatio
    {
        get => (T[])_explainedVarianceRatio.Clone();
        init => _explainedVarianceRatio = (T[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Cumulative sum of <see cref="ExplainedVarianceRatio"/>. The last element equals 1
    /// (up to rounding) for any full-rank covariance matrix.
    /// </summary>
    public required T[] CumulativeExplainedVariance
    {
        get => (T[])_cumulativeExplainedVariance.Clone();
        init => _cumulativeExplainedVariance = (T[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Per-variable mean used to center observations before projection. Empty when the
    /// result came from <c>Decompose(covarianceMatrix)</c> (no observation data available).
    /// </summary>
    public required T[] Mean
    {
        get => (T[])_mean.Clone();
        init => _mean = (T[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Smallest <c>k</c> such that the cumulative explained variance reaches <paramref name="threshold"/>.
    /// </summary>
    /// <param name="threshold">Target cumulative variance, in <c>(0, 1]</c>.</param>
    /// <returns>
    /// <c>k ∈ [1, N]</c> where <c>N = Eigenvectors.GetLength(1)</c>. Always at least 1 for a
    /// non-empty result.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is not in <c>(0, 1]</c>.</exception>
    public int NumComponentsForExplainedVariance(double threshold)
    {
        if (!double.IsFinite(threshold) || threshold <= 0.0 || threshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(threshold),
                threshold,
                "Threshold must be in the interval (0, 1].");
        }

        ValidateShape();
        for (var k = 0; k < _cumulativeExplainedVariance.Length; k++)
        {
            if (ReachesThreshold(_cumulativeExplainedVariance[k], threshold))
            {
                return k + 1;
            }
        }

        return _cumulativeExplainedVariance.Length;
    }

    internal static bool ReachesThreshold(T cumulative, double threshold)
    {
        if (typeof(T) != typeof(decimal))
        {
            return double.CreateChecked(cumulative) >= threshold;
        }

        // Compare the exact decimal and binary rationals. Decimal.CreateChecked(double)
        // keeps only 15 significant digits and can move either side of this boundary.
        var parts = decimal.GetBits(decimal.CreateChecked(cumulative));
        var numerator = (BigInteger)(uint)parts[0]
            + ((BigInteger)(uint)parts[1] << 32) + ((BigInteger)(uint)parts[2] << 64);
        if (parts[3] < 0)
        {
            numerator = -numerator;
        }

        var bits = BitConverter.DoubleToInt64Bits(threshold);
        var exponent = (int)((bits >> 52) & 0x7ff);
        var significand = (BigInteger)(bits & 0xfffffffffffffL);
        significand = exponent == 0 ? significand : significand + (BigInteger.One << 52);
        var shift = exponent == 0 ? -1074 : exponent - 1075;
        var decimalDenominator = BigInteger.Pow(10, (parts[3] >> 16) & 0xff);
        return shift < 0
            ? (numerator << -shift) >= significand * decimalDenominator
            : numerator >= (significand << shift) * decimalDenominator;
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
    /// where <c>N</c> is the eigenvector column count.
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

        ValidateShape();
        var variables = _eigenvectors.GetLength(0);
        if (data.GetLength(1) != variables)
        {
            throw new ArgumentException(
                $"Data has {data.GetLength(1)} columns but PCA was fit on {variables} variables.",
                nameof(data));
        }

        var availableComponents = _eigenvectors.GetLength(1);
        if (numComponents < 1 || numComponents > availableComponents)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numComponents),
                numComponents,
                $"Number of components must be in [1, {availableComponents}].");
        }

        var rows = data.GetLength(0);
        var scores = new T[rows, numComponents];

        for (var t = 0; t < rows; t++)
        {
            for (var k = 0; k < numComponents; k++)
            {
                scores[t, k] = PcaProjection<T>.Score(data, _mean, _eigenvectors, t, k);
            }
        }

        return scores;
    }

    private void ValidateShape() => PcaResultShape.Validate(
        _eigenvalues.Length,
        _eigenvectors.GetLength(0),
        _eigenvectors.GetLength(1),
        _explainedVarianceRatio.Length,
        _cumulativeExplainedVariance.Length,
        _mean.Length);
}

/// <summary>
/// Legacy facade for <see cref="PcaResult{T}"/> at <c>T = decimal</c>.
/// Preserves source compatibility for existing callers.
/// </summary>
public sealed record PcaResult
{
    private decimal[] _eigenvalues = null!;
    private decimal[,] _eigenvectors = null!;
    private decimal[] _explainedVarianceRatio = null!;
    private decimal[] _cumulativeExplainedVariance = null!;
    private decimal[] _mean = null!;
    private decimal[] _standardDeviations = [];
    private decimal[] _fittedOrigins = [];
    private decimal[] _fittedShiftedMean = [];

    /// <summary>
    /// Fitted sample standard deviations, one per variable for a standardized fit; empty otherwise.
    /// Projection divides centered observations by these scales. Values must be positive.
    /// Both initialization and access copy the array, so callers may retain one accessor result
    /// when rebuilding a result or reversing a standardized projection.
    /// </summary>
    public decimal[] StandardDeviations
    {
        get => (decimal[])_standardDeviations.Clone();
        init => _standardDeviations = (decimal[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Per-column origin of the fitted centering: the observed value the fit measured deviations
    /// from. The fitted absolute mean is the exact sum <c>origin + shifted mean</c>.
    /// </summary>
    internal decimal[] FittedOrigins
    {
        init => _fittedOrigins = (decimal[])value.Clone();
    }

    /// <summary>
    /// Per-column mean of the fitted deviations from <see cref="FittedOrigins"/>. Unlike the
    /// absolute mean it keeps the fractional part that decimal's 28-29 significant digits drop
    /// once the origin is large (for example 1e28 + 0.5).
    /// </summary>
    internal decimal[] FittedShiftedMean
    {
        init => _fittedShiftedMean = (decimal[])value.Clone();
    }

    internal static decimal StandardizedDeviation(decimal value, decimal mean, decimal standardDeviation)
    {
        decimal deviation;
        try
        {
            deviation = value - mean;
        }
        catch (OverflowException)
        {
            // Opposite-sign values may have an unrepresentable difference even
            // when their standardized difference fits in the working type.
            return value / standardDeviation - mean / standardDeviation;
        }

        return deviation / standardDeviation;
    }

    internal static decimal StandardizedDeviation(
        decimal value, decimal origin, decimal shiftedMean, decimal standardDeviation)
    {
        decimal deviation;
        try
        {
            deviation = (value - origin) - shiftedMean;
        }
        catch (OverflowException)
        {
            return ((value / standardDeviation) - (origin / standardDeviation)) - (shiftedMean / standardDeviation);
        }

        return deviation / standardDeviation;
    }

    /// <summary>
    /// Whether the exact fitted (origin, shifted mean) pair still describes <see cref="Mean"/>.
    /// The pair is the centering the fit used; <see cref="Mean"/> is its sum rounded to decimal.
    /// A caller who replaces <see cref="Mean"/> (a <see langword="with"/> expression or a new
    /// initializer) has chosen a different centering, and then the supplied <see cref="Mean"/>
    /// is the only authority.
    /// </summary>
    private bool UsesFittedCentering()
    {
        var n = _mean.Length;
        if (n == 0 || _fittedOrigins.Length != n || _fittedShiftedMean.Length != n)
        {
            return false;
        }

        for (var j = 0; j < n; j++)
        {
            try
            {
                if (_fittedOrigins[j] + _fittedShiftedMean[j] != _mean[j])
                {
                    return false;
                }
            }
            catch (OverflowException)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Eigenvalues of the covariance matrix, sorted in descending order.</summary>
    public required decimal[] Eigenvalues
    {
        get => (decimal[])_eigenvalues.Clone();
        init => _eigenvalues = (decimal[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Eigenvector matrix — column <c>k</c> is the loading vector for principal component <c>k</c>.</summary>
    public required decimal[,] Eigenvectors
    {
        get => (decimal[,])_eigenvectors.Clone();
        init => _eigenvectors = (decimal[,])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>Per-component fraction of total variance, <c>λₖ / Σⱼ λⱼ</c>.</summary>
    public required decimal[] ExplainedVarianceRatio
    {
        get => (decimal[])_explainedVarianceRatio.Clone();
        init => _explainedVarianceRatio = (decimal[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Cumulative sum of <see cref="ExplainedVarianceRatio"/>. The last element equals 1
    /// (up to rounding) for any full-rank covariance matrix.
    /// </summary>
    public required decimal[] CumulativeExplainedVariance
    {
        get => (decimal[])_cumulativeExplainedVariance.Clone();
        init => _cumulativeExplainedVariance = (decimal[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Per-variable mean used to center observations before projection. Empty when the
    /// result came from <c>Decompose(covarianceMatrix)</c> (no observation data available).
    /// </summary>
    /// <remarks>
    /// This is the absolute mean rounded to the nearest <see cref="decimal"/>. When the columns sit
    /// at an offset where decimal cannot hold the fractional part of the mean (rows near 1e28 whose
    /// mean is a half-integer), the rounded value differs from the true mean; a result produced by
    /// <c>FromReturns</c> keeps the exact fitted centering internally and <see cref="Project"/> uses
    /// it, so the training rows still project to centered scores.
    /// </remarks>
    public required decimal[] Mean
    {
        get => (decimal[])_mean.Clone();
        init => _mean = (decimal[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <summary>
    /// Smallest <c>k</c> such that the cumulative explained variance reaches <paramref name="threshold"/>.
    /// </summary>
    /// <param name="threshold">Target cumulative variance, in <c>(0, 1]</c>.</param>
    /// <returns>
    /// <c>k ∈ [1, N]</c> where <c>N = Eigenvectors.GetLength(1)</c>. Always at least 1 for a
    /// non-empty result.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is not in <c>(0, 1]</c>.</exception>
    public int NumComponentsForExplainedVariance(double threshold)
    {
        if (!double.IsFinite(threshold) || threshold <= 0.0 || threshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(threshold),
                threshold,
                "Threshold must be in the interval (0, 1].");
        }

        ValidateShape();
        for (var k = 0; k < _cumulativeExplainedVariance.Length; k++)
        {
            if (PcaResult<decimal>.ReachesThreshold(_cumulativeExplainedVariance[k], threshold))
            {
                return k + 1;
            }
        }

        return _cumulativeExplainedVariance.Length;
    }

    /// <summary>
    /// Projects observations onto the top <paramref name="numComponents"/> principal components.
    /// </summary>
    /// <param name="data">
    /// Observations in rows-by-variables layout. Rows are observations, columns are
    /// variables, in the same order as the variables that produced this PCA result.
    /// When <see cref="Mean"/> is populated, each row is centered by subtracting
    /// <see cref="Mean"/> before projection (for a fit produced by <c>FromReturns</c> whose
    /// <see cref="Mean"/> is unchanged, by subtracting the exact fitted mean that
    /// <see cref="Mean"/> rounds); when <see cref="Mean"/> is empty, the caller
    /// must pass pre-centered data. For a standardized fit, centered values are also
    /// divided by the fitted sample standard deviations.
    /// </param>
    /// <param name="numComponents">
    /// Number of principal components <c>k</c> onto which to project. Must be in <c>[1, N]</c>
    /// where <c>N</c> is the eigenvector column count.
    /// </param>
    /// <returns>
    /// Projected scores with shape <c>T × k</c>, where row <c>t</c> contains the scores of
    /// observation <paramref name="data"/>'s row <c>t</c> on the first <paramref name="numComponents"/>
    /// components.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="data"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Column count of <paramref name="data"/> does not match the PCA variable count.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="numComponents"/> is outside <c>[1, N]</c>.</exception>
    /// <exception cref="InvalidOperationException">The result's array dimensions or fitted standard deviations are inconsistent.</exception>
    public decimal[,] Project(decimal[,] data, int numComponents)
    {
        ArgumentNullException.ThrowIfNull(data);

        ValidateShape();
        var variables = _eigenvectors.GetLength(0);
        if (data.GetLength(1) != variables)
        {
            throw new ArgumentException(
                $"Data has {data.GetLength(1)} columns but PCA was fit on {variables} variables.",
                nameof(data));
        }

        var availableComponents = _eigenvectors.GetLength(1);
        if (numComponents < 1 || numComponents > availableComponents)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numComponents),
                numComponents,
                $"Number of components must be in [1, {availableComponents}].");
        }

        var rows = data.GetLength(0);
        var scores = new decimal[rows, numComponents];
        var fitted = UsesFittedCentering();

        for (var t = 0; t < rows; t++)
        {
            for (var k = 0; k < numComponents; k++)
            {
                if (_standardDeviations.Length == 0)
                {
                    scores[t, k] = fitted
                        ? PcaProjection<decimal>.Score(data, _fittedOrigins, _eigenvectors, t, k, _fittedShiftedMean)
                        : PcaProjection<decimal>.Score(data, _mean, _eigenvectors, t, k);
                    continue;
                }

                var score = 0m;
                for (var j = 0; j < variables; j++)
                {
                    decimal value;
                    if (_standardDeviations.Length != 0)
                    {
                        value = fitted
                            ? StandardizedDeviation(
                                data[t, j], _fittedOrigins[j], _fittedShiftedMean[j], _standardDeviations[j])
                            : StandardizedDeviation(
                                data[t, j], _mean.Length != 0 ? _mean[j] : 0m, _standardDeviations[j]);
                    }
                    else
                    {
                        value = _mean.Length != 0 ? data[t, j] - _mean[j] : data[t, j];
                    }

                    score += value * _eigenvectors[j, k];
                }

                scores[t, k] = score;
            }
        }

        return scores;
    }

    private void ValidateShape()
    {
        PcaResultShape.Validate(
            _eigenvalues.Length,
            _eigenvectors.GetLength(0),
            _eigenvectors.GetLength(1),
            _explainedVarianceRatio.Length,
            _cumulativeExplainedVariance.Length,
            _mean.Length);
        if (_standardDeviations.Length != 0
            && (_standardDeviations.Length != _eigenvectors.GetLength(0)
                || Array.Exists(_standardDeviations, static value => value <= 0m)))
        {
            throw new InvalidOperationException("StandardDeviations must contain one positive scale per variable when populated.");
        }
    }
}

internal static class PcaProjection<T> where T : IFloatingPoint<T>
{
    /// <summary>
    /// Projects one observation onto one loading vector. <paramref name="mean"/> is the centering
    /// to subtract; when <paramref name="shiftedMean"/> is supplied, <paramref name="mean"/> is
    /// instead the origin and the observation is centered as <c>(x − origin) − shiftedMean</c>,
    /// which keeps a fractional mean that the absolute <c>origin + shiftedMean</c> cannot hold.
    /// </summary>
    internal static T Score(
        T[,] data, T[] mean, T[,] loadings, int row, int component, T[]? shiftedMean = null)
    {
        var variables = data.GetLength(1);
        try
        {
            var score = T.Zero;
            for (var j = 0; j < variables; j++)
            {
                var value = mean.Length != 0 ? data[row, j] - mean[j] : data[row, j];
                if (shiftedMean is not null)
                {
                    value -= shiftedMean[j];
                }

                score += value * loadings[j, component];
            }

            if (T.IsFinite(score))
            {
                return score;
            }
        }
        catch (OverflowException)
        {
            // Centering or a partial sum may overflow before the weighted cancellation.
        }

        var scale = T.Zero;
        for (var j = 0; j < variables; j++)
        {
            scale = T.Max(scale, T.Abs(data[row, j]));
            if (mean.Length != 0)
            {
                scale = T.Max(scale, T.Abs(mean[j]));
            }

            if (shiftedMean is not null)
            {
                scale = T.Max(scale, T.Abs(shiftedMean[j]));
            }
        }

        var normalized = T.Zero;
        for (var j = 0; j < variables; j++)
        {
            var value = data[row, j] / scale;
            if (mean.Length != 0)
            {
                value -= mean[j] / scale;
            }

            if (shiftedMean is not null)
            {
                value -= shiftedMean[j] / scale;
            }

            normalized += value * loadings[j, component];
        }

        return normalized * scale;
    }
}

internal static class PcaResultShape
{
    internal static void Validate(
        int eigenvalueCount,
        int eigenvectorRows,
        int eigenvectorColumns,
        int explainedCount,
        int cumulativeCount,
        int meanCount)
    {
        if (eigenvalueCount != eigenvectorColumns)
        {
            throw new InvalidOperationException("Eigenvalues length must match Eigenvectors column count.");
        }

        if (explainedCount != eigenvectorColumns)
        {
            throw new InvalidOperationException("ExplainedVarianceRatio length must match Eigenvectors column count.");
        }

        if (cumulativeCount != eigenvectorColumns)
        {
            throw new InvalidOperationException("CumulativeExplainedVariance length must match Eigenvectors column count.");
        }

        if (meanCount != 0 && meanCount != eigenvectorRows)
        {
            throw new InvalidOperationException("Mean length must match Eigenvectors row count when populated.");
        }
    }
}
