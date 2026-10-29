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
/// Generic random-matrix-theory denoised covariance estimator. Identifies noise
/// eigenvalues of the sample correlation matrix via the Marcenko-Pastur upper
/// bound and replaces them with their mean.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="Internal.NumericPrecision{T}.Sqrt"/> via <see cref="CovarianceHelpers{T}"/>.
/// </para>
/// <para>
/// Reference: Lopez de Prado, M. (2020). <em>Machine Learning for Asset
/// Managers</em>, §2.5 (Snippet 2.5).
/// </para>
/// <para>
/// Pipeline: sample covariance -> correlation -> Jacobi eigendecomposition ->
/// Marcenko-Pastur split -> average the noise eigenvalues -> reconstruct
/// correlation -> rescale to a unit diagonal (<c>C_ij / sqrt(C_ii . C_jj)</c>, clipped
/// to [-1, 1]) -> covariance. The Marcenko-Pastur edge is <c>sigma²(1 + 1/sqrt(q))²</c>,
/// <c>q = T / N</c>. The noise variance is fitted to the eigenvalue density using
/// the 1000-point MP / Gaussian-KDE squared error with bandwidth 0.01, over
/// [1e-5, 1-1e-5]. A deterministic grid scan and golden-section refinement of
/// candidate minima select the lowest objective; failed non-finite fits use 1.
/// This model-selection fit runs in double and can select a different minimum
/// than scipy's local L-BFGS-B search from 0.5 on a multimodal objective.
/// At T = decimal the selected structure has double resolution; eigenvalue
/// averaging, reconstruction and all returned values keep T precision. Denoising is ill-defined for fewer than 3
/// assets: for N &lt; 3 the estimator degenerates to the sample covariance
/// when shrinkage is disabled, or to the Ledoit-Wolf shrinkage estimate when
/// shrinkage is enabled.
/// </para>
/// <para>
/// When every eigenvalue lies at or below the Marčenko-Pastur edge (the all-noise
/// case), the complete spectrum is replaced by its mean before reconstruction.
/// </para>
/// <para>Half inputs widen exactly to double for estimation; only the final covariance is rounded back.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class DenoisedCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();
    private static readonly T s_shrinkageIntensity = T.CreateChecked(0.1);

    private readonly bool _applyLedoitWolfShrinkage;

    /// <summary>
    /// Initializes a new instance of the <see cref="DenoisedCovarianceEstimator{T}"/> type.
    /// </summary>
    /// <param name="applyLedoitWolfShrinkage">
    /// When <see langword="true"/>, applies a light Ledoit-Wolf-style shrinkage
    /// (delta = 0.1) to the denoised covariance. Default <see langword="false"/>.
    /// </param>
    public DenoisedCovarianceEstimator(bool applyLedoitWolfShrinkage = false)
    {
        _applyLedoitWolfShrinkage = applyLedoitWolfShrinkage;
    }

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            // Avoid Half count overflow and loss of convergence in the spectral step.
            return CovarianceHelpers<T>.EstimateHalfShrinkage(
                returns, new DenoisedCovarianceEstimator<double>(_applyLedoitWolfShrinkage));
        }

        var n = returns.GetLength(1);

        // Denoising is ill-defined for fewer than 3 assets.
        if (n < 3)
        {
            return _applyLedoitWolfShrinkage
                ? new LedoitWolfShrinkageEstimator<T>().Estimate(returns)
                : s_sampleEstimator.Estimate(returns);
        }

        var (cleanedCorr, stdDevs, _) = DenoiseReturns(returns);

        var result = CovarianceHelpers<T>.CorrelationToCovariance(cleanedCorr, stdDevs);

        if (_applyLedoitWolfShrinkage)
        {
            result = ApplyShrinkageToCovariance(result);
        }

        return result;
    }

    /// <summary>
    /// Shared Marcenko-Pastur denoising kernel (snippet 2.5): converts the sample
    /// covariance to a correlation matrix, replaces the eigenvalues at or below the
    /// Marcenko-Pastur edge with their mean, including when the whole spectrum is noise,
    /// reconstructs, and rescales to a unit diagonal.
    /// </summary>
    /// <param name="sampleCov">Sample covariance of <paramref name="t"/> observations (N &gt;= 3 assets).</param>
    /// <param name="t">Observation count.</param>
    /// <returns>
    /// The cleaned correlation matrix, the per-asset standard deviations, and the number
    /// of signal eigenvalues (above the edge). A signal count of 0 is the all-noise case,
    /// in which the complete spectrum is averaged.
    /// </returns>
    internal static (T[,] Correlation, T[] StdDevs, int SignalCount) DenoiseCorrelation(T[,] sampleCov, int t)
    {
        var (corr, stdDevs) = CovarianceHelpers<T>.CovarianceToCorrelation(sampleCov);
        return DenoiseCorrelation(corr, stdDevs, t);
    }

    /// <summary>
    /// Denoises the sample correlation of a returns panel (N &gt;= 3 assets). The correlation and
    /// standard deviations come from <see cref="CovarianceHelpers{T}.SampleCorrelation"/>, which
    /// keeps a column whose variance leaves the type's range from losing its correlations.
    /// </summary>
    /// <param name="returns">Returns panel, T observations by N assets.</param>
    /// <returns>As <see cref="DenoiseCorrelation(T[,], int)"/>.</returns>
    internal static (T[,] Correlation, T[] StdDevs, int SignalCount) DenoiseReturns(T[,] returns)
    {
        var (corr, stdDevs) = CovarianceHelpers<T>.SampleCorrelation(returns);
        return DenoiseCorrelation(corr, stdDevs, returns.GetLength(0));
    }

    private static (T[,] Correlation, T[] StdDevs, int SignalCount) DenoiseCorrelation(T[,] corr, T[] stdDevs, int t)
    {
        var n = corr.GetLength(0);
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);
        var eigenvalues = eigen.Values;
        var eigenvectors = eigen.Vectors;

        var q = T.CreateChecked(t) / T.CreateChecked(n);
        var noiseVariance = CovarianceHelpers<T>.FitNoiseVariance(eigenvalues, (double)t / n);
        var mpUpperBound = T.CreateChecked(noiseVariance) * CovarianceHelpers<T>.MarcenkoPasturUpperBound(q);

        // Split into signal / noise and replace noise eigenvalues with their mean.
        var noiseCount = 0;
        var noiseSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            if (eigenvalues[i] <= mpUpperBound)
            {
                noiseCount++;
                noiseSum += eigenvalues[i];
            }
        }

        if (noiseCount > 0)
        {
            var noiseAvg = noiseSum / T.CreateChecked(noiseCount);
            for (var i = 0; i < n; i++)
            {
                if (eigenvalues[i] <= mpUpperBound)
                {
                    eigenvalues[i] = noiseAvg;
                }
            }
        }

        var cleanedCorr = RescaleToUnitDiagonal(
            CovarianceHelpers<T>.ReconstructFromEigen(eigenvalues, eigenvectors));

        return (cleanedCorr, stdDevs, n - noiseCount);
    }

    /// <summary>
    /// Rescales a reconstructed symmetric matrix to a unit diagonal, the <c>cov2corr</c>
    /// step of López de Prado's snippets 2.5 and 2.6: <c>C_ij / sqrt(C_ii . C_jj)</c>,
    /// clipped to [-1, 1].
    /// </summary>
    /// <remarks>
    /// Eigenvalue cleaning moves the reconstructed diagonal away from 1 structurally
    /// (averaging noise eigenvalues redistributes variance across assets), so the
    /// matrix is rescaled rather than having its diagonal overwritten. Rescaling is a
    /// congruence by a positive diagonal matrix, so a PSD reconstruction stays PSD.
    /// A row whose diagonal is not positive carries no residual variance; it is written
    /// as <c>1</c> on the diagonal and <c>0</c> elsewhere, the same convention
    /// <see cref="CovarianceHelpers{T}.CovarianceToCorrelation"/> applies to a
    /// zero-variance asset. The diagonal is stored as exactly <c>1</c>.
    /// </remarks>
    internal static T[,] RescaleToUnitDiagonal(T[,] matrix)
    {
        var n = matrix.GetLength(0);
        var scale = new T[n];
        for (var i = 0; i < n; i++)
        {
            scale[i] = matrix[i, i] > T.Zero ? Internal.NumericPrecision<T>.Sqrt(matrix[i, i]) : T.Zero;
        }

        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            result[i, i] = T.One;
            for (var j = i + 1; j < n; j++)
            {
                var value = scale[i] == T.Zero || scale[j] == T.Zero
                    ? T.Zero
                    : T.Clamp(matrix[i, j] / (scale[i] * scale[j]), -T.One, T.One);
                result[i, j] = value;
                result[j, i] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Applies a light shrinkage pass toward the scaled identity.
    /// </summary>
    private static T[,] ApplyShrinkageToCovariance(T[,] cov)
    {
        var n = cov.GetLength(0);

        var mu = T.Zero;
        var count = T.CreateChecked(n);
        for (var i = 0; i < n; i++)
        {
            // The mean can fit even when the trace does not.
            mu += cov[i, i] / count;
        }

        var result = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var target = i == j ? mu : T.Zero;
                result[i, j] = s_shrinkageIntensity * target + (T.One - s_shrinkageIntensity) * cov[i, j];
            }
        }

        return result;
    }
}

/// <summary>
/// Random-matrix-theory denoised covariance estimator. Identifies noise
/// eigenvalues of the sample correlation matrix via the Marcenko-Pastur upper
/// bound and replaces them with their mean. Preserves the trace of the
/// correlation matrix while removing estimation noise.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Lopez de Prado, M. (2020). <em>Machine Learning for Asset
/// Managers</em>, §2.5 (Snippet 2.5).
/// </para>
/// </remarks>
[Obsolete(
    "Use DenoisedCovarianceEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class DenoisedCovarianceEstimator : ICovarianceEstimator
{
    private readonly DenoisedCovarianceEstimator<decimal> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="DenoisedCovarianceEstimator"/> type.
    /// </summary>
    /// <param name="applyLedoitWolfShrinkage">
    /// When <see langword="true"/>, applies a light Ledoit-Wolf-style shrinkage
    /// (delta = 0.1) to the denoised covariance. Default <see langword="false"/>.
    /// </param>
    public DenoisedCovarianceEstimator(bool applyLedoitWolfShrinkage = false)
    {
        _inner = new DenoisedCovarianceEstimator<decimal>(applyLedoitWolfShrinkage);
    }

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
