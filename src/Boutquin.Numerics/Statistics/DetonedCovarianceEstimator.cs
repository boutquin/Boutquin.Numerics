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
/// Generic detoned covariance estimator. Extends Marcenko-Pastur denoising by
/// removing the market component — the first principal component of the denoised
/// correlation matrix — and rescaling the remainder to a unit diagonal.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="Internal.NumericPrecision{T}.Sqrt"/> via <see cref="CovarianceHelpers{T}"/>.
/// The denoising model-selection fit runs in double: for decimal, the selected
/// structure has double resolution while the returned covariance retains T precision.
/// </para>
/// <para>
/// Reference: Lopez de Prado, M. (2020). <em>Machine Learning for Asset
/// Managers</em>, §2.6.
/// </para>
/// <para>
/// Pipeline: denoise as <see cref="DenoisedCovarianceEstimator{T}"/> -> eigendecompose
/// the denoised correlation <c>C</c> -> subtract <c>alpha . lambda_1 . v_1 . v_1^T</c> ->
/// rescale to a unit diagonal (<c>C_ij / sqrt(C_ii . C_jj)</c>, clipped to [-1, 1]) ->
/// covariance. At <c>alpha = 1</c> this is the book's detoning, which removes the market
/// component entirely; smaller <c>alpha</c> removes that fraction of it, and
/// <c>alpha = 0</c> returns the denoised estimate.
/// </para>
/// <para>
/// <strong>PSD contract: PSD up to rounding; full removal can be singular.</strong> Removing
/// <c>alpha . lambda_1 . v_1 . v_1^T</c> leaves <c>v_1</c> with eigenvalue
/// <c>(1 - alpha) . lambda_1 &gt;= 0</c>, and the unit-diagonal rescale is a congruence by
/// a positive diagonal matrix, so no negative eigenvalue is introduced. When a market component is removed
/// at alpha = 1 and the residual diagonal is positive, the matrix is singular.
/// No singularity is guaranteed for fewer than three assets, an all-noise spectrum,
/// or zero residual rows that use the identity fallback. Check positive definiteness
/// before Cholesky factorization or inversion.
/// </para>
/// <para>
/// When every eigenvalue lies at or below the Marčenko-Pastur edge (the all-noise
/// case, which is common when T &lt; N) no eigenvalue qualifies as a market component:
/// the complete noise spectrum is averaged, but no market component is removed.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class DetonedCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();

    private readonly T _detoningAlpha;

    /// <summary>
    /// Initializes a new instance of the <see cref="DetonedCovarianceEstimator{T}"/> type
    /// with full detoning (<c>alpha = 1</c>).
    /// </summary>
    public DetonedCovarianceEstimator()
        : this(T.One)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DetonedCovarianceEstimator{T}"/> type.
    /// </summary>
    /// <param name="detoningAlpha">
    /// Fraction of the market component (the first principal component of the denoised
    /// correlation) that is removed. Must lie in [0, 1]. Use the parameterless constructor for the default of 1.0 (full detoning); for a
    /// value-type <typeparamref name="T"/>, an explicit <see langword="default"/> argument
    /// is 0 (no detoning).
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Alpha is outside [0, 1].</exception>
    public DetonedCovarianceEstimator(T? detoningAlpha = default)
    {
        detoningAlpha ??= T.One;

        if (!T.IsFinite(detoningAlpha) || detoningAlpha < T.Zero || detoningAlpha > T.One)
        {
            throw new ArgumentOutOfRangeException(
                nameof(detoningAlpha), detoningAlpha, "Detoning alpha must be in [0, 1].");
        }

        _detoningAlpha = detoningAlpha;
    }

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            return CovarianceHelpers<T>.EstimateHalfShrinkage(
                returns, new DetonedCovarianceEstimator<double>(double.CreateChecked(_detoningAlpha)));
        }

        var n = returns.GetLength(1);

        if (n < 3)
        {
            return s_sampleEstimator.Estimate(returns);
        }

        var (denoisedCorr, stdDevs, signalCount) =
            DenoisedCovarianceEstimator<T>.DenoiseReturns(returns);

        // All-noise spectrum: PC1 is not a market component, so nothing is removed.
        if (signalCount == 0 || _detoningAlpha == T.Zero)
        {
            return CovarianceHelpers<T>.CorrelationToCovariance(denoisedCorr, stdDevs);
        }

        // Detone (§2.6): subtract alpha . lambda_1 . v_1 . v_1^T, the market component
        // of the denoised correlation (eigenvalues sorted descending), then rescale.
        var eigen = JacobiEigenDecomposition<T>.Decompose(denoisedCorr);
        // Reconstruct the surviving PSD components directly. Subtracting two nearly
        // equal matrices destroys relative accuracy for collinear input; rescaling
        // that cancellation can turn roundoff into a material negative eigenvalue.
        var residualValues = eigen.Values;
        var unitRoundoff = T.CreateChecked(typeof(T) == typeof(decimal) ? 1e-28
            : typeof(T) == typeof(float) ? 1.1920928955078125e-7 : 2.2204460492503131e-16);
        var roundingBound = T.CreateChecked(n) * unitRoundoff * residualValues[0];
        residualValues[0] *= T.One - _detoningAlpha;
        var residualScale = T.Zero;
        for (var i = 0; i < n; i++)
        {
            residualValues[i] = residualValues[i] > (i == 0 ? T.Zero : roundingBound) ? residualValues[i] : T.Zero;
            residualScale = T.Max(residualScale, residualValues[i]);
        }

        // A common positive scale cancels in the subsequent correlation rescale.
        // Removing it before reconstruction prevents tiny decimal products from
        // rounding independently to an indefinite residual matrix.
        if (residualScale > T.Zero)
        {
            for (var i = 0; i < n; i++)
            {
                residualValues[i] /= residualScale;
            }
        }

        var detoned = CovarianceHelpers<T>.ReconstructFromEigen(residualValues, eigen.Vectors);

        var cleanedCorr = DenoisedCovarianceEstimator<T>.RescaleToUnitDiagonal(detoned);

        return CovarianceHelpers<T>.CorrelationToCovariance(cleanedCorr, stdDevs);
    }
}

/// <summary>
/// Detoned covariance estimator. Extends Marcenko-Pastur denoising by removing
/// the market component — the first principal component of the denoised
/// correlation matrix — and rescaling to a unit diagonal, so that optimizers
/// can see residual diversification structure past the dominant common factor.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Lopez de Prado, M. (2020). <em>Machine Learning for Asset
/// Managers</em>, §2.6.
/// </para>
/// </remarks>
[Obsolete(
    "Use DetonedCovarianceEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class DetonedCovarianceEstimator : ICovarianceEstimator
{
    private readonly DetonedCovarianceEstimator<decimal> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="DetonedCovarianceEstimator"/> type.
    /// </summary>
    /// <param name="detoningAlpha">
    /// Fraction of the market component (the first principal component of the denoised
    /// correlation) that is removed. Must lie in [0, 1]. Default 1.0 (full detoning).
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Alpha is outside [0, 1].</exception>
    public DetonedCovarianceEstimator(decimal detoningAlpha = 1m)
    {
        _inner = new DetonedCovarianceEstimator<decimal>(detoningAlpha);
    }

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
