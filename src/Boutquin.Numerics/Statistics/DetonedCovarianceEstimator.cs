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
/// Generic detoned covariance estimator. Extends Marcenko-Pastur denoising with
/// market-factor (PC1) shrinkage — the largest signal eigenvalue is pulled
/// toward the mean of the remaining signal eigenvalues.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="Internal.NumericPrecision{T}.Sqrt"/> via <see cref="CovarianceHelpers{T}"/>.
/// </para>
/// <para>
/// Reference: Lopez de Prado, M. (2020). <em>Machine Learning for Asset
/// Managers</em>, Chapter 2.
/// </para>
/// <para>
/// <strong>PSD contract: conditionally PSD.</strong> When the shrunk PC1
/// eigenvalue is dragged below some other signal eigenvalue, the reconstructed
/// matrix remains symmetric but the descending-order invariant is broken.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class DetonedCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();
    private static readonly DenoisedCovarianceEstimator<T> s_denoisedEstimator = new();

    private readonly T _detoningAlpha;

    /// <summary>
    /// Initializes a new instance of the <see cref="DetonedCovarianceEstimator{T}"/> type.
    /// </summary>
    /// <param name="detoningAlpha">
    /// Shrinkage intensity applied to PC1's eigenvalue. Must lie in [0, 1].
    /// Default 1.0 (full detoning).
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Alpha is outside [0, 1].</exception>
    public DetonedCovarianceEstimator(T? detoningAlpha = default)
    {
        detoningAlpha ??= T.One;

        if (detoningAlpha < T.Zero || detoningAlpha > T.One)
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

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        if (n < 3)
        {
            return s_sampleEstimator.Estimate(returns);
        }

        if (_detoningAlpha == T.Zero)
        {
            return s_denoisedEstimator.Estimate(returns);
        }

        var sampleCov = s_sampleEstimator.Estimate(returns);
        var (corr, stdDevs) = CovarianceHelpers<T>.CovarianceToCorrelation(sampleCov);
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);
        var eigenvalues = (T[])eigen.Values.Clone();

        var q = T.CreateChecked(t) / T.CreateChecked(n);
        var mpUpperBound = CovarianceHelpers<T>.MarcenkoPasturUpperBound(q);

        // Denoise: replace noise eigenvalues with their mean.
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

        var noiseAvg = noiseCount > 0 ? noiseSum / T.CreateChecked(noiseCount) : T.Zero;

        if (noiseCount > 0 && noiseCount < n)
        {
            for (var i = 0; i < n; i++)
            {
                if (eigenvalues[i] <= mpUpperBound)
                {
                    eigenvalues[i] = noiseAvg;
                }
            }
        }

        // Detone: shrink PC1 (index 0, eigenvalues are sorted descending)
        // toward the mean of the remaining signal eigenvalues.
        var signalCount = 0;
        var signalSum = T.Zero;
        for (var i = 1; i < n; i++)
        {
            if (eigenvalues[i] > mpUpperBound)
            {
                signalCount++;
                signalSum += eigenvalues[i];
            }
        }

        if (signalCount > 0)
        {
            var signalAvg = signalSum / T.CreateChecked(signalCount);
            eigenvalues[0] = (T.One - _detoningAlpha) * eigenvalues[0] + _detoningAlpha * signalAvg;
        }
        else if (noiseCount > 0)
        {
            // Only PC1 qualifies as signal — shrink toward the noise mean.
            eigenvalues[0] = (T.One - _detoningAlpha) * eigenvalues[0] + _detoningAlpha * noiseAvg;
        }

        var cleanedCorr = CovarianceHelpers<T>.ReconstructFromEigen(eigenvalues, eigen.Vectors);

        for (var i = 0; i < n; i++)
        {
            cleanedCorr[i, i] = T.One;
        }

        return CovarianceHelpers<T>.CorrelationToCovariance(cleanedCorr, stdDevs);
    }
}

/// <summary>
/// Detoned covariance estimator. Extends Marcenko-Pastur denoising with
/// market-factor (PC1) shrinkage — the largest signal eigenvalue is pulled
/// toward the mean of the remaining signal eigenvalues so that optimizers
/// can see residual diversification structure past the dominant common factor.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Lopez de Prado, M. (2020). <em>Machine Learning for Asset
/// Managers</em>, Chapter 2.
/// </para>
/// </remarks>
public sealed class DetonedCovarianceEstimator : ICovarianceEstimator
{
    private readonly DetonedCovarianceEstimator<decimal> _inner;

    /// <summary>
    /// Initializes a new instance of the <see cref="DetonedCovarianceEstimator"/> type.
    /// </summary>
    /// <param name="detoningAlpha">
    /// Shrinkage intensity applied to PC1's eigenvalue. Must lie in [0, 1].
    /// Default 1.0 (full detoning).
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Alpha is outside [0, 1].</exception>
    public DetonedCovarianceEstimator(decimal detoningAlpha = 1m)
    {
        _inner = new DetonedCovarianceEstimator<decimal>(detoningAlpha);
    }

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
