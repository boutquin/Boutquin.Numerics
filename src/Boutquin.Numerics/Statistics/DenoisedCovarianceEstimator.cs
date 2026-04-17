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
/// Reference: Lopez de Prado, M. (2018). <em>Advances in Financial Machine
/// Learning</em>, Chapter 2.
/// </para>
/// <para>
/// Pipeline: sample covariance -> correlation -> Jacobi eigendecomposition ->
/// Marcenko-Pastur split -> average the noise eigenvalues -> reconstruct
/// correlation -> covariance. For N &lt; 3 the estimator degenerates to the
/// sample covariance.
/// </para>
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

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        var sampleCov = s_sampleEstimator.Estimate(returns);

        // Denoising is ill-defined for fewer than 3 assets.
        if (n < 3)
        {
            return _applyLedoitWolfShrinkage
                ? new LedoitWolfShrinkageEstimator<T>().Estimate(returns)
                : sampleCov;
        }

        var (corr, stdDevs) = CovarianceHelpers<T>.CovarianceToCorrelation(sampleCov);
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);
        var eigenvalues = (T[])eigen.Values.Clone();

        var q = T.CreateChecked(t) / T.CreateChecked(n);
        var mpUpperBound = CovarianceHelpers<T>.MarcenkoPasturUpperBound(q);

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

        if (noiseCount > 0 && noiseCount < n)
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

        var cleanedCorr = CovarianceHelpers<T>.ReconstructFromEigen(eigenvalues, eigen.Vectors);

        // Force unit diagonal (removes numerical drift from reconstruction).
        for (var i = 0; i < n; i++)
        {
            cleanedCorr[i, i] = T.One;
        }

        var result = CovarianceHelpers<T>.CorrelationToCovariance(cleanedCorr, stdDevs);

        if (_applyLedoitWolfShrinkage)
        {
            result = ApplyShrinkageToCovariance(result);
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
        for (var i = 0; i < n; i++)
        {
            mu += cov[i, i];
        }

        mu /= T.CreateChecked(n);

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
/// Reference: Lopez de Prado, M. (2018). <em>Advances in Financial Machine
/// Learning</em>, Chapter 2.
/// </para>
/// </remarks>
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
