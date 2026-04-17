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
/// Generic Tracy-Widom denoised covariance estimator. Uses the Tracy-Widom
/// distribution (Johnstone 2001) to set a sharper signal/noise threshold
/// than the asymptotic Marcenko-Pastur upper bound, accounting for
/// finite-sample fluctuations of the largest noise eigenvalue.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="Internal.NumericPrecision{T}.Sqrt"/> via <see cref="CovarianceHelpers{T}"/>.
/// </para>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Johnstone, I. M. (2001). "On the Distribution of the Largest Eigenvalue in Principal Components Analysis."</description></item>
/// <item><description>Bun, J., Bouchaud, J.-P. &amp; Potters, M. (2017). "Cleaning Large Correlation Matrices: Tools from Random Matrix Theory."</description></item>
/// </list>
/// </para>
/// <para>
/// Threshold: <c>lambda_TW = (1 + 1/sqrt(q))^2 + N^(-2/3) . c_alpha . mu_TW</c>
/// where the finite-N correction shifts the MP edge by approximately the standard
/// deviation of the Tracy-Widom-1 distribution scaled by the appropriate rate.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class TracyWidomDenoisedCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private const double TracyWidomConstant = 2.02;
    private const double TracyWidomLocation = 1.21;

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        var sampleCov = new SampleCovarianceEstimator<T>().Estimate(returns);

        if (n < 3)
        {
            return sampleCov;
        }

        var (corr, stdDevs) = CovarianceHelpers<T>.CovarianceToCorrelation(sampleCov);
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);
        var eigenvalues = (T[])eigen.Values.Clone();

        // Tracy-Widom threshold computed in double for transcendental precision.
        var q = (double)t / n;
        var sqrtQ = Math.Sqrt(q);
        var mpEdge = (1.0 + 1.0 / sqrtQ) * (1.0 + 1.0 / sqrtQ);
        var twShift = Math.Pow(n, -2.0 / 3.0) * TracyWidomConstant * TracyWidomLocation;
        var threshold = T.CreateChecked(mpEdge + twShift);

        // Replace noise eigenvalues with their average.
        var noiseCount = 0;
        var noiseSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            if (eigenvalues[i] <= threshold)
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
                if (eigenvalues[i] <= threshold)
                {
                    eigenvalues[i] = noiseAvg;
                }
            }
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
/// Tracy-Widom denoised covariance estimator. Uses the Tracy-Widom
/// distribution (Johnstone 2001) to set a sharper signal/noise threshold
/// than the asymptotic Marcenko-Pastur upper bound, accounting for
/// finite-sample fluctuations of the largest noise eigenvalue.
/// </summary>
/// <remarks>
/// <para>
/// References:
/// <list type="bullet">
/// <item><description>Johnstone, I. M. (2001). "On the Distribution of the Largest Eigenvalue in Principal Components Analysis."</description></item>
/// <item><description>Bun, J., Bouchaud, J.-P. &amp; Potters, M. (2017). "Cleaning Large Correlation Matrices: Tools from Random Matrix Theory."</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class TracyWidomDenoisedCovarianceEstimator : ICovarianceEstimator
{
    private readonly TracyWidomDenoisedCovarianceEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
