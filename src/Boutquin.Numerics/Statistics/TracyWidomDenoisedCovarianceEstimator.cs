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
/// Threshold: <c>lambda_TW = mu_np / t + (sigma_np / t) . s_0.99</c>, Johnstone
/// (2001)'s centering and scaling of the largest null eigenvalue,
/// <c>mu_np = (sqrt(t-1) + sqrt(n))^2</c> (eq. 1.3) and
/// <c>sigma_np = (sqrt(t-1) + sqrt(n)) . (1/sqrt(t-1) + 1/sqrt(n))^(1/3)</c> (eq. 1.4)
/// (Johnstone's observation count is <c>t</c>, variable count is <c>n</c>),
/// divided by the observation count <c>t</c> to map his <c>n.lhat1</c>-scaled limit
/// (Theorem 1.1) onto the correlation-eigenvalue scale. <c>s_0.99 = F1^-1(0.99) = 2.0234</c>
/// is the Tracy-Widom (beta = 1) 99th percentile (2.02 in Johnstone's Table 1).
/// For t &lt; n, reverse t and n in mu and sigma (p. 300 after Theorem 1.1);
/// the covariance normalization divisor remains the observation count t. The
/// location of the null law is carried by <c>mu_np</c>, so no factor for the TW mean
/// enters the product. Applying this null-Wishart law to the eigenvalues of a sample
/// <em>correlation</em> matrix is an extrapolation neither cited paper states.
/// </para>
/// <para>
/// When every eigenvalue lies at or below the Tracy-Widom threshold (the all-noise
/// case, which is common when T &lt; N) no eigenvalue cleaning is applied and the
/// raw reconstruction passes through unchanged.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class TracyWidomDenoisedCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// <c>F1^-1(0.99)</c>, the 99th percentile of the Tracy-Widom law for real
    /// (beta = 1) ensembles; Johnstone (2001), Table 1, tabulates it as 2.02.
    /// </summary>
    private const double TracyWidomQuantile99 = 2.0234;

    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();

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

        var (corr, stdDevs) = CovarianceHelpers<T>.SampleCorrelation(returns);
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);
        var eigenvalues = eigen.Values;
        var eigenvectors = eigen.Vectors;

        // Tracy-Widom threshold computed in double for transcendental precision.
        var threshold = T.CreateChecked(TracyWidomCentering(t, n) + TracyWidomShift(t, n));

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

        var cleanedCorr = DenoisedCovarianceEstimator<T>.RescaleToUnitDiagonal(
            CovarianceHelpers<T>.ReconstructFromEigen(eigenvalues, eigenvectors));

        return CovarianceHelpers<T>.CorrelationToCovariance(cleanedCorr, stdDevs);
    }

    /// <summary>
    /// Tracy-Widom finite-N shift <c>(sigma_np / t) . s_0.99</c> added to the centering
    /// <see cref="TracyWidomCentering"/> to form the 99% signal/noise threshold.
    /// </summary>
    /// <remarks>
    /// Johnstone (2001), Theorem 1.1, gives <c>(lhat1 - mu_np) / sigma_np -&gt; W1 ~ F1</c>
    /// for <c>lhat1</c> the largest eigenvalue of <c>X^T X</c> over <c>n</c> observations of
    /// <c>p</c> variables, with <c>mu_np = (sqrt(n-1) + sqrt(p))^2</c> (eq. 1.3) and
    /// <c>sigma_np = (sqrt(n-1) + sqrt(p)) . (1/sqrt(n-1) + 1/sqrt(p))^(1/3)</c> (eq. 1.4).
    /// Here Johnstone's observation count <c>n</c> is this code's <paramref name="t"/> and his
    /// variable count <c>p</c> is this code's <paramref name="n"/>; dividing by
    /// <paramref name="t"/> maps the relation onto the correlation-eigenvalue scale.
    /// </remarks>
    internal static double TracyWidomShift(int t, int n)
    {
        var sqrtNm1 = Math.Sqrt(Math.Max(t, n) - 1);
        var sqrtP = Math.Sqrt(Math.Min(t, n));
        var sigmaNp = (sqrtNm1 + sqrtP) * Math.Pow(1.0 / sqrtNm1 + 1.0 / sqrtP, 1.0 / 3.0);
        return sigmaNp / t * TracyWidomQuantile99;
    }

    /// <summary>
    /// Johnstone (2001)'s centering, reversing dimension roles when t &lt; n, mapped onto the
    /// correlation-eigenvalue scale by dividing by the observation count <paramref name="t"/>.
    /// </summary>
    internal static double TracyWidomCentering(int t, int n)
    {
        var root = Math.Sqrt(Math.Max(t, n) - 1) + Math.Sqrt(Math.Min(t, n));
        return root * root / t;
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
[Obsolete(
    "Use TracyWidomDenoisedCovarianceEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class TracyWidomDenoisedCovarianceEstimator : ICovarianceEstimator
{
    private readonly TracyWidomDenoisedCovarianceEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
