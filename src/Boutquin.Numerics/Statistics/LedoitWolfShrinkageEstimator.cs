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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Generic Ledoit-Wolf linear shrinkage estimator. Blends the sample covariance <c>S</c>
/// with the scaled-identity target <c>F = mu . I</c> (where <c>mu = avg(diag(S))</c>)
/// using the asymptotically optimal shrinkage intensity <c>delta* in [0, 1]</c>.
/// PSD by construction — a convex combination of two PSD matrices (the sample
/// covariance and the scaled identity) is PSD.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2004). "A well-conditioned estimator
/// for large-dimensional covariance matrices." Journal of Multivariate
/// Analysis, 88(2), 365-411.
/// </para>
/// <para>
/// Intensity formula: <c>delta* = (pi - rho) / (T . gamma)</c>, clamped to <c>[0, 1]</c>,
/// where <c>pi</c> is the sum of asymptotic variances of sample covariance
/// entries, <c>rho</c> is the sum of asymptotic covariances of sample entries
/// with the target, and <c>gamma = ||S - F||^2_F</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class LedoitWolfShrinkageEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        // Step 1: sample covariance.
        var sampleCov = s_sampleEstimator.Estimate(returns);

        // Step 2: target F = mu . I, mu = average of diagonal entries.
        var mu = T.Zero;
        for (var i = 0; i < n; i++)
        {
            mu += sampleCov[i, i];
        }

        mu /= T.CreateChecked(n);

        // Step 3: gamma = ||S - F||_F^2 (squared Frobenius norm of the deviation).
        var gamma = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var target = i == j ? mu : T.Zero;
                var diff = sampleCov[i, j] - target;
                gamma += diff * diff;
            }
        }

        // Compute deviations once, reused for pi and rho.
        var means = CovarianceHelpers<T>.ComputeMeans(returns);

        // Step 4: pi = Sum_ij asymptotic variance of s_ij.
        var tCount = T.CreateChecked(t);
        var piSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    var x = (returns[k, i] - means[i]) * (returns[k, j] - means[j]) - sampleCov[i, j];
                    sum += x * x;
                }

                piSum += sum / tCount;
            }
        }

        // Step 5: rho = diagonal terms only (off-diagonal target is zero).
        var rhoSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            var sum = T.Zero;
            for (var k = 0; k < t; k++)
            {
                var zki = returns[k, i] - means[i];
                var term = zki * zki - sampleCov[i, i];
                sum += term * term;
            }

            rhoSum += sum / tCount;
        }

        // Step 6: shrinkage intensity delta in [0, 1].
        var delta = gamma == T.Zero ? T.One : (piSum - rhoSum) / (tCount * gamma);
        delta = T.Max(T.Zero, T.Min(T.One, delta));

        // Step 7: shrunk covariance = delta . F + (1 - delta) . S.
        var shrunk = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var target = i == j ? mu : T.Zero;
                shrunk[i, j] = delta * target + (T.One - delta) * sampleCov[i, j];
            }
        }

        return shrunk;
    }
}

/// <summary>
/// Ledoit-Wolf linear shrinkage estimator. Blends the sample covariance <c>S</c>
/// with the scaled-identity target <c>F = mu . I</c> (where <c>mu = avg(diag(S))</c>)
/// using the asymptotically optimal shrinkage intensity <c>delta* in [0, 1]</c>.
/// PSD by construction — a convex combination of two PSD matrices (the sample
/// covariance and the scaled identity) is PSD.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2004). "A well-conditioned estimator
/// for large-dimensional covariance matrices." Journal of Multivariate
/// Analysis, 88(2), 365-411.
/// </para>
/// <para>
/// Sample regime: preferred over <see cref="SampleCovarianceEstimator"/>
/// whenever the concentration ratio <c>c = N / T</c> is non-negligible.
/// The optimal intensity is derived from large-dimensional random matrix
/// theory and does not require a user-tuned regularization parameter.
/// </para>
/// <para>
/// The scaled-identity target is the right choice when asset variances are
/// broadly similar and no prior structural information is available. For
/// universes with dominant pairwise correlations, prefer
/// <see cref="LedoitWolfConstantCorrelationEstimator"/>; when a market factor
/// explains most of the covariance, prefer
/// <see cref="LedoitWolfSingleFactorEstimator"/>.
/// </para>
/// </remarks>
public sealed class LedoitWolfShrinkageEstimator : ICovarianceEstimator
{
    private readonly LedoitWolfShrinkageEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
