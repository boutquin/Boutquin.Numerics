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

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Generic Ledoit-Wolf shrinkage toward the <em>constant-correlation</em> target.
/// The target matrix has each asset's own variance on the diagonal and the
/// average pairwise sample correlation on every off-diagonal.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/>.
/// </para>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2004). "Honey, I Shrunk the Sample
/// Covariance Matrix." Journal of Portfolio Management, 30(4), 110-119.
/// </para>
/// <para>
/// Shrinkage intensity follows the Schafer-Strimmer analytical formula
/// <c>delta* = Sum_ij Var(s_ij) / Sum_ij (s_ij - f_ij)^2</c> where <c>s</c> is the
/// sample covariance and <c>f</c> is the target. Clamped to [0, 1].
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class LedoitWolfConstantCorrelationEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var tCount = T.CreateChecked(t);

        var means = CovarianceHelpers<T>.ComputeMeans(returns);
        var sampleCov = CovarianceHelpers<T>.ComputeSampleCovariance(returns, means);

        // Standard deviations.
        var stdDev = new T[n];
        for (var i = 0; i < n; i++)
        {
            stdDev[i] = NumericPrecision<T>.Sqrt(sampleCov[i, i]);
        }

        // Average pairwise correlation r_bar.
        var rSum = T.Zero;
        var pairCount = 0;
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                if (stdDev[i] > T.Zero && stdDev[j] > T.Zero)
                {
                    rSum += sampleCov[i, j] / (stdDev[i] * stdDev[j]);
                    pairCount++;
                }
            }
        }

        var rBar = pairCount > 0 ? rSum / T.CreateChecked(pairCount) : T.Zero;

        // Target F: diagonal = sample variance, off-diagonal = r_bar . sigma_i . sigma_j.
        var target = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            target[i, i] = sampleCov[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var off = rBar * stdDev[i] * stdDev[j];
                target[i, j] = off;
                target[j, i] = off;
            }
        }

        // Numerator: pi = Sum_ij var(s_ij).
        var piSum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var acc = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    var dev = (returns[k, i] - means[i]) * (returns[k, j] - means[j]) - sampleCov[i, j];
                    acc += dev * dev;
                }

                piSum += acc / tCount;
            }
        }

        // Denominator: gamma = Sum_ij (s_ij - f_ij)^2.
        var gamma = T.Zero;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var diff = sampleCov[i, j] - target[i, j];
                gamma += diff * diff;
            }
        }

        var delta = gamma == T.Zero ? T.One : piSum / (tCount * gamma);
        delta = T.Max(T.Zero, T.Min(T.One, delta));

        var shrunk = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                shrunk[i, j] = delta * target[i, j] + (T.One - delta) * sampleCov[i, j];
            }
        }

        return shrunk;
    }
}

/// <summary>
/// Ledoit-Wolf shrinkage toward the <em>constant-correlation</em> target.
/// The target matrix has each asset's own variance on the diagonal and the
/// average pairwise sample correlation on every off-diagonal. This is the
/// correct target for equity universes where pairwise correlations are
/// similar across pairs — substantially outperforms the scaled-identity
/// target that <see cref="LedoitWolfShrinkageEstimator"/> uses.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2004). "Honey, I Shrunk the Sample
/// Covariance Matrix." Journal of Portfolio Management, 30(4), 110-119.
/// </para>
/// </remarks>
public sealed class LedoitWolfConstantCorrelationEstimator : ICovarianceEstimator
{
    private readonly LedoitWolfConstantCorrelationEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
