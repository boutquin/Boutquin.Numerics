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
/// Generic Ledoit-Wolf shrinkage toward a <em>single-factor</em> (market) target.
/// The target covariance is induced by a one-factor model where each asset
/// loads on an equally-weighted market factor.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2003). "Improved Estimation of
/// the Covariance Matrix of Stock Returns with an Application to Portfolio
/// Selection." Journal of Empirical Finance, 10(5), 603-621.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class LedoitWolfSingleFactorEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var tCount = T.CreateChecked(t);
        var nCount = T.CreateChecked(n);
        var divisor = T.CreateChecked(t - 1);

        var means = CovarianceHelpers<T>.ComputeMeans(returns);
        var sampleCov = CovarianceHelpers<T>.ComputeSampleCovariance(returns, means);

        // Market factor = equal-weight average of asset returns.
        var marketReturns = new T[t];
        var marketMean = T.Zero;
        for (var i = 0; i < t; i++)
        {
            var sum = T.Zero;
            for (var j = 0; j < n; j++)
            {
                sum += returns[i, j];
            }

            marketReturns[i] = sum / nCount;
            marketMean += marketReturns[i];
        }

        marketMean /= tCount;

        // Market variance (sample).
        var marketVar = T.Zero;
        for (var i = 0; i < t; i++)
        {
            var dev = marketReturns[i] - marketMean;
            marketVar += dev * dev;
        }

        marketVar /= divisor;

        // Per-asset covariance with the market factor.
        var covWithMarket = new T[n];
        for (var j = 0; j < n; j++)
        {
            var acc = T.Zero;
            for (var i = 0; i < t; i++)
            {
                acc += (returns[i, j] - means[j]) * (marketReturns[i] - marketMean);
            }

            covWithMarket[j] = acc / divisor;
        }

        // Betas.
        var beta = new T[n];
        for (var j = 0; j < n; j++)
        {
            beta[j] = marketVar != T.Zero ? covWithMarket[j] / marketVar : T.Zero;
        }

        // Target F: off-diagonal = beta_i . beta_j . Var(market), diagonal = sample variance.
        var target = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            target[i, i] = sampleCov[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var off = beta[i] * beta[j] * marketVar;
                target[i, j] = off;
                target[j, i] = off;
            }
        }

        // Numerator pi and denominator gamma (same Schafer-Strimmer form as LW-CC).
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
/// Ledoit-Wolf shrinkage toward a <em>single-factor</em> (market) target.
/// The target covariance is induced by a one-factor model where each asset
/// loads on an equally-weighted market factor. The off-diagonal of the
/// target is <c>beta_i . beta_j . Var(market)</c> with <c>beta_i = Cov(r_i, market) / Var(market)</c>,
/// and the diagonal is the sample variance (idiosyncratic + factor).
/// </summary>
/// <remarks>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2003). "Improved Estimation of
/// the Covariance Matrix of Stock Returns with an Application to Portfolio
/// Selection." Journal of Empirical Finance, 10(5), 603-621.
/// </para>
/// </remarks>
public sealed class LedoitWolfSingleFactorEstimator : ICovarianceEstimator
{
    private readonly LedoitWolfSingleFactorEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
