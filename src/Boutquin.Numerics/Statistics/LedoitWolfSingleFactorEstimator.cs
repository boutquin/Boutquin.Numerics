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
/// <para>
/// The paper's Theorem 2 estimates <c>k = (p - r) / c</c>, giving the intensity
/// <c>k / T</c> in equation (10). This library clips that ratio to <c>[0, 1]</c>
/// to retain a convex blend. Here <c>p</c> sums the
/// asymptotic variances of the sample covariance entries (Lemma 1), <c>r</c> sums their
/// asymptotic covariances with the target (Lemma 2: <c>r_ii = p_ii</c>, eq. 8 off the
/// diagonal), and <c>c = Sum_ij (f_ij - s_ij)^2</c> (Lemma 3). All three use the paper's
/// 1/T moments of the demeaned returns.
/// </para>
/// <para>
/// Scale convention: the returned matrix blends the target and the sample covariance on
/// the library's Bessel (T - 1) scale, i.e. it is <c>T / (T - 1)</c> times the paper's
/// estimator with the library's clipped intensity.
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
        if (typeof(T) == typeof(Half))
        {
            return CovarianceHelpers<T>.EstimateHalfShrinkage(returns, new LedoitWolfSingleFactorEstimator<double>());
        }

        var originalReturns = returns;
        var (workingReturns, scale) = CovarianceHelpers<T>.NormalizeForShrinkage(returns);
        returns = workingReturns;

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

        // The target entry beta_i . beta_j . Var(market) equals Cov(r_i, market) . Cov(r_j, market) / Var(market),
        // so the market's own scale cancels. The market deviations are carried as u_k = m_k / s_m (s_m their
        // largest magnitude) and every beta as loading_j = beta_j . s_m = Cov(r_j, u) / Var(u), a quantity in the
        // units of the returns bounded by sd(r_j) . sqrt(T - 1). A market that nearly cancels then neither
        // overflows a beta product (betas of 1e224) nor underflows its variance (1e-451 once normalized).
        var marketDeviation = new T[t];
        var marketScale = T.Zero;
        for (var i = 0; i < t; i++)
        {
            marketDeviation[i] = marketReturns[i] - marketMean;
            marketScale = T.Max(marketScale, T.Abs(marketDeviation[i]));
        }

        var unitMarket = new T[t];
        if (marketScale != T.Zero)
        {
            for (var i = 0; i < t; i++)
            {
                unitMarket[i] = marketDeviation[i] / marketScale;
            }
        }

        // Var(u) lies in [1 / (T - 1), T / (T - 1)] for a non-constant market.
        var unitMarketVar = T.Zero;
        for (var i = 0; i < t; i++)
        {
            unitMarketVar += unitMarket[i] * unitMarket[i];
        }

        unitMarketVar /= divisor;

        // Per-asset loading on the market.
        var loading = new T[n];
        for (var j = 0; j < n; j++)
        {
            var acc = T.Zero;
            for (var i = 0; i < t; i++)
            {
                acc += (returns[i, j] - means[j]) * unitMarket[i];
            }

            loading[j] = unitMarketVar != T.Zero ? (acc / divisor) / unitMarketVar : T.Zero;
        }

        // Target F: off-diagonal = loading_i . loading_j . Var(u), diagonal = sample variance. Var(u) lies in
        // [1 / (T - 1), T / (T - 1)] and these coordinates are bounded (unnormalized only while the fourth-order
        // bound fits, otherwise divided by the common maximum), so the product stays in range here; the
        // original-units target of the normalized path below needs the balanced order.
        var target = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            target[i, i] = sampleCov[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var off = loading[i] * loading[j] * unitMarketVar;
                target[i, j] = off;
                target[j, i] = off;
            }
        }

        // LW (2003) section 2.6 works on 1/T moments of the demeaned returns: s_ij, the
        // market covariances s_i0 and the market variance s_00. The betas s_i0 / s_00 are
        // scale-free, so the Bessel loadings above serve (the 1/T moments of the unit market u are
        // those of the market divided by s_m^2); the blend below stays on the
        // library's Bessel scale, i.e. T/(T - 1) times the paper's estimator.
        var momentFactor = divisor / tCount;
        var momentMarketVar = unitMarketVar * momentFactor;

        // The intensity is evaluated pair by pair on per-column unit coordinates. Write z_i = c_i v_i with c_i the
        // column's largest anchored magnitude (in the original units), so |v_i| <= 2, and loading_i = c_i l_i with
        // l_i the loading of v_i on the unit market. Every off-diagonal p_ij, r_ij and (f_ij - s_ij)^2 is then
        // (c_i c_j)^2 times the same expression in v_i, v_j, l_i, l_j, so each pair is formed at full precision in
        // its own coordinates and weighted by (c_i c_j)^2, with the radix exponent retained separately
        // through both sums and their final ratio. The diagonal is not summed: r_ii = p_ii [Lemma 2] cancels
        // exactly from p - r, and f_ii = s_ii makes the diagonal of c zero. A common scale (or a dominant column's
        // p_ii) would otherwise round away the pair terms that decide the intensity.
        var originalShifted = CovarianceHelpers<T>.ShiftToAnchor(originalReturns);
        var columnScale = new T[n];
        var unit = new T[t, n];
        for (var j = 0; j < n; j++)
        {
            for (var k = 0; k < t; k++)
            {
                columnScale[j] = T.Max(columnScale[j], T.Abs(originalShifted[k, j]));
            }

            if (columnScale[j] == T.Zero)
            {
                continue;
            }

            var unitMean = T.Zero;
            for (var k = 0; k < t; k++)
            {
                unit[k, j] = originalShifted[k, j] / columnScale[j];
                unitMean += unit[k, j];
            }

            unitMean /= tCount;
            for (var k = 0; k < t; k++)
            {
                unit[k, j] -= unitMean;
            }
        }

        var unitLoading = new T[n];
        for (var j = 0; j < n; j++)
        {
            var acc = T.Zero;
            for (var k = 0; k < t; k++)
            {
                acc += unit[k, j] * unitMarket[k];
            }

            unitLoading[j] = unitMarketVar != T.Zero ? (acc / divisor) / unitMarketVar : T.Zero;
        }

        var numerator = T.Zero;
        var gamma = T.Zero;
        long numeratorExponent = 0;
        long gammaExponent = 0;
        // With fewer than two non-constant columns every pair term is zero (and c = 0 gives delta = 1).
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var (weight, weightExponent) = ScaledArithmetic<T>.ProductScaled(
                    columnScale[i], columnScale[j], columnScale[i], columnScale[j]);
                if (weight == T.Zero)
                {
                    continue;
                }

                var moment = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    moment += unit[k, i] * unit[k, j];
                }

                moment /= tCount;

                // p_ij = (1/T) Sum_t {z_it z_jt - s_ij}^2 [Lemma 1].
                var p = T.Zero;

                // Off the diagonal, eq. 8 with s_i0 / s_00 = beta_i:
                // r_ij = (1/T) Sum_t (beta_j z_it + beta_i z_jt - beta_i beta_j z_0t) z_0t z_it z_jt - f_ij s_ij,
                // where f_ij = beta_i beta_j s_00. With z_0t = s_m u_t and loading = beta . s_m the market scale
                // cancels: (beta_j z_it + beta_i z_jt - beta_i beta_j z_0t) z_0t = (loading_j z_it + loading_i z_jt) u_t
                // - loading_i loading_j u_t^2.
                var r = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    var product = unit[k, i] * unit[k, j];
                    var dev = product - moment;
                    p += dev * dev;
                    var weighted = (unitLoading[j] * unit[k, i] + unitLoading[i] * unit[k, j]) * unitMarket[k]
                        - unitLoading[i] * unitLoading[j] * unitMarket[k] * unitMarket[k];
                    r += weighted * product;
                }

                var factorMoment = unitLoading[i] * unitLoading[j] * momentMarketVar;
                r = r / tCount - factorMoment * moment;

                // c_ij = (f_ij - s_ij)^2 [Lemma 3].
                var diff = moment - factorMoment;
                var numeratorTerm = ScaledArithmetic<T>.MultiplyScaled(weight, weightExponent, p / tCount - r);
                ScaledArithmetic<T>.AddScaled(ref numerator, ref numeratorExponent, numeratorTerm.Significand, numeratorTerm.Exponent);
                var gammaTerm = ScaledArithmetic<T>.MultiplyScaled(weight, weightExponent, diff);
                gammaTerm = ScaledArithmetic<T>.MultiplyScaled(gammaTerm.Significand, gammaTerm.Exponent, diff);
                ScaledArithmetic<T>.AddScaled(ref gamma, ref gammaExponent, gammaTerm.Significand, gammaTerm.Exponent);
            }
        }

        // Theorem 2 / eq. (10) gives k/T; clipping it to [0,1] is this library's variant.
        var (delta, deltaExponent) = gamma == T.Zero ? (T.One, 0L)
            : ScaledArithmetic<T>.UnitRatioScaled(numerator, numeratorExponent, gamma, gammaExponent, tCount);

        var shrunk = new T[n, n];
        if (scale == T.One)
        {
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    shrunk[i, j] = ScaledArithmetic<T>.BlendScaled(sampleCov[i, j], target[i, j], delta, deltaExponent);
                }
            }

            return shrunk;
        }

        // Normalized path. The intensity above retains the physical exponents of the fourth-order pair sums. The entries are not: a column far
        // below the common maximum has its products formed beyond the type's range there (and, in decimal,
        // its digits rounded off by the division), and a plain restore by scale^2 cannot recover them. Blend
        // in the original units instead: the sample covariance of the original returns (which conditions every
        // pair by its own scales) and the target loading_i . loading_j . Var(u) of the original columns.
        var originalMeans = CovarianceHelpers<T>.ComputeMeans(originalShifted);
        var originalCov = CovarianceHelpers<T>.ComputeSampleCovariance(originalShifted, originalMeans);
        var originalLoading = new T[n];
        for (var j = 0; j < n; j++)
        {
            var acc = T.Zero;
            for (var k = 0; k < t; k++)
            {
                acc += (originalShifted[k, j] - originalMeans[j]) * unitMarket[k];
            }

            originalLoading[j] = unitMarketVar != T.Zero ? (acc / divisor) / unitMarketVar : T.Zero;
        }

        for (var i = 0; i < n; i++)
        {
            // The target and the sample covariance share the diagonal, so it is the sample variance.
            shrunk[i, i] = originalCov[i, i];
            for (var j = i + 1; j < n; j++)
            {
                // loading_i . loading_j alone is the target divided by Var(u), and Var(u) can be as small as
                // 1 / (T - 1): it leaves the range where the target does not. Balance the order instead.
                var originalTarget = CovarianceHelpers<T>.ScaleSafeProduct(originalLoading[i], originalLoading[j], unitMarketVar);
                var entry = ScaledArithmetic<T>.BlendScaled(originalCov[i, j], originalTarget, delta, deltaExponent);
                shrunk[i, j] = entry;
                shrunk[j, i] = entry;
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
[Obsolete(
    "Use LedoitWolfSingleFactorEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class LedoitWolfSingleFactorEstimator : ICovarianceEstimator
{
    private readonly LedoitWolfSingleFactorEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
