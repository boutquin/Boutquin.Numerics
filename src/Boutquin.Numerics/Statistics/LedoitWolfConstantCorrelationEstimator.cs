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
/// Shrinkage intensity is the paper's Appendix B estimator
/// <c>delta* = max(0, min(kappa / T, 1))</c> with <c>kappa = (pi - rho) / gamma</c> (eq. 5):
/// <c>pi</c> sums the asymptotic variances of the sample covariance entries,
/// <c>rho</c> is the diagonal <c>pi_ii</c> plus the <c>r_bar / 2</c>-weighted
/// <c>theta_ii,ij</c> / <c>theta_jj,ij</c> cross terms, and
/// <c>gamma = Sum_ij (f_ij - s_ij)^2</c>. All three use the paper's 1/T moments of the
/// demeaned returns.
/// </para>
/// <para>
/// Evaluation: the target has the sample diagonal, so the diagonal terms cancel exactly
/// between <c>pi</c> and <c>rho</c> and vanish from <c>gamma</c>; they are omitted. Every
/// remaining term belongs to one pair <c>(i, j)</c> and carries the factor
/// <c>(c_i . c_j)^2</c> of the two columns' scales, so the terms, <c>r_bar</c> and the
/// standard deviations are formed from the demeaned columns divided by their own largest
/// magnitudes <c>c_i</c>, and each pair is weighted by <c>(c_i . c_j)^2</c> relative to the
/// largest pair. A column far below the largest keeps its correlations and its weight at any
/// scale instead of leaving the type's range (or decimal's grid) in a common scale.
/// </para>
/// <para>
/// Scale convention: the returned matrix blends the target and the sample covariance on
/// the library's Bessel (T - 1) scale, i.e. it is <c>T / (T - 1)</c> times the paper's
/// estimator. The intensity is identical to the paper's.
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
        if (typeof(T) == typeof(Half))
        {
            return CovarianceHelpers<T>.EstimateHalfShrinkage(returns, new LedoitWolfConstantCorrelationEstimator<double>());
        }

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var tCount = T.CreateChecked(t);
        var divisor = T.CreateChecked(t - 1);

        // Sample covariance in the original units; it conditions every pair by its own scales.
        var shifted = CovarianceHelpers<T>.ShiftToAnchor(returns);
        var means = CovarianceHelpers<T>.ComputeMeans(shifted);
        var sampleCov = CovarianceHelpers<T>.ComputeSampleCovariance(shifted, means);

        // Unit columns u_i = z_i / c_i: the demeaned returns divided by their own largest magnitude c_i, so
        // |u| <= 1. A moment of the pair (i, j) is c_i^p . c_j^q times the same moment of the unit columns.
        var columnScale = new T[n];
        var unit = new T[t, n];
        for (var j = 0; j < n; j++)
        {
            for (var k = 0; k < t; k++)
            {
                unit[k, j] = shifted[k, j] - means[j];
                columnScale[j] = T.Max(columnScale[j], T.Abs(unit[k, j]));
            }

            if (columnScale[j] != T.Zero)
            {
                for (var k = 0; k < t; k++)
                {
                    unit[k, j] /= columnScale[j];
                }
            }
        }

        // Honey (2004) App. B works on the 1/T moment matrix S_T of the demeaned returns; the blend below stays
        // on the library's Bessel scale, i.e. T/(T - 1) times the paper's estimator.
        var unitSum = new T[n, n];
        var unitMoment = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var acc = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    acc += unit[k, i] * unit[k, j];
                }

                unitSum[i, j] = acc;
                unitSum[j, i] = acc;
                unitMoment[i, j] = acc / tCount;
                unitMoment[j, i] = unitMoment[i, j];
            }
        }

        // Bessel standard deviations of the unit columns, and of the returns: sigma_i = c_i . sigma(u_i).
        var unitStdDev = new T[n];
        var stdDev = new T[n];
        for (var i = 0; i < n; i++)
        {
            unitStdDev[i] = NumericPrecision<T>.Sqrt(unitSum[i, i] / divisor);
            stdDev[i] = columnScale[i] * unitStdDev[i];
        }

        // Average pairwise correlation r_bar (correlations are invariant under the column scaling).
        var rSum = T.Zero;
        var pairCount = 0;
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                if (unitStdDev[i] > T.Zero && unitStdDev[j] > T.Zero)
                {
                    rSum += unitSum[i, j] / divisor / (unitStdDev[i] * unitStdDev[j]);
                    pairCount++;
                }
            }
        }

        var rBar = pairCount > 0 ? rSum / T.CreateChecked(pairCount) : T.Zero;

        // Retain physical pair-weight exponents: the largest pair can have an
        // identically zero contribution while much smaller pairs decide the intensity.
        // kappa-hat numerator pi-hat - rho-hat = Sum_{i != j} pi_ij - r_bar . Sum_{i != j} (sigma_j / sigma_i)
        // . theta_ii,ij with pi_ij = (1/T) Sum_t (z_it z_jt - s_ij)^2 and theta_ii,ij = (1/T) Sum_t (z_it^2 - s_ii)
        // (z_it z_jt - s_ij); the ordered-pair sum folds rho-hat's (r_bar / 2)-weighted theta_ii,ij and theta_jj,ij
        // halves. gamma-hat = Sum_{i != j} (s_ij - f_ij . (T - 1)/T)^2, f_ij = r_bar . sigma_i . sigma_j. A
        // zero-variance asset has an all-zero centered column, so every term it enters is zero and is skipped.
        var momentFactor = divisor / tCount;
        var numerator = T.Zero;
        var gamma = T.Zero;
        long numeratorExponent = 0;
        long gammaExponent = 0;
        for (var i = 0; i < n; i++)
        {
            if (columnScale[i] == T.Zero)
            {
                continue;
            }

            for (var j = 0; j < n; j++)
            {
                if (j == i || columnScale[j] == T.Zero)
                {
                    continue;
                }

                var (weight, weightExponent) = ScaledArithmetic<T>.ProductScaled(
                    columnScale[i], columnScale[j], columnScale[i], columnScale[j]);

                var piAcc = T.Zero;
                var thetaAcc = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    var cross = unit[k, i] * unit[k, j] - unitMoment[i, j];
                    piAcc += cross * cross;
                    thetaAcc += (unit[k, i] * unit[k, i] - unitMoment[i, i]) * cross;
                }

                var numeratorTerm = ScaledArithmetic<T>.MultiplyScaled(weight, weightExponent,
                    piAcc / tCount - rBar * (unitStdDev[j] / unitStdDev[i]) * (thetaAcc / tCount));
                ScaledArithmetic<T>.AddScaled(ref numerator, ref numeratorExponent, numeratorTerm.Significand, numeratorTerm.Exponent);
                var diff = unitMoment[i, j] - rBar * unitStdDev[i] * unitStdDev[j] * momentFactor;
                var gammaTerm = ScaledArithmetic<T>.MultiplyScaled(weight, weightExponent, diff);
                gammaTerm = ScaledArithmetic<T>.MultiplyScaled(gammaTerm.Significand, gammaTerm.Exponent, diff);
                ScaledArithmetic<T>.AddScaled(ref gamma, ref gammaExponent, gammaTerm.Significand, gammaTerm.Exponent);
            }
        }

        // delta* = max(0, min(kappa-hat / T, 1)), kappa-hat = (pi-hat - rho-hat) / gamma-hat [eq. 5].
        var (delta, deltaExponent) = gamma == T.Zero ? (T.One, 0L)
            : ScaledArithmetic<T>.UnitRatioScaled(numerator, numeratorExponent, gamma, gammaExponent, tCount);

        // Target F: diagonal = sample variance, off-diagonal = r_bar . sigma_i . sigma_j.
        var shrunk = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            shrunk[i, i] = sampleCov[i, i];
            for (var j = i + 1; j < n; j++)
            {
                var entry = ScaledArithmetic<T>.BlendScaled(sampleCov[i, j], rBar * stdDev[i] * stdDev[j], delta, deltaExponent);
                shrunk[i, j] = entry;
                shrunk[j, i] = entry;
            }
        }

        return shrunk;
    }
}

/// <summary>
/// Ledoit-Wolf shrinkage toward the <em>constant-correlation</em> target.
/// The target matrix has each asset's own variance on the diagonal and the
/// average pairwise sample correlation on every off-diagonal. This is the
/// target can be useful when pairwise population correlations are similar.
/// Its accuracy relative to the scaled-identity target of
/// <see cref="LedoitWolfShrinkageEstimator"/> depends on the population and sample.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2004). "Honey, I Shrunk the Sample
/// Covariance Matrix." Journal of Portfolio Management, 30(4), 110-119.
/// </para>
/// </remarks>
[Obsolete(
    "Use LedoitWolfConstantCorrelationEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class LedoitWolfConstantCorrelationEstimator : ICovarianceEstimator
{
    private readonly LedoitWolfConstantCorrelationEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
