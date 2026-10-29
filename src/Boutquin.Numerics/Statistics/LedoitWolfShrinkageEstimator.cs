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
/// Intensity (eq. 14 with Lemmas 3.2-3.4): <c>delta = b^2 / d^2</c> with
/// <c>d^2 = ||S_T - m . I||^2</c>, <c>b-bar^2 = (1/T^2) . Sum_k ||z_k z_k' - S_T||^2</c> and
/// <c>b^2 = min(b-bar^2, d^2)</c>, so <c>delta</c> lies in <c>[0, 1]</c>. <c>S_T</c> is the
/// paper's 1/T moment matrix of the demeaned returns <c>z_k</c> and <c>m = tr(S_T) / N</c>.
/// There is no <c>rho</c> term in this estimator.
/// </para>
/// <para>
/// Scale convention: the returned matrix is <c>delta . mu . I + (1 - delta) . S</c> on the
/// library's Bessel scale <c>S = T / (T - 1) . S_T</c>, i.e. <c>T / (T - 1)</c> times the
/// paper's estimator. The intensity is identical to the paper's.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class LedoitWolfShrinkageEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            return CovarianceHelpers<T>.EstimateHalfShrinkage(returns, new LedoitWolfShrinkageEstimator<double>());
        }

        var shifted = CovarianceHelpers<T>.ShiftToAnchor(returns);
        var means = CovarianceHelpers<T>.ComputeMeans(shifted);
        var sampleCov = CovarianceHelpers<T>.ComputeSampleCovariance(shifted, means);
        var t = returns.GetLength(0);
        var n = returns.GetLength(1);
        var tCount = T.CreateChecked(t);
        var momentFactor = T.CreateChecked(t - 1) / tCount;
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

        var mu = T.Zero;
        for (var i = 0; i < n; i++)
        {
            mu += sampleCov[i, i] / T.CreateChecked(n);
        }

        // JMVA 2004 intensity on 1/T moments, with each fourth-order pair
        // retained in its own physical scale until the final ratio and blend.
        var numerator = T.Zero;
        var dispersion = T.Zero;
        long numeratorExponent = 0;
        long dispersionExponent = 0;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var moment = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    moment += unit[k, i] * unit[k, j];
                }
                moment /= tCount;

                var pi = T.Zero;
                for (var k = 0; k < t; k++)
                {
                    var deviation = unit[k, i] * unit[k, j] - moment;
                    pi += deviation * deviation;
                }
                pi /= tCount;
                var pair = ScaledArithmetic<T>.ProductScaled(columnScale[i], columnScale[j], columnScale[i], columnScale[j]);
                var term = ScaledArithmetic<T>.MultiplyScaled(pair.Significand, pair.Exponent, pi);
                ScaledArithmetic<T>.AddScaled(ref numerator, ref numeratorExponent, term.Significand, term.Exponent);

                var difference = (sampleCov[i, j] - (i == j ? mu : T.Zero)) * momentFactor;
                term = ScaledArithmetic<T>.ProductScaled(difference, difference, T.One, T.One);
                ScaledArithmetic<T>.AddScaled(ref dispersion, ref dispersionExponent, term.Significand, term.Exponent);
            }
        }

        var (delta, deltaExponent) = dispersion == T.Zero ? (T.One, 0L)
            : ScaledArithmetic<T>.UnitRatioScaled(numerator, numeratorExponent, dispersion, dispersionExponent, tCount);
        var shrunk = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                shrunk[i, j] = ScaledArithmetic<T>.BlendScaled(sampleCov[i, j], i == j ? mu : T.Zero, delta, deltaExponent);
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
[Obsolete(
    "Use LedoitWolfShrinkageEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class LedoitWolfShrinkageEstimator : ICovarianceEstimator
{
    private readonly LedoitWolfShrinkageEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
