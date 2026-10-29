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
/// Generic Oracle Approximating Shrinkage (OAS) — closed-form shrinkage intensity
/// that approximates the oracle shrinkage (which would require population
/// moments) under the Gaussian assumption.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A (arithmetic-only). Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Reference: Chen, Y., Wiesel, A., Eldar, Y. C., &amp; Hero, A. O. (2010).
/// "Shrinkage Algorithms for MMSE Covariance Estimation." IEEE Transactions
/// on Signal Processing, 58(10), 5016-5029.
/// </para>
/// <para>
/// Shrinkage intensity (Theorem 3 in the paper):
/// <c>rho* = min(1, ((1 - 2/p) . tr(S^2) + tr(S)^2) / ((n + 1 - 2/p) . (tr(S^2) - tr(S)^2/p)))</c>
/// where p = assets, n = observations, S = sum of raw outer products divided by n (known zero mean). The target
/// is the scaled-identity <c>F = (tr S / p) . I</c>.
/// </para>
/// <para>Half inputs widen exactly to double for estimation; only the final covariance is rounded back.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class OracleApproximatingShrinkageEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);

    // Relative tolerance for the eigenvalue-dispersion degeneracy test (scale-invariant).
    private static readonly T s_dispersionRelativeTolerance = T.CreateChecked(1e-12);

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            // Widen before forming second and fourth moments.
            return CovarianceHelpers<T>.EstimateHalfShrinkage(returns, new OracleApproximatingShrinkageEstimator<double>());
        }

        var t = returns.GetLength(0);
        var p = returns.GetLength(1);
        // Scale raw observations without subtracting a location: the mean is known zero.
        var scale = T.Zero;
        foreach (var value in returns)
        {
            scale = T.Max(scale, T.Abs(value));
        }

        if (scale == T.Zero)
        {
            return new T[p, p];
        }

        var sampleCov = new T[p, p];
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                for (var row = 0; row < t; row++)
                {
                    sampleCov[i, j] += (returns[row, i] / scale) * (returns[row, j] / scale) / T.CreateChecked(t);
                }
            }
        }

        // mu = tr(S) / p; target F = mu . I.
        var traceS = T.Zero;
        for (var i = 0; i < p; i++)
        {
            traceS += sampleCov[i, i];
        }

        var pT = T.CreateChecked(p);
        var mu = traceS / pT;

        // tr(S^2) = Sum_ij S_ij^2 (S symmetric).
        var traceS2 = T.Zero;
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                traceS2 += sampleCov[i, j] * sampleCov[i, j];
            }
        }

        // OAS rho* (Chen et al. 2010, Theorem 3).
        var nT = T.CreateChecked(t);
        var numerator = (T.One - s_two / pT) * traceS2 + traceS * traceS;

        // The denominator's only data-dependent factor is the eigenvalue-dispersion term
        // tr(S²) − tr(S)²/p (zero iff S is a scaled identity). The degeneracy test is made
        // scale-RELATIVE by comparing it to tr(S²): an absolute threshold spuriously fires
        // for small-magnitude inputs (where the denominator underflows the constant) and so
        // makes ρ — a scale-free quantity — depend on the magnitude of the returns.
        var dispersion = traceS2 - traceS * traceS / pT;
        T rho;
        if (dispersion <= s_dispersionRelativeTolerance * traceS2)
        {
            rho = T.One;
        }
        else
        {
            var denominator = (nT + T.One - s_two / pT) * dispersion;
            rho = numerator / denominator;
            if (rho > T.One)
            {
                rho = T.One;
            }
            else if (rho < T.Zero)
            {
                rho = T.Zero;
            }
        }

        // Sigma_hat = (1 - rho) . S + rho . F where F = mu . I. The intensity rho is a ratio of sums that the
        // largest columns dominate, so it is accurate in the common-scale coordinates above. The entries are
        // not: a column far below the common maximum has its second moments formed beyond the type's range
        // there, and a plain restore by scale^2 cannot recover them. Each entry is therefore blended in the
        // original units, with each S_ij product conditioned before division by the count.
        var muOriginal = (mu * scale) * scale;
        var shrunk = new T[p, p];
        for (var i = 0; i < p; i++)
        {
            for (var j = i; j < p; j++)
            {
                var moment = PairSecondMoment(returns, i, j);
                var entry = ScaledArithmetic<T>.BlendScaled(moment, i == j ? muOriginal : T.Zero, rho, 0);
                shrunk[i, j] = entry;
                shrunk[j, i] = entry;
            }
        }

        return shrunk;
    }

    /// <summary>
    /// Known-zero-mean second moment, retaining physical product exponents through the sum
    /// and dividing only once. Both normalization before amplification and division before
    /// aggregation can otherwise erase a representable entry.
    /// </summary>
    private static T PairSecondMoment(T[,] returns, int i, int j)
    {
        var count = T.CreateChecked(returns.GetLength(0));
        var sum = T.Zero;
        long exponent = 0;
        for (var row = 0; row < returns.GetLength(0); row++)
        {
            var term = ScaledArithmetic<T>.ProductScaled(returns[row, i], returns[row, j], T.One, T.One);
            ScaledArithmetic<T>.AddScaled(ref sum, ref exponent, term.Significand, term.Exponent);
        }

        return ScaledArithmetic<T>.ProductQuotient(sum, T.One, T.One, count, exponent);
    }
}

/// <summary>
/// Oracle Approximating Shrinkage (OAS) — closed-form shrinkage intensity
/// that approximates the oracle shrinkage (which would require population
/// moments) under the Gaussian assumption. Competes with Ledoit-Wolf and
/// often wins for Gaussian data, particularly at moderate sample sizes.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Chen, Y., Wiesel, A., Eldar, Y. C., &amp; Hero, A. O. (2010).
/// "Shrinkage Algorithms for MMSE Covariance Estimation." IEEE Transactions
/// on Signal Processing, 58(10), 5016-5029.
/// </para>
/// </remarks>
[Obsolete(
    "Use OracleApproximatingShrinkageEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class OracleApproximatingShrinkageEstimator : ICovarianceEstimator
{
    private readonly OracleApproximatingShrinkageEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
