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
/// where p = assets, n = observations, S = sample covariance. The target
/// is the scaled-identity <c>F = (tr S / p) . I</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class OracleApproximatingShrinkageEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly T s_denominatorTolerance = T.CreateChecked(1e-28);

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var p = returns.GetLength(1);

        var means = CovarianceHelpers<T>.ComputeMeans(returns);
        var sampleCov = CovarianceHelpers<T>.ComputeSampleCovariance(returns, means);

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
        var denominator = (nT + T.One - s_two / pT) * (traceS2 - traceS * traceS / pT);
        T rho;
        if (denominator <= s_denominatorTolerance)
        {
            rho = T.One;
        }
        else
        {
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

        // Sigma_hat = (1 - rho) . S + rho . F where F = mu . I.
        var shrunk = new T[p, p];
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                var target = i == j ? mu : T.Zero;
                shrunk[i, j] = (T.One - rho) * sampleCov[i, j] + rho * target;
            }
        }

        return shrunk;
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
public sealed class OracleApproximatingShrinkageEstimator : ICovarianceEstimator
{
    private readonly OracleApproximatingShrinkageEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
