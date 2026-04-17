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
/// Generic Quadratic-Inverse Shrinkage (QIS) — Ledoit-Wolf analytical nonlinear
/// shrinkage. Shrinks each sample eigenvalue individually using the
/// quadratic-inverse kernel estimator.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="Internal.NumericPrecision{T}.Sqrt"/> via <see cref="CovarianceHelpers{T}"/>.
/// </para>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2022). "Quadratic shrinkage for
/// large covariance matrices." Bernoulli, 28(3), 1519-1547.
/// </para>
/// <para>
/// The implementation follows the sample-eigenvalue formulation in the
/// Ledoit-Wolf 2022 paper (Section 4.1). Internals use
/// <see langword="double"/> precision for numerical stability of the
/// kernel arithmetic, with <typeparamref name="T"/> at the boundary.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class QuadraticInverseShrinkageEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        // Step 1: sample covariance.
        var sampleCov = new SampleCovarianceEstimator<T>().Estimate(returns);

        // Step 2: convert to correlation so the eigenvalues are on a comparable scale.
        var (corr, stdDevs) = CovarianceHelpers<T>.CovarianceToCorrelation(sampleCov);

        // Step 3: eigendecomposition of the correlation matrix.
        var eigen = JacobiEigenDecomposition<T>.Decompose(corr);

        // Sample eigenvalues in ascending order for QIS (paper convention).
        // JacobiEigenDecomposition returns descending — reverse.
        var lambda = new double[n];
        for (var i = 0; i < n; i++)
        {
            lambda[i] = double.CreateChecked(eigen.Values[n - 1 - i]);
        }

        // Concentration ratio c = p/n. Paper uses 1/c too.
        var c = (double)n / t;

        // Sorted eigenvalues for kernel evaluation.
        var h = Math.Pow((double)t, -0.35); // bandwidth: Ledoit-Wolf default t^(-0.35).

        // Step 4: QIS eigenvalue transform.
        var shrunk = new double[n];
        var effectiveRank = Math.Min(n, t - 1);
        for (var i = 0; i < n; i++)
        {
            var ell = lambda[i];
            var density = 0.0;
            var hilbert = 0.0;
            for (var j = 0; j < effectiveRank; j++)
            {
                var ej = lambda[n - effectiveRank + j];
                var u = (ell - ej) / (h * ell);
                var disc = 4.0 - u * u;
                if (disc > 0.0 && Math.Abs(u) > 1e-12)
                {
                    density += Math.Sqrt(disc) / (2.0 * Math.PI * u * u);
                }

                if (Math.Abs(u) >= 2.0)
                {
                    hilbert += (Math.Sign(u) / (Math.PI * u * u)) *
                               (Math.Abs(u) * 0.5 * Math.Sqrt(u * u - 4.0) - 1.0);
                }
                else if (Math.Abs(u) > 1e-12)
                {
                    hilbert += -1.0 / (Math.PI * u * u);
                }
            }

            density /= n * h * ell;
            hilbert /= n * h * ell;

            var a = Math.PI * c * ell * density;
            var b = 1.0 - c - Math.PI * c * ell * hilbert;
            var denom = a * a + b * b;
            shrunk[i] = denom > 1e-18 ? ell / denom : ell;
        }

        // Step 5: reassemble correlation matrix with shrunk eigenvalues
        // (reverse back to descending order used by our EigenResult).
        var shrunkDesc = new T[n];
        for (var i = 0; i < n; i++)
        {
            shrunkDesc[i] = T.CreateChecked(shrunk[n - 1 - i]);
        }

        var cleanedCorr = CovarianceHelpers<T>.ReconstructFromEigen(shrunkDesc, eigen.Vectors);

        // Force unit diagonal (removes numerical drift).
        for (var i = 0; i < n; i++)
        {
            cleanedCorr[i, i] = T.One;
        }

        // Step 6: back to covariance.
        return CovarianceHelpers<T>.CorrelationToCovariance(cleanedCorr, stdDevs);
    }
}

/// <summary>
/// Quadratic-Inverse Shrinkage (QIS) — Ledoit-Wolf analytical nonlinear
/// shrinkage. Shrinks each sample eigenvalue individually using the
/// quadratic-inverse kernel estimator, producing the oracle-optimal
/// rotation-equivariant estimator up to asymptotic order.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2022). "Quadratic shrinkage for
/// large covariance matrices." Bernoulli, 28(3), 1519-1547.
/// </para>
/// </remarks>
public sealed class QuadraticInverseShrinkageEstimator : ICovarianceEstimator
{
    private readonly QuadraticInverseShrinkageEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
