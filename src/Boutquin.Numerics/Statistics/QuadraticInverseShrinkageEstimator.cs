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
using Boutquin.Numerics.LinearAlgebra;

namespace Boutquin.Numerics.Statistics;

/// <summary>
/// Generic Quadratic-Inverse Shrinkage (QIS) — Ledoit-Wolf (2022) analytical nonlinear
/// shrinkage of the sample covariance matrix. Each sample eigenvalue is replaced by the
/// quadratic-inverse shrinkage of its inverse, computed with a Cauchy kernel.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. The eigendecomposition, the smoothed Stein shrinker,
/// and the eigenvalue transform run in <typeparamref name="T"/>, except that Half inputs
/// widen exactly to double and only the final covariance is rounded back. The fractional
/// bandwidth powers are evaluated by a bounded integer-root helper; square roots (inside the
/// Jacobi eigendecomposition) use <see cref="NumericPrecision{T}.Sqrt(T)"/>.
/// </para>
/// <para>
/// Reference: Ledoit, O. &amp; Wolf, M. (2022). "Quadratic shrinkage for
/// large covariance matrices." Bernoulli, 28(3), 1519-1547.
/// </para>
/// <para>
/// Algorithm (the paper's Appendix A code with its default demeaning): the sample
/// covariance <c>S = Y'Y / n</c> of the demeaned returns with <c>n = T - 1</c>, concentration
/// <c>c = N / n</c>, ascending eigenvalues <c>lambda</c>, bandwidth
/// <c>h = min(c^2, 1/c^2)^0.35 / N^0.35</c>, and over the <c>min(N, n)</c> non-null inverse
/// eigenvalues the smoothed Stein shrinker <c>theta</c> and its conjugate <c>H-theta</c>.
/// When <c>N &lt;= n</c> each eigenvalue becomes
/// <c>1 / ((1 - c)^2 / lambda + 2c(1 - c) theta / lambda + c^2 |theta|^2 / lambda)</c>;
/// when <c>N &gt; n</c> the non-null eigenvalues become <c>1 / (|theta|^2 / lambda)</c> and the
/// <c>N - n</c> null eigenvalues receive <c>1 / ((c - 1) . mean(1 / lambda))</c>. The shrunk
/// eigenvalues are rescaled to preserve the trace and recombined with the sample
/// eigenvectors. The result is positive definite whenever the non-null eigenvalues are.
/// </para>
/// <para>
/// Library extensions: the spectral step runs on <c>S / (tr(S) / N)</c> and the result is
/// scaled back (QIS is scale-equivariant, so this only keeps inverse eigenvalues
/// representable). The trace-preserving rescale and the recombination with the eigenvectors
/// are also carried out in those normalised units, and the physical scale is restored once,
/// on the reconstructed entries; the estimate is therefore representable whenever its entries
/// are, even when its leading eigenvalue (up to the trace) is not. An eigenvalue inside the
/// non-null block that is non-positive is kept at zero and excluded from the kernel averages; the
/// paper assumes that block is positive. A matrix with zero trace is returned unchanged.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class QuadraticInverseShrinkageEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();
    private static readonly T s_two = T.CreateChecked(2);

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            // Keep observation counts and spectral intermediates representable.
            return CovarianceHelpers<T>.EstimateHalfShrinkage(returns, new QuadraticInverseShrinkageEstimator<double>());
        }

        var observations = returns.GetLength(0);
        var p = returns.GetLength(1);

        // Appendix A default: demean and use n = T - 1, so the sample matrix is the
        // Bessel sample covariance.
        var sampleCov = s_sampleEstimator.Estimate(returns);
        var n = observations - 1;

        var diagonalScale = T.Zero;
        for (var i = 0; i < p; i++)
        {
            diagonalScale = T.Max(diagonalScale, sampleCov[i, i]);
        }

        if (diagonalScale <= T.Zero)
        {
            return sampleCov;
        }

        var normalizedTrace = T.Zero;
        for (var i = 0; i < p; i++)
        {
            normalizedTrace += sampleCov[i, i] / diagonalScale;
        }

        var scale = diagonalScale * (normalizedTrace / T.CreateChecked(p));
        var normalized = new T[p, p];
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                normalized[i, j] = sampleCov[i, j] / scale;
            }
        }

        // Spectral decomposition; JacobiEigenDecomposition returns descending order,
        // the paper works in ascending order.
        var eigen = JacobiEigenDecomposition<T>.Decompose(normalized);
        var descending = eigen.Values;
        var eigenvectors = eigen.Vectors;
        var lambda = new T[p];
        var lambdaSum = T.Zero;
        for (var i = 0; i < p; i++)
        {
            lambda[i] = descending[p - 1 - i];
            lambdaSum += lambda[i];
        }

        // Non-null block: the top min(p, n) eigenvalues (p - n null ones when p > n).
        var nullCount = Math.Max(0, p - n);
        var active = new List<int>(p - nullCount);
        for (var i = nullCount; i < p; i++)
        {
            if (lambda[i] > T.Zero)
            {
                active.Add(i);
            }
        }

        if (active.Count == 0)
        {
            return sampleCov;
        }

        // Reciprocal ratios, bounded by one, suffice for the harmonic mean used by the null block.
        var smallest = lambda[active[0]];
        var inverseSum = T.Zero;
        for (var a = 0; a < active.Count; a++)
        {
            inverseSum += smallest / lambda[active[a]];
        }

        var c = T.CreateChecked(p) / T.CreateChecked(n);

        // h = min(c^2, 1/c^2)^0.35 / p^0.35 = (min(p, n) / max(p, n))^0.7 . p^(-0.35).
        var h = NumericPrecision<T>.NegativeRationalPowerOfPositiveInteger(Math.Max(p, n), 14, 20)
            / NumericPrecision<T>.NegativeRationalPowerOfPositiveInteger(Math.Min(p, n), 14, 20)
            * NumericPrecision<T>.NegativeRationalPowerOfPositiveInteger(p, 7, 20);
        var hSquared = h * h;
        var activeCount = T.CreateChecked(active.Count);
        var oneMinusC = T.One - c;

        var delta = new T[p];
        for (var i = 0; i < p; i++)
        {
            delta[i] = T.Zero;
        }

        for (var a = 0; a < active.Count; a++)
        {
            // Multiplying each Cauchy-kernel numerator and denominator by (lambda_i lambda_j)^2
            // cancels the reciprocals: theta's numerator is lambda_i (lambda_i-lambda_j),
            // H-theta's is h lambda_i^2. Pair scaling keeps both squares bounded even when
            // a resolved eigenvalue is many orders below the largest.
            var theta = T.Zero;
            var hTheta = T.Zero;
            for (var b = 0; b < active.Count; b++)
            {
                var pairScale = T.Max(lambda[active[a]], lambda[active[b]]);
                var own = lambda[active[a]] / pairScale;
                var other = lambda[active[b]] / pairScale;
                var gap = own - other;
                var denominator = gap * gap + hSquared * own * own;
                theta += own * gap / denominator;
                hTheta += own * h * own / denominator;
            }

            theta /= activeCount;
            hTheta /= activeCount;

            if (p <= n)
            {
                // c^2 |theta|^2 is formed as (c . theta)^2 + (c . H-theta)^2: for long samples
                // H-theta grows like 1/h while c shrinks, and squaring each factor separately
                // overflows H-theta^2 and underflows c^2 in narrow types (Half), giving 0 . inf.
                var cTheta = c * theta;
                var cHTheta = c * hTheta;
                delta[active[a]] = lambda[active[a]] / (oneMinusC * oneMinusC
                                                          + s_two * oneMinusC * cTheta
                                                          + cTheta * cTheta
                                                          + cHTheta * cHTheta);
            }
            else
            {
                delta[active[a]] = lambda[active[a]] / (theta * theta + hTheta * hTheta);
            }
        }

        if (nullCount > 0)
        {
            var nullDelta = smallest / ((c - T.One) * (inverseSum / activeCount));
            for (var i = 0; i < nullCount; i++)
            {
                delta[i] = nullDelta;
            }
        }

        // Preserve the trace and recombine in descending order, all in normalised units: both
        // the trace ratio and the shrunk eigenvalues are O(1) there (the trace is p).
        var deltaSum = T.Zero;
        for (var i = 0; i < p; i++)
        {
            deltaSum += delta[i];
        }

        var rescale = lambdaSum / deltaSum;
        var shrunkDescending = new T[p];
        for (var i = 0; i < p; i++)
        {
            shrunkDescending[i] = delta[p - 1 - i] * rescale;
        }

        // Undo the normalisation once, on the reconstructed entries. Folding the physical scale
        // into the ratio or the eigenvalues would leave the range of T whenever the leading
        // eigenvalue (up to the trace) exceeds it, although every entry of the result fits.
        var shrunk = CovarianceHelpers<T>.ReconstructFromEigen(shrunkDescending, eigenvectors);
        for (var i = 0; i < p; i++)
        {
            for (var j = i; j < p; j++)
            {
                var restored = shrunk[i, j] * scale;
                shrunk[i, j] = restored;
                shrunk[j, i] = restored;
            }
        }

        return shrunk;
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
[Obsolete(
    "Use QuadraticInverseShrinkageEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class QuadraticInverseShrinkageEstimator : ICovarianceEstimator
{
    private readonly QuadraticInverseShrinkageEstimator<decimal> _inner = new();

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
