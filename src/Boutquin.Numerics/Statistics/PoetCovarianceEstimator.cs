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
/// Generic POET — Principal Orthogonal complEment Thresholding (Fan-Liao-Mincheva 2013).
/// Decomposes the sample covariance into a low-rank factor component
/// (top <c>K</c> principal components) plus a sparse residual covariance
/// obtained by soft-thresholding off-diagonal entries of the residual.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>.
/// </para>
/// <para>
/// Reference: Fan, J., Liao, Y. &amp; Mincheva, M. (2013). "Large covariance
/// estimation by thresholding principal orthogonal complements." Journal of
/// the Royal Statistical Society, Series B, 75(4), 603-680.
/// </para>
/// <para>
/// Algorithm:
/// <list type="number">
/// <item><description>Sample covariance <c>S</c> and its eigendecomposition. When the spectrum of
/// <c>S</c> (bounded by <c>N . max|S_ij|</c>) could leave the range of <typeparamref name="T"/> even though
/// every entry of <c>S</c> is representable, the decomposition runs on <c>S / max|S_ij|</c> and the
/// factor term is scaled back, so the estimate is representable whenever its entries are.</description></item>
/// <item><description>Low-rank factor component from top K eigenvalues.</description></item>
/// <item><description>Residual covariance <c>R = S - Sigma_F</c>.</description></item>
/// <item><description>Sparsify <c>R</c>: diagonal entries kept; off-diagonal entries soft-thresholded at the
/// adaptive, entry-specific level <c>tau_ij = C . sqrt(theta_ij) . omega_T</c> (FLM eq. 3.2), with
/// <c>omega_T = 1/sqrt(N) + sqrt(log N / T)</c> and <c>theta_ij</c> the variance over time of
/// <c>u_it . u_jt</c>, where <c>u_t</c> is the demeaned observation projected off the top-K
/// eigenvectors. Because <c>tau_ij</c> scales with the data, the estimator is scale-equivariant.
/// <c>theta_ij</c> is evaluated on residuals normalised by each column's own largest magnitude and
/// <c>tau_ij</c> is restored with the product of the two column scales, so a small column paired with a much
/// larger one keeps its own threshold.</description></item>
/// <item><description>Return <c>Sigma_hat = Sigma_F + R_tilde</c>.</description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class PoetCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : struct, IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);
    private static readonly SampleCovarianceEstimator<T> s_sampleEstimator = new();

    private readonly int _numFactors;
    private readonly T _thresholdMultiplier;

    /// <summary>Creates a one-factor POET estimator with a 0.5 threshold multiplier.</summary>
    public PoetCovarianceEstimator()
        : this(1, T.CreateChecked(0.5))
    {
    }

    /// <summary>
    /// Creates a POET estimator with the specified factor count and a 0.5 threshold multiplier.
    /// </summary>
    /// <param name="numFactors">
    /// Number of leading principal components retained as factors. Must be positive.
    /// </param>
    public PoetCovarianceEstimator(int numFactors)
        : this(numFactors, T.CreateChecked(0.5))
    {
    }

    /// <summary>
    /// Creates a POET estimator with the specified factor count and threshold multiplier.
    /// </summary>
    /// <param name="numFactors">
    /// Number of leading principal components retained as factors. Must be positive.
    /// </param>
    /// <param name="thresholdMultiplier">
    /// Multiplier <c>C</c> in the adaptive threshold
    /// <c>tau_ij = C . sqrt(theta_ij) . (1/sqrt(N) + sqrt(log N / T))</c>.
    /// Must be non-negative, finite, and representable in T; <see langword="null"/> (including
    /// the <see langword="default"/> literal) selects 0.5.
    /// </param>
    public PoetCovarianceEstimator(int numFactors = 1, T? thresholdMultiplier = null)
    {
        if (numFactors < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numFactors),
                numFactors,
                "Number of factors must be at least 1.");
        }

        var multiplier = thresholdMultiplier ?? T.CreateChecked(0.5);

        if (!T.IsFinite(multiplier) || multiplier < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(thresholdMultiplier),
                multiplier,
                "Threshold multiplier must be a non-negative number.");
        }

        _numFactors = numFactors;
        _thresholdMultiplier = multiplier;
    }

    /// <summary>Number of leading factors retained.</summary>
    public int NumFactors => _numFactors;

    /// <summary>Threshold multiplier used in the residual sparsification step.</summary>
    public T ThresholdMultiplier => _thresholdMultiplier;

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);
        if (typeof(T) == typeof(Half))
        {
            // Half cannot represent long-panel counts or their fourth-moment sums.
            // Widen the complete calculation exactly and narrow only the covariance.
            return CovarianceHelpers<T>.EstimateHalfShrinkage(
                returns, new PoetCovarianceEstimator<double>(_numFactors, double.CreateChecked(_thresholdMultiplier)));
        }

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        if (_numFactors >= n)
        {
            throw new ArgumentException(
                $"Number of factors ({_numFactors}) must be strictly less than number of assets ({n}).",
                nameof(returns));
        }

        var sampleCov = s_sampleEstimator.Estimate(returns);

        // The eigenvalues of S can reach tr(S) <= n . max|S_ij|, which can leave the range of T
        // although every entry of S, and of the POET estimate, is representable. In that case
        // decompose S / max|S_ij| (eigenvectors are scale-invariant) and scale the factor back.
        var spectralScale = SpectralScale(sampleCov, n);
        var spectralScaled = spectralScale != T.One;
        var spectralInput = sampleCov;
        if (spectralScaled)
        {
            spectralInput = new T[n, n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    spectralInput[i, j] = sampleCov[i, j] / spectralScale;
                }
            }
        }

        var eigen = JacobiEigenDecomposition<T>.Decompose(spectralInput);
        var values = eigen.Values;
        var vectors = eigen.Vectors;

        // Select the K largest eigenvalues.
        var indices = new int[n];
        for (var i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        Array.Sort(indices, (a, b) => values[b].CompareTo(values[a]));

        // Low-rank factor component Sigma_F = Sum lambda_k . v_k . v_k^T over the top K.
        var factor = new T[n, n];
        for (var f = 0; f < _numFactors; f++)
        {
            var idx = indices[f];
            var lambda = values[idx];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    factor[i, j] += lambda * vectors[i, idx] * vectors[j, idx];
                }
            }
        }

        if (spectralScaled)
        {
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    factor[i, j] *= spectralScale;
                }
            }
        }

        // Residual R = S - Sigma_F.
        var residual = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                residual[i, j] = sampleCov[i, j] - factor[i, j];
            }
        }

        // Soft-threshold off-diagonal residual entries at the adaptive,
        // entry-specific level tau_ij = C . sqrt(theta_ij) . omega_T (FLM eq. 3.2).
        var thresholds = ComputeAdaptiveThresholds(returns, vectors, indices, t, n);

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (i == j)
                {
                    continue;
                }

                var x = residual[i, j];
                var abs = T.Abs(x);
                var threshold = thresholds[i, j];
                if (abs <= threshold)
                {
                    residual[i, j] = T.Zero;
                }
                else
                {
                    residual[i, j] = x > T.Zero ? abs - threshold : -(abs - threshold);
                }
            }
        }

        // Sigma_hat = Sigma_F + R_tilde.
        var estimate = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                estimate[i, j] = factor[i, j] + residual[i, j];
            }
        }

        // Enforce exact symmetry. Halve before adding: both entries can lie above MaxValue / 2
        // (S itself can), and their sum would leave the range of T.
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var avg = (estimate[i, j] / s_two) + (estimate[j, i] / s_two);
                estimate[i, j] = avg;
                estimate[j, i] = avg;
            }
        }

        return estimate;
    }

    /// <summary>
    /// Returns the scale by which the sample covariance must be divided before its
    /// eigendecomposition, or <c>1</c> when the unscaled decomposition cannot leave the range of
    /// <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// Every eigenvalue is bounded by the trace, so <c>n . max|S_ij|</c> bounds the spectrum; the
    /// extra factor four covers the few-times-larger intermediate sums of a Jacobi rotation.
    /// Decimal overflow throws rather than returning an infinity.
    /// </remarks>
    private static T SpectralScale(T[,] covariance, int n)
    {
        var largest = T.Zero;
        foreach (var entry in covariance)
        {
            largest = T.Max(largest, T.Abs(entry));
        }

        try
        {
            if (T.IsFinite(largest * T.CreateChecked(n) * T.CreateChecked(4)))
            {
                return T.One;
            }
        }
        catch (OverflowException)
        {
            // Decimal signals overflow with an exception rather than infinity.
        }

        return largest;
    }

    /// <summary>
    /// Computes the FLM (2013, eq. 3.2) adaptive thresholds
    /// <c>tau_ij = C . sqrt(theta_ij) . omega_T</c> with
    /// <c>omega_T = 1/sqrt(N) + sqrt(log N / T)</c>,
    /// <c>theta_ij = (1/T) . Sum_t (u_it . u_jt - sigma_ij)^2</c> and
    /// <c>sigma_ij = (1/T) . Sum_t u_it . u_jt</c>, where <c>u_t</c> is the demeaned
    /// observation projected onto the orthogonal complement of the top-K eigenvectors.
    /// </summary>
    private T[,] ComputeAdaptiveThresholds(T[,] returns, T[,] vectors, int[] indices, int t, int n)
    {
        var thresholds = new T[n, n];
        if (n < 2 || _thresholdMultiplier == T.Zero)
        {
            return thresholds;
        }

        returns = CovarianceHelpers<T>.ShiftToAnchor(returns);
        var means = CovarianceHelpers<T>.ComputeMeans(returns);
        var residuals = new T[t, n];
        var centered = new T[n];
        var columnScale = new T[n];
        for (var s = 0; s < t; s++)
        {
            for (var j = 0; j < n; j++)
            {
                centered[j] = returns[s, j] - means[j];
                residuals[s, j] = centered[j];
            }

            for (var f = 0; f < _numFactors; f++)
            {
                var idx = indices[f];
                var projection = T.Zero;
                for (var j = 0; j < n; j++)
                {
                    projection += centered[j] * vectors[j, idx];
                }

                for (var j = 0; j < n; j++)
                {
                    residuals[s, j] -= projection * vectors[j, idx];
                }
            }

            for (var j = 0; j < n; j++)
            {
                columnScale[j] = T.Max(columnScale[j], T.Abs(residuals[s, j]));
            }
        }

        // theta_ij is fourth order in the data. Normalise each residual column by its own largest
        // magnitude so every product u_it . u_jt lies in [-1, 1] however the columns' scales differ,
        // then restore tau_ij from the two column scales. One shared scale (the global maximum)
        // would underflow theta_ij to zero for a small column paired with a much larger one and
        // switch that pair's threshold off. A column whose residual is identically zero keeps a
        // zero scale: every product with it, hence theta_ij and tau_ij, is zero.
        var anyResidual = false;
        for (var j = 0; j < n; j++)
        {
            if (columnScale[j] == T.Zero)
            {
                continue;
            }

            anyResidual = true;
            for (var s = 0; s < t; s++)
            {
                residuals[s, j] /= columnScale[j];
            }
        }

        if (!anyResidual)
        {
            return thresholds;
        }

        var tCount = T.CreateChecked(t);
        var omega = (T.One / NumericPrecision<T>.Sqrt(T.CreateChecked(n)))
                    + NumericPrecision<T>.Sqrt(NumericPrecision<T>.LogPositiveInteger(n) / tCount);
        // The paper uses 1/T for both residual covariance and its threshold.
        // Our covariance uses 1/(T-1), so the threshold needs the same conversion.
        var multiplier = _thresholdMultiplier * omega * (tCount / T.CreateChecked(t - 1));

        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var productSum = T.Zero;
                for (var s = 0; s < t; s++)
                {
                    productSum += residuals[s, i] * residuals[s, j];
                }

                var sigma = productSum / tCount;
                var deviationSum = T.Zero;
                for (var s = 0; s < t; s++)
                {
                    var deviation = (residuals[s, i] * residuals[s, j]) - sigma;
                    deviationSum += deviation * deviation;
                }

                var theta = deviationSum / tCount;
                T tau;
                try
                {
                    tau = CovarianceHelpers<T>.ScaleSafeProductOrdered(
                        multiplier, NumericPrecision<T>.Sqrt(theta), columnScale[i], columnScale[j]);
                }
                catch (OverflowException)
                {
                    // An unrepresentable positive decimal threshold removes every representable
                    // residual entry. Saturating has the same soft-threshold result as infinity.
                    tau = T.CreateSaturating(decimal.MaxValue);
                }
                thresholds[i, j] = tau;
                thresholds[j, i] = tau;
            }
        }

        return thresholds;
    }
}

/// <summary>
/// POET — Principal Orthogonal complEment Thresholding (Fan-Liao-Mincheva 2013).
/// Decomposes the sample covariance into a low-rank factor component
/// (top <c>K</c> principal components) plus a sparse residual covariance
/// obtained by soft-thresholding off-diagonal entries of the residual.
/// </summary>
/// <remarks>
/// <para>
/// Reference: Fan, J., Liao, Y. &amp; Mincheva, M. (2013). "Large covariance
/// estimation by thresholding principal orthogonal complements." Journal of
/// the Royal Statistical Society, Series B, 75(4), 603-680.
/// </para>
/// </remarks>
[Obsolete(
    "Use PoetCovarianceEstimator<decimal> instead. This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
public sealed class PoetCovarianceEstimator : ICovarianceEstimator
{
    private readonly PoetCovarianceEstimator<decimal> _inner;

    /// <summary>
    /// Creates a POET estimator with the specified factor count and threshold multiplier.
    /// </summary>
    /// <param name="numFactors">
    /// Number of leading principal components retained as factors. Must be
    /// positive and strictly less than <c>N</c>.
    /// </param>
    /// <param name="thresholdMultiplier">
    /// Multiplier <c>C</c> in the adaptive threshold
    /// <c>tau_ij = C . sqrt(theta_ij) . (1/sqrt(N) + sqrt(log N / T))</c>.
    /// Must be non-negative; defaults to 0.5.
    /// </param>
    public PoetCovarianceEstimator(int numFactors = 1, double thresholdMultiplier = 0.5)
    {
        if (thresholdMultiplier < 0.0 || !double.IsFinite(thresholdMultiplier))
        {
            throw new ArgumentOutOfRangeException(
                nameof(thresholdMultiplier),
                thresholdMultiplier,
                "Threshold multiplier must be a non-negative finite number.");
        }

        // The double representation of decimal.MaxValue rounds up to 2^96,
        // which itself cannot be converted back to decimal.
        if (thresholdMultiplier >= (double)decimal.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(thresholdMultiplier),
                thresholdMultiplier,
                "Threshold multiplier must fit in the decimal range used by this facade.");
        }

        _inner = new PoetCovarianceEstimator<decimal>(numFactors, (decimal)thresholdMultiplier);
    }

    /// <summary>Number of leading factors retained.</summary>
    public int NumFactors => _inner.NumFactors;

    /// <summary>Threshold multiplier used in the residual sparsification step.</summary>
    public double ThresholdMultiplier => (double)_inner.ThresholdMultiplier;

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
