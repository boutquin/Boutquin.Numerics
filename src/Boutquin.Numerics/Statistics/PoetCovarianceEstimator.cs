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
/// <item><description>Sample covariance <c>S</c> and its eigendecomposition.</description></item>
/// <item><description>Low-rank factor component from top K eigenvalues.</description></item>
/// <item><description>Residual covariance <c>R = S - Sigma_F</c>.</description></item>
/// <item><description>Sparsify <c>R</c>: diagonal entries kept; off-diagonal entries soft-thresholded.</description></item>
/// <item><description>Return <c>Sigma_hat = Sigma_F + R_tilde</c>.</description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class PoetCovarianceEstimator<T> : ICovarianceEstimator<T>
    where T : IFloatingPoint<T>
{
    private static readonly T s_two = T.CreateChecked(2);

    private readonly int _numFactors;
    private readonly T _thresholdMultiplier;

    /// <summary>
    /// Creates a POET estimator with the specified factor count and threshold multiplier.
    /// </summary>
    /// <param name="numFactors">
    /// Number of leading principal components retained as factors. Must be positive.
    /// </param>
    /// <param name="thresholdMultiplier">
    /// Multiplier <c>c</c> in the threshold <c>tau = c . sqrt(log N / T)</c>.
    /// Must be non-negative; defaults to 0.5.
    /// </param>
    public PoetCovarianceEstimator(int numFactors = 1, T? thresholdMultiplier = default)
    {
        if (numFactors < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numFactors),
                numFactors,
                "Number of factors must be at least 1.");
        }

        thresholdMultiplier ??= T.CreateChecked(0.5);

        if (thresholdMultiplier < T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(thresholdMultiplier),
                thresholdMultiplier,
                "Threshold multiplier must be a non-negative number.");
        }

        _numFactors = numFactors;
        _thresholdMultiplier = thresholdMultiplier;
    }

    /// <summary>Number of leading factors retained.</summary>
    public int NumFactors => _numFactors;

    /// <summary>Threshold multiplier used in the residual sparsification step.</summary>
    public T ThresholdMultiplier => _thresholdMultiplier;

    /// <inheritdoc />
    public T[,] Estimate(T[,] returns)
    {
        CovarianceHelpers<T>.ValidateReturns(returns);

        var t = returns.GetLength(0);
        var n = returns.GetLength(1);

        if (_numFactors >= n)
        {
            throw new ArgumentException(
                $"Number of factors ({_numFactors}) must be strictly less than number of assets ({n}).",
                nameof(returns));
        }

        var sampleCov = new SampleCovarianceEstimator<T>().Estimate(returns);

        var eigen = JacobiEigenDecomposition<T>.Decompose(sampleCov);
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

        // Residual R = S - Sigma_F.
        var residual = new T[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                residual[i, j] = sampleCov[i, j] - factor[i, j];
            }
        }

        // Soft-threshold off-diagonal residual entries with tau = c . sqrt(log N / T).
        var threshold = T.Zero;
        if (n > 1 && t > 0 && _thresholdMultiplier > T.Zero)
        {
            threshold = T.CreateChecked(
                double.CreateChecked(_thresholdMultiplier) * Math.Sqrt(Math.Log(n) / t));
        }

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

        // Enforce exact symmetry.
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var avg = (estimate[i, j] + estimate[j, i]) / s_two;
                estimate[i, j] = avg;
                estimate[j, i] = avg;
            }
        }

        return estimate;
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
    /// Multiplier <c>c</c> in the threshold <c>tau = c . sqrt(log N / T)</c>.
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

        _inner = new PoetCovarianceEstimator<decimal>(numFactors, (decimal)thresholdMultiplier);
    }

    /// <summary>Number of leading factors retained.</summary>
    public int NumFactors => _inner.NumFactors;

    /// <summary>Threshold multiplier used in the residual sparsification step.</summary>
    public double ThresholdMultiplier => (double)_inner.ThresholdMultiplier;

    /// <inheritdoc />
    public decimal[,] Estimate(decimal[,] returns) => _inner.Estimate(returns);
}
