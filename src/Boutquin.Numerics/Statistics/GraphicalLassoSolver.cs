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

// ─────────────────────────────────────────────────────────────────────────────
//  Supporting types
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// The result returned by <see cref="GraphicalLassoSolver{T}.Solve"/> and
/// <see cref="GraphicalLassoSolver{T}.SolveWithBic"/>.
/// </summary>
/// <typeparam name="T">IEEE 754 floating-point type (Tier B).</typeparam>
/// <param name="Precision">
/// The estimated precision (inverse-covariance) matrix Θ̂ ≻ 0.
/// </param>
/// <param name="Covariance">
/// The estimated covariance matrix Σ̂ = Θ̂⁻¹.
/// </param>
/// <param name="Lambda">The regularisation parameter λ used in this solution.</param>
/// <param name="Iterations">Total BCD iterations performed.</param>
/// <param name="Converged">
/// <c>true</c> when mean(abs(Wᵏ − Wᵏ⁻¹)) &lt; tolerance * mean(abs(off-diagonal S)) before the iteration cap
/// and every inner coordinate cycle in that sweep converged. Inner exhaustion
/// returns a non-converged result even if successive precision estimates repeat.
/// </param>
public sealed record GraphicalLassoResult<T>(
    T[,] Precision,
    T[,] Covariance,
    T Lambda,
    int Iterations,
    bool Converged)
    where T : struct, IFloatingPointIeee754<T>
{
    private T[,] _precision = CloneMatrix(Precision);
    private T[,] _covariance = CloneMatrix(Covariance);

    /// <summary>The estimated precision (inverse-covariance) matrix Θ̂ ≻ 0.</summary>
    public T[,] Precision
    {
        get => CloneMatrix(_precision);
        init => _precision = CloneMatrix(value);
    }

    /// <summary>The estimated covariance matrix Σ̂ = Θ̂⁻¹.</summary>
    public T[,] Covariance
    {
        get => CloneMatrix(_covariance);
        init => _covariance = CloneMatrix(value);
    }

    private static T[,] CloneMatrix(T[,] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return (T[,])source.Clone();
    }
}

/// <summary>
/// Options for <see cref="GraphicalLassoSolver{T}"/>.
/// </summary>
/// <typeparam name="T">IEEE 754 floating-point type (Tier B).</typeparam>
/// <param name="MaxIterations">
/// Maximum block-coordinate-descent iterations. Must be positive. Default 100.
/// </param>
/// <param name="Tolerance">
/// Relative mean absolute covariance-change convergence threshold. Must be finite and strictly positive.
/// Default 0.001. For diagonal S, exact stationarity is accepted as a library boundary rule.
/// </param>
/// <param name="Weights">
/// Optional p×p weight matrix for a weighted ℓ₁ penalty
/// λ · Wᵢⱼ · |θᵢⱼ|. The matrix must be finite, non-negative, and symmetric.
/// All entries, including the diagonal, participate in the penalty. Pass <c>null</c> for the uniform penalty.
/// </param>
public sealed record GraphicalLassoOptions<T>(
    int MaxIterations = 100,
    T? Tolerance = null,
    T[,]? Weights = null)
    where T : struct, IFloatingPointIeee754<T>
{
    private T[,]? _weights = Weights is null ? null : CloneMatrix(Weights);

    /// <summary>Optional p×p weight matrix for a weighted ℓ₁ penalty.</summary>
    public T[,]? Weights
    {
        get => _weights is null ? null : CloneMatrix(_weights);
        init => _weights = value is null ? null : CloneMatrix(value);
    }

    private static T[,] CloneMatrix(T[,] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return (T[,])source.Clone();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Solver
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// ℓ₁-penalised inverse-covariance (graphical lasso) estimator.
/// Solves min_Θ [ tr(SΘ) − log det Θ + λ‖Θ‖₁ ] via the block-coordinate-descent
/// algorithm of Friedman, Hastie &amp; Tibshirani (2008).
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> B — the objective involves log det Θ (T.Log).
/// Works for <c>double</c>, <c>float</c>, and <c>Half</c>. Does NOT work for
/// <c>decimal</c> (no <c>IFloatingPointIeee754&lt;decimal&gt;</c>).
/// </para>
/// <para>
/// Reference: Friedman, J., Hastie, T., &amp; Tibshirani, R. (2008).
/// "Sparse inverse covariance estimation with the graphical lasso."
/// Biostatistics, 9(3), 432-441.
/// </para>
/// <para>
/// <b>Algorithm (BCD):</b> For each column j, the lasso sub-problem
/// min_β ½ β^T W₁₁ β − s₁₂^T β + λ‖β‖₁ is solved by coordinate descent using
/// soft-thresholding: β_i ← Soft(s₁₂ᵢ − W₁₁,−ᵢ·β_{−ᵢ}, λ) / W₁₁,ᵢᵢ.
/// The final coefficients give the sparse precision through equations (13)–(14).
/// Its inverse is computed by Cholesky so the returned covariance and precision agree.
/// </para>
/// <para>
/// Every returned precision is positive definite. A sweep that loses definiteness
/// returns the previous positive-definite iterate with <c>Converged=false</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">IEEE 754 floating-point type (<see cref="IFloatingPointIeee754{TSelf}"/>).</typeparam>
public static class GraphicalLassoSolver<T>
    where T : struct, IFloatingPointIeee754<T>
{
    private static readonly T s_defaultTolerance = T.CreateChecked(0.001);
    private static readonly T s_symmetryTolerance = T.CreateChecked(1e-8);

    // ─────────────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Fits the graphical lasso at a fixed regularisation parameter <paramref name="lambda"/>.
    /// </summary>
    /// <param name="sampleCovariance">
    /// p×p symmetric positive-definite sample covariance matrix S.
    /// </param>
    /// <param name="lambda">
    /// ℓ₁ penalty strength λ ≥ 0. At λ = 0, Θ̂ = S⁻¹ (no shrinkage).
    /// </param>
    /// <param name="options">Algorithm options; <c>null</c> uses defaults.</param>
    /// <returns>
    /// A <see cref="GraphicalLassoResult{T}"/> containing the estimated precision
    /// matrix Θ̂ ≻ 0, the covariance Σ̂ = Θ̂⁻¹, the λ used, the iteration count,
    /// and a convergence flag.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="sampleCovariance"/> is non-finite, not square,
    /// not symmetric, or not positive definite; or when the optional weights matrix
    /// has invalid dimensions or entries.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="lambda"/> is non-finite or negative, or when an
    /// iteration limit or tolerance is invalid.
    /// </exception>
    public static GraphicalLassoResult<T> Solve(
        T[,] sampleCovariance,
        T lambda,
        GraphicalLassoOptions<T>? options = null)
    {
        ArgumentNullException.ThrowIfNull(sampleCovariance);

        if (!T.IsFinite(lambda) || lambda < T.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda),
                lambda, "Lambda must be finite and non-negative.");
        }

        var p = sampleCovariance.GetLength(0);
        if (sampleCovariance.GetLength(1) != p)
        {
            throw new ArgumentException(
                "Sample covariance matrix must be square.", nameof(sampleCovariance));
        }

        if (p < 1)
        {
            throw new ArgumentException(
                "Sample covariance matrix must not be empty (p must be ≥ 1).",
                nameof(sampleCovariance));
        }

        ValidateFinite(sampleCovariance, p, nameof(sampleCovariance));
        ValidateSymmetric(sampleCovariance, p, nameof(sampleCovariance));

        ValidatePositiveDefinite(sampleCovariance, nameof(sampleCovariance));

        options ??= new GraphicalLassoOptions<T>();
        var tol = options.Tolerance ?? s_defaultTolerance;
        var maxIter = options.MaxIterations;
        var weights = options.Weights;
        if (!T.IsFinite(tol))
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), tol, "Options tolerance must be finite.");
        }

        if (tol <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), tol, "Options tolerance must be strictly positive.");
        }
        if (maxIter < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), maxIter, "MaxIterations must be positive.");
        }

        if (weights != null && (weights.GetLength(0) != p || weights.GetLength(1) != p))
        {
            throw new ArgumentException(
                $"Weights matrix must be p×p ({p}×{p}) to match the covariance matrix dimension.",
                nameof(options));
        }

        if (weights != null)
        {
            ValidateWeights(weights, p, nameof(options));
        }

        return SolveCore(sampleCovariance, lambda, p, tol, maxIter, weights);
    }

    /// <summary>
    /// Fits the graphical lasso across a grid of λ values and selects the one
    /// minimising BIC = −2·ℓ̂ + k(λ)·ln(n), where k(λ) is the number of non-zero
    /// off-diagonal entries in the estimated precision matrix. Entries whose absolute
    /// partial correlation is at most 1e-10 are treated as numerical zeros.
    /// </summary>
    /// <param name="sampleCovariance">p×p symmetric positive-definite sample covariance S.</param>
    /// <param name="nObservations">Number of observations n used to compute S.</param>
    /// <param name="lambdaGrid">Grid of candidate λ values (must be non-empty, all ≥ 0).</param>
    /// <param name="options">Algorithm options applied to every grid point.</param>
    /// <returns>
    /// The <see cref="GraphicalLassoResult{T}"/> at the BIC-optimal λ.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="sampleCovariance"/> is non-finite, not square,
    /// not symmetric, or not positive definite; when <paramref name="lambdaGrid"/>
    /// is empty; or when the optional weights matrix has invalid dimensions or entries.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="nObservations"/> is not positive, a lambda-grid
    /// entry is non-finite or negative, or an iteration limit or tolerance is invalid.
    /// </exception>
    /// <exception cref="InvalidOperationException">No grid candidate converged with a finite BIC.</exception>
    public static GraphicalLassoResult<T> SolveWithBic(
        T[,] sampleCovariance,
        int nObservations,
        T[] lambdaGrid,
        GraphicalLassoOptions<T>? options = null)
    {
        ArgumentNullException.ThrowIfNull(sampleCovariance);
        ArgumentNullException.ThrowIfNull(lambdaGrid);

        if (lambdaGrid.Length == 0)
        {
            throw new ArgumentException("Lambda grid must be non-empty.", nameof(lambdaGrid));
        }

        var p = sampleCovariance.GetLength(0);
        if (sampleCovariance.GetLength(1) != p)
        {
            throw new ArgumentException(
                "Sample covariance matrix must be square.", nameof(sampleCovariance));
        }

        if (p < 1)
        {
            throw new ArgumentException(
                "Sample covariance matrix must not be empty (p must be ≥ 1).",
                nameof(sampleCovariance));
        }

        ValidateFinite(sampleCovariance, p, nameof(sampleCovariance));
        ValidateSymmetric(sampleCovariance, p, nameof(sampleCovariance));

        if (nObservations <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(nObservations),
                nObservations,
                "Number of observations must be positive.");
        }

        ValidatePositiveDefinite(sampleCovariance, nameof(sampleCovariance));

        options ??= new GraphicalLassoOptions<T>();
        var tol = options.Tolerance ?? s_defaultTolerance;
        var maxIter = options.MaxIterations;
        var weights = options.Weights;
        if (!T.IsFinite(tol))
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), tol, "Options tolerance must be finite.");
        }

        if (tol <= T.Zero)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), tol, "Options tolerance must be strictly positive.");
        }
        if (maxIter < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), maxIter, "MaxIterations must be positive.");
        }

        if (weights != null && (weights.GetLength(0) != p || weights.GetLength(1) != p))
        {
            throw new ArgumentException(
                $"Weights matrix must be p×p ({p}×{p}) to match the covariance matrix dimension.",
                nameof(options));
        }

        if (weights != null)
        {
            ValidateWeights(weights, p, nameof(options));
        }

        for (var i = 0; i < lambdaGrid.Length; i++)
        {
            var candidate = lambdaGrid[i];
            if (!T.IsFinite(candidate) || candidate < T.Zero)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(lambdaGrid), candidate,
                    $"Lambda grid entry at index {i} must be finite and non-negative.");
            }
        }

        // Score the reduced likelihood in double: a valid int observation count
        // can exceed Half's range. The fitted matrices remain in T.
        var logN = Math.Log(nObservations);
        var bestBic = double.PositiveInfinity;
        GraphicalLassoResult<T>? bestResult = null;

        foreach (var lam in lambdaGrid)
        {
            var result = SolveCore(sampleCovariance, lam, p, tol, maxIter, weights);
            if (!result.Converged)
            {
                continue;
            }

            var precision = result.Precision;

            // BIC = −2·ℓ̂ + k·ln(n), with the Gaussian log-likelihood
            //   ℓ̂(Θ) = (n/2)·(log det Θ − tr(S·Θ)) up to a constant. Friedman-Hastie-
            //   Tibshirani (2008) maximize log det Θ − tr(S·Θ) without the n/2 factor and
            //   define no BIC; this selection rule is a library extension.
            // The n/2 factor is essential: dropping it under-weights the data fit by a
            // factor of n relative to the k·ln(n) penalty, which systematically
            // over-sparsifies the selected λ.
            var logDet = LogDet(precision, p);
            var trSTheta = TraceProduct(sampleCovariance, precision, p);
            var logLik = nObservations / 2.0
                * (double.CreateChecked(logDet) - double.CreateChecked(trSTheta));

            // k = number of non-zero off-diagonal precision entries (count unique pairs × 2)
            long k = 0;
            for (var i = 0; i < p; i++)
            {
                for (var j = 0; j < p; j++)
                {
                    var edgeTolerance = T.CreateChecked(1e-10) * T.Sqrt(precision[i, i]) * T.Sqrt(precision[j, j]);
                    if (i != j && T.Abs(precision[i, j]) > edgeTolerance)
                    {
                        k++;
                    }
                }
            }

            var bic = (-2.0 * logLik) + (k * logN);
            if (bic < bestBic)
            {
                bestBic = bic;
                bestResult = result;
            }
        }

        return bestResult
            ?? throw new InvalidOperationException(
                "No lambda in the grid converged with a finite BIC result.");
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Core BCD algorithm
    // ─────────────────────────────────────────────────────────────────────

    private static GraphicalLassoResult<T> SolveCore(
        T[,] s, T lambda, int p, T tol, int maxIter, T[,]? weights)
    {
        // With no penalty, S itself is the exact solution. Inexact regressions can only
        // move it away from that solution, especially for ill-conditioned input.
        if (lambda == T.Zero)
        {
            return new GraphicalLassoResult<T>(InvertSpd(s, p), s, lambda, 1, true);
        }

        // Exact zeros split independent blocks. Solve them in their own units so a
        // large penalty or variance in another block cannot erase a small variance.
        var visited = new bool[p];
        var blocks = new List<List<int>>();
        for (var start = 0; start < p; start++)
        {
            if (visited[start])
            {
                continue;
            }

            var block = new List<int> { start };
            visited[start] = true;
            for (var cursor = 0; cursor < block.Count; cursor++)
            {
                for (var j = 0; j < p; j++)
                {
                    if (!visited[j] && s[block[cursor], j] != T.Zero)
                    {
                        visited[j] = true;
                        block.Add(j);
                    }
                }
            }

            blocks.Add(block);
        }

        if (blocks.Count > 1)
        {
            var blockCovariance = new T[p, p];
            var blockPrecision = new T[p, p];
            var iterations = 0;
            var converged = true;
            foreach (var block in blocks)
            {
                var size = block.Count;
                var sub = new T[size, size];
                var subWeights = new T[size, size];
                for (var i = 0; i < size; i++)
                {
                    for (var j = 0; j < size; j++)
                    {
                        sub[i, j] = s[block[i], block[j]];
                        subWeights[i, j] = weights?[block[i], block[j]] ?? T.One;
                    }
                }

                var result = SolveCore(sub, lambda, size, tol, maxIter, subWeights);
                var subCovariance = result.Covariance;
                var subPrecision = result.Precision;
                iterations = Math.Max(iterations, result.Iterations);
                converged &= result.Converged;
                for (var i = 0; i < size; i++)
                {
                    for (var j = 0; j < size; j++)
                    {
                        blockCovariance[block[i], block[j]] = subCovariance[i, j];
                        blockPrecision[block[i], block[j]] = subPrecision[i, j];
                    }
                }
            }

            return new GraphicalLassoResult<T>(blockPrecision, blockCovariance, lambda, iterations, converged);
        }

        // The problem is scale-equivariant: for S -> cS and lambda -> c lambda the solution is
        // W -> cW and Theta -> Theta / c. W is as large as S, and the coordinate-descent products
        // W11 * beta carry coefficients above one, so a panel whose entries sit within a factor
        // of a few of the type's maximum overflows although its W and Theta are representable.
        // Solve in unit-magnitude coordinates, scaling by an even power of two so that every
        // rounding (including each square root of the Cholesky factor) is unchanged, and restore
        // the result. A panel whose largest entry lies in [1, 4) takes the unscaled path as before.
        var exponent = EvenScaleExponent(s, lambda, p, weights);
        if (exponent == 0)
        {
            return SolveUnscaled(s, lambda, p, tol, maxIter, weights);
        }

        var scaledS = new T[p, p];
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                scaledS[i, j] = T.ScaleB(s[i, j], -exponent);
            }
        }

        var scaled = SolveUnscaled(scaledS, T.ScaleB(lambda, -exponent), p, tol, maxIter, weights);
        var precision = scaled.Precision;
        var covariance = scaled.Covariance;
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                covariance[i, j] = T.ScaleB(covariance[i, j], exponent);
                precision[i, j] = T.ScaleB(precision[i, j], -exponent);
            }
        }

        return new GraphicalLassoResult<T>(precision, covariance, lambda, scaled.Iterations, scaled.Converged);
    }

    /// <summary>
    /// Even binary exponent of the largest magnitude that enters the initial W: an entry of S
    /// or the largest diagonal penalty lambda times a weight. Zero when that magnitude lies in [1, 4).
    /// </summary>
    private static int EvenScaleExponent(T[,] s, T lambda, int p, T[,]? weights)
    {
        var exponent = T.ILogB(MaxAbs(s, p));
        if (lambda > T.Zero)
        {
            var weightScale = T.One;
            if (weights is not null)
            {
                weightScale = T.Zero;
                for (var i = 0; i < p; i++)
                {
                    weightScale = T.Max(weightScale, weights[i, i]);
                }
            }
            if (weightScale > T.Zero)
            {
                exponent = Math.Max(exponent, T.ILogB(lambda) + T.ILogB(weightScale));
            }
        }

        // Keep the exponent even: an odd shift would change the rounding of each Cholesky square root.
        exponent -= exponent & 1;
        return exponent;
    }

    private static GraphicalLassoResult<T> SolveUnscaled(
        T[,] s, T lambda, int p, T tol, int maxIter, T[,]? weights)
    {
        // FHT: initialize W=S+lambda*diag(weights), retaining this diagonal.
        var w = new T[p, p];
        var offDiagonalScale = T.Zero;
        var offDiagonalCount = 0L;
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                w[i, j] = s[i, j] + (i == j ? lambda * (weights is null ? T.One : weights[i, i]) : T.Zero);
                if (i != j)
                {
                    // A running mean stays bounded by the largest entry; the raw
                    // sum and the product p*(p-1) need not fit the working type.
                    offDiagonalCount++;
                    offDiagonalScale += (T.Abs(s[i, j]) - offDiagonalScale)
                        * T.CreateChecked(1.0 / offDiagonalCount);
                }
            }
        }

        // β array for the lasso sub-problem (size p-1 per column).
        var beta = new T[p - 1];

        var theta = InvertSpd(w, p);

        var previousW = new T[p, p];
        var iter = 0;
        var w11 = new T[p - 1, p - 1];
        var s12 = new T[p - 1];
        var sparsePrecision = new T[p, p];

        while (iter < maxIter)
        {
            iter++;
            var innerConverged = true;
            Array.Copy(w, previousW, p * p);

            // BCD sweep: cycle through columns j = 0..p-1.
            for (var j = 0; j < p; j++)
            {
                // Build W_{11} (the (p-1)×(p-1) sub-matrix excluding row/col j)
                // and s_{12} (column j of S, excluding entry j).

                var ri = 0;
                for (var r = 0; r < p; r++)
                {
                    if (r == j)
                    {
                        continue;
                    }

                    var ci = 0;
                    for (var c = 0; c < p; c++)
                    {
                        if (c == j)
                        {
                            continue;
                        }

                        w11[ri, ci] = w[r, c];
                        ci++;
                    }

                    s12[ri] = s[r, j];
                    ri++;
                }

                // Solve the lasso sub-problem via coordinate descent:
                //   min_β ½ β^T W_{11} β - s_{12}^T β + λ ‖β‖₁
                // Cold-start β = 0; a finite inner budget may be exhausted.
                Array.Clear(beta, 0, p - 1);

                innerConverged &= SolveLassoSubproblem(w11, s12, beta, lambda, j, p, weights);
                if (beta.Any(value => !T.IsFinite(value)))
                {
                    return new GraphicalLassoResult<T>(theta, previousW, lambda, iter, false);
                }

                // Update column/row j of W: w_{12} = W_{11} · β
                ri = 0;
                for (var r = 0; r < p; r++)
                {
                    if (r == j)
                    {
                        continue;
                    }

                    T val = T.Zero;
                    for (var k = 0; k < p - 1; k++)
                    {
                        val += w11[ri, k] * beta[k];
                    }

                    w[r, j] = val;
                    w[j, r] = val;
                    ri++;
                }

                // FHT (13)–(14). Preserve the lasso's exact zero coefficients.
                var schur = w[j, j];
                ri = 0;
                for (var r = 0; r < p; r++)
                {
                    if (r != j)
                    {
                        schur -= w[r, j] * beta[ri++];
                    }
                }

                sparsePrecision[j, j] = T.One / schur;
                ri = 0;
                for (var r = 0; r < p; r++)
                {
                    if (r != j)
                    {
                        sparsePrecision[r, j] = -beta[ri++] * sparsePrecision[j, j];
                    }
                }
            }

            // Recompute Θ = W^{-1} after this sweep.
            try
            {
                theta = InvertSpd(w, p);
            }
            catch (InvalidOperationException)
            {
                return new GraphicalLassoResult<T>(theta, previousW, lambda, iter, false);
            }

            var meanChange = T.Zero;
            for (var i = 0; i < p; i++)
            {
                for (var j = 0; j < p; j++)
                {
                    meanChange += T.Abs(w[i, j] - previousW[i, j]);
                }
            }

            // Divide by p twice: p * p leaves Half's range at p = 256 and the quotient would collapse to zero.
            meanChange = meanChange / T.CreateChecked(p) / T.CreateChecked(p);
            // The primary strict criterion has no solution at zero scale; require exact stationarity there.
            if (offDiagonalScale == T.Zero
                ? meanChange == T.Zero
                : meanChange < tol * offDiagonalScale)
            {
                // The outer iterate meets its requested tolerance. Stop, while reporting
                // that an exhausted inner solve still prevents a convergence claim.
                if (!innerConverged)
                {
                    return new GraphicalLassoResult<T>(theta, w, lambda, iter, false);
                }

                for (var i = 0; i < p; i++)
                {
                    for (var j = i + 1; j < p; j++)
                    {
                        // A zero in either final regression represents an absent edge.
                        var entry = sparsePrecision[i, j] == T.Zero || sparsePrecision[j, i] == T.Zero
                            ? T.Zero : sparsePrecision[i, j] / T.CreateChecked(2) + sparsePrecision[j, i] / T.CreateChecked(2);
                        sparsePrecision[i, j] = sparsePrecision[j, i] = entry;
                    }
                }

                try
                {
                    return new GraphicalLassoResult<T>(sparsePrecision, InvertSpd(sparsePrecision, p), lambda, iter, true);
                }
                catch (InvalidOperationException)
                {
                    return new GraphicalLassoResult<T>(theta, w, lambda, iter, false);
                }
            }

        }

        // Did not converge — return the last computed Θ. W is unchanged since Θ was formed, so
        // re-inverting would reproduce the same matrix bit-for-bit.
        var finalCov = new T[p, p];
        Array.Copy(w, finalCov, p * p);

        return new GraphicalLassoResult<T>(theta, finalCov, lambda, iter, false);
    }

    private static void ValidatePositiveDefinite(T[,] matrix, string paramName)
    {
        try
        {
            _ = CholeskyDecomposition<T>.Decompose(matrix);
        }
        catch (InvalidOperationException ex)
        {
            throw new ArgumentException(
                "Sample covariance matrix must be positive definite.", paramName, ex);
        }
    }

    private static void ValidateWeights(T[,] weights, int p, string paramName)
    {
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                var weight = weights[i, j];
                if (!T.IsFinite(weight) || weight < T.Zero)
                {
                    throw new ArgumentException(
                        $"Weight entry [{i}, {j}] must be finite and non-negative.",
                        paramName);
                }
            }

            for (var j = i + 1; j < p; j++)
            {
                if (weights[i, j] != weights[j, i])
                {
                    throw new ArgumentException(
                        $"Weights matrix must be symmetric; entries [{i}, {j}] and " +
                        $"[{j}, {i}] differ.",
                        paramName);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Lasso sub-problem: coordinate descent with soft-thresholding
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Solves min_β ½ β^T W_{11} β − s_{12}^T β + λ‖β‖₁
    /// in place, updating <paramref name="beta"/>.
    /// </summary>
    private static bool SolveLassoSubproblem(
        T[,] w11, T[] s12, T[] beta, T lambda, int col, int p, T[,]? weights)
    {
        var m = p - 1;
        const int MaxLassoIter = 1000;
        var tol = T.CreateChecked(1e-8);

        for (var iter = 0; iter < MaxLassoIter; iter++)
        {
            var maxChange = T.Zero;

            for (var i = 0; i < m; i++)
            {
                // Partial residual: r_i = s12[i] - W11_{-i} · β_{-i}
                var r = s12[i];
                for (var k = 0; k < m; k++)
                {
                    if (k != i)
                    {
                        r -= w11[i, k] * beta[k];
                    }
                }

                // Effective λ for this entry (weighted or uniform)
                T effectiveLambda;
                if (weights != null)
                {
                    // Map beta index i (excluding column col) to the original matrix indices.
                    var origRow = i >= col ? i + 1 : i;
                    effectiveLambda = lambda * weights[origRow, col];
                }
                else
                {
                    effectiveLambda = lambda;
                }

                // β_i ← Soft(r, λ) / W11[i,i]
                var oldBeta = beta[i];
                beta[i] = SoftThreshold(r, effectiveLambda) / w11[i, i];
                var change = T.Abs(beta[i] - oldBeta);
                if (change > maxChange)
                {
                    maxChange = change;
                }
            }

            // The requested inner tolerance rounds to zero in Half. Exact
            // stationarity still converges; exhausted moving cycles do not.
            if (maxChange == T.Zero || maxChange < tol)
            {
                return true;
            }
        }

        return false;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Soft-thresholding operator: sign(x) · max(|x| − λ, 0).
    /// </summary>
    private static T SoftThreshold(T x, T lambda)
    {
        var absX = T.Abs(x);
        if (absX <= lambda)
        {
            return T.Zero;
        }

        return T.CopySign(absX - lambda, x);
    }

    /// <summary>
    /// Inverts a symmetric positive-definite p×p matrix using Cholesky factorisation.
    /// </summary>
    private static T[,] InvertSpd(T[,] a, int p)
    {
        var lower = CholeskyDecomposition<T>.Decompose(a);
        var inv = new T[p, p];
        var e = new T[p];

        for (var j = 0; j < p; j++)
        {
            for (var k = 0; k < p; k++)
            {
                e[k] = T.Zero;
            }

            e[j] = T.One;
            var col = CholeskyDecomposition<T>.Solve(lower, e);
            for (var i = 0; i < p; i++)
            {
                inv[i, j] = col[i];
            }
        }

        for (var i = 0; i < p; i++)
        {
            for (var j = i + 1; j < p; j++)
            {
                inv[i, j] = inv[j, i] = inv[i, j] / T.CreateChecked(2) + inv[j, i] / T.CreateChecked(2);
            }
        }
        return inv;
    }

    /// <summary>
    /// Computes log det A for a p×p SPD matrix via Cholesky: log det A = 2 Σ log L_{ii}.
    /// </summary>
    private static T LogDet(T[,] a, int p)
    {
        try
        {
            var lower = CholeskyDecomposition<T>.Decompose(a);
            var logDet = T.Zero;
            for (var i = 0; i < p; i++)
            {
                logDet += T.Log(lower[i, i]);
            }

            return T.CreateChecked(2) * logDet;
        }
        catch (InvalidOperationException)
        {
            // Cholesky throws InvalidOperationException for a non-positive-definite
            // iterate; treat log det as −∞ so the corresponding λ yields BIC = +∞ and
            // is never selected. Programmer errors (IndexOutOfRangeException, OOM, …)
            // are NOT swallowed — they propagate.
            return T.NegativeInfinity;
        }
    }

    /// <summary>
    /// Computes tr(A · B) = Σᵢⱼ Aᵢⱼ · Bᵢⱼ (for symmetric matrices).
    /// </summary>
    private static T TraceProduct(T[,] a, T[,] b, int p)
    {
        var sum = T.Zero;
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                sum += a[i, j] * b[i, j];
            }
        }

        return sum;
    }

    /// <summary>
    /// Validates that every entry of <paramref name="m"/> is finite (no NaN or ±∞).
    /// A single non-finite entry otherwise poisons the BCD sweep, the Cholesky
    /// inversion, and the BIC criterion, silently producing a NaN-corrupted result.
    /// </summary>
    private static void ValidateFinite(T[,] m, int p, string paramName)
    {
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                if (!T.IsFinite(m[i, j]))
                {
                    throw new ArgumentException(
                        $"Sample covariance entry [{i}, {j}] is not finite (value {m[i, j]}).",
                        paramName);
                }
            }
        }
    }

    /// <summary>
    /// Validates that <paramref name="m"/> is symmetric (within tolerance).
    /// </summary>
    private static void ValidateSymmetric(T[,] m, int p, string paramName)
    {
        for (var i = 0; i < p; i++)
        {
            for (var j = i + 1; j < p; j++)
            {
                var diff = T.Abs(m[i, j] - m[j, i]);
                var scale = T.Max(T.Abs(m[i, j]), T.Abs(m[j, i]));
                // Keep the rounding allowance local to this entry's magnitude. T.Epsilon
                // is the smallest subnormal, not a relative machine epsilon.
                var ulp = scale - T.BitDecrement(scale);
                var tolerance = T.Max(ulp * T.CreateChecked(2), s_symmetryTolerance * scale);
                if (diff > tolerance)
                {
                    throw new ArgumentException(
                        $"Sample covariance matrix must be symmetric. " +
                        $"Entries [{i},{j}]={m[i, j]:G6} and [{j},{i}]={m[j, i]:G6} differ by {diff:G6}.",
                        paramName);
                }
            }
        }
    }

    private static T MaxAbs(T[,] matrix, int p)
    {
        var max = T.Zero;
        for (var i = 0; i < p; i++)
        {
            for (var j = 0; j < p; j++)
            {
                var value = T.Abs(matrix[i, j]);
                if (value > max)
                {
                    max = value;
                }
            }
        }

        return max;
    }
}
