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
using Boutquin.Numerics.LinearAlgebra.Internal;

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Generic ordinary least-squares estimator solving <c>min_beta ||y - X.beta||2^2</c> via
/// Householder QR decomposition of the design matrix. Delivers NIST-bar accuracy
/// (10 significant digits on coefficients, 8 digits on standard errors) on every problem
/// in the NIST StRD linear-regression suite, including the notoriously ill-conditioned
/// <c>Longley</c>, <c>Wampler4</c>, and <c>Wampler5</c> datasets.
/// </summary>
/// <remarks>
/// <para>
/// <b>Generic-math tier:</b> A+sqrt. Works for any <typeparamref name="T"/>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Square-root operations use
/// <see cref="NumericPrecision{T}.Sqrt"/> to support <c>decimal</c> (which lacks
/// <see cref="IRootFunctions{TSelf}"/>).
/// </para>
/// <para>
/// <b>Why QR and not the normal equations.</b> The classical closed form
/// <c>beta-hat = (XtX)^-1 . Xt.y</c> forms <c>XtX</c> explicitly, which squares the condition
/// number of the design matrix. For well-conditioned problems the two approaches agree,
/// but NIST StRD includes <c>Wampler5</c> (cond ~ 5e13) where the normal-equation
/// solver loses every digit of the response. Householder QR preserves the original
/// conditioning, so the estimator's accuracy floor is set by the problem, not by the
/// solver. References: Golub &amp; Van Loan, <i>Matrix Computations</i>, 4th ed., section 5.3;
/// Longley, J. W. (1967), <i>JASA</i> 62, 819-841; Wampler, R. H. (1970), <i>JASA</i>
/// 65, 549-565.
/// </para>
/// <para>
/// <b>Intercept handling.</b> When <c>includeIntercept</c> is
/// <see langword="true"/> (the default), a column of ones is prepended to <c>X</c>
/// before factorisation, and the resulting coefficient vector
/// has length <c>p + 1</c> with the intercept in index <c>0</c>. Set to
/// <see langword="false"/> only when the modelling assumption genuinely has no
/// intercept (NIST's <c>NoInt1</c>/<c>NoInt2</c> are two such cases).
/// </para>
/// <para>
/// <b>Mixed-precision refinement.</b> When <typeparamref name="T"/> is not
/// <see cref="decimal"/>, iterative refinement computes residuals in <c>decimal</c>
/// (~28 significant digits) to lift the stall floor from <c>kappa . u</c> to <c>u</c>.
/// When <typeparamref name="T"/> is <c>decimal</c>, the QR is already in 28-digit
/// precision and refinement is unnecessary — residuals are computed directly in <c>T</c>.
/// </para>
/// <para>
/// <b>Complexity.</b> <c>O(m . n^2)</c> flops for the QR, plus <c>O(n^3)</c> for the
/// covariance inversion. Memory: <c>O(m . n)</c> for the working copy of <c>X</c> and
/// <c>O(n^2)</c> for the covariance. Suitable for dense problems up to roughly
/// <c>n . p &lt; 10^6</c> entries; beyond that, use an iterative method.
/// </para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class OrdinaryLeastSquares<T>
    where T : IFloatingPoint<T>
{
    /// <summary>
    /// Fits <c>y = X . beta + epsilon</c> by minimising the residual sum of squares and returns
    /// the coefficient vector, classical standard errors, fit residuals, residual sum
    /// of squares, residual standard deviation, degrees of freedom, coefficient of
    /// determination, and the covariance matrix of the estimator.
    /// </summary>
    /// <param name="x">Design matrix, <c>m x p</c>, in row-major order. Not mutated.</param>
    /// <param name="y">Response vector, length <c>m</c>. Not mutated.</param>
    /// <param name="includeIntercept">
    /// When <see langword="true"/>, a column of ones is prepended to <paramref name="x"/>
    /// and the first returned coefficient is the intercept. Defaults to <see langword="true"/>.
    /// </param>
    /// <returns>
    /// An <see cref="OlsResult{T}"/> with coefficients, standard errors, residuals, RSS, sigma-hat,
    /// degrees of freedom, R-squared, and the covariance matrix of the estimator.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="y"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// Dimensions are inconsistent: <paramref name="x"/> has no observations, <paramref name="y"/>
    /// length differs from <paramref name="x"/>'s row count, or the design matrix (after
    /// optional intercept prepending) has more columns than observations.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The design matrix is numerically rank-deficient (a QR column is zero or R has a
    /// zero diagonal). OLS is undefined on a rank-deficient design; use a regularised
    /// estimator such as ridge regression instead.
    /// </exception>
    public static OlsResult<T> Fit(T[,] x, T[] y, bool includeIntercept = true)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        var m = x.GetLength(0);
        var p = x.GetLength(1);

        if (m == 0)
        {
            throw new ArgumentException("Design matrix must contain at least one observation.", nameof(x));
        }

        if (y.Length != m)
        {
            throw new ArgumentException(
                $"Response vector length {y.Length} does not match design-matrix row count {m}.",
                nameof(y));
        }

        // Working design matrix and response: copy the caller's inputs so Fit never mutates.
        // Prepend a column of ones when an intercept is requested.
        var designP = includeIntercept ? p + 1 : p;

        if (designP > m)
        {
            throw new ArgumentException(
                $"Design matrix has {designP} columns after intercept adjustment but only {m} observations; " +
                $"the system is under-determined and OLS is undefined.",
                nameof(x));
        }

        var design = new T[m, designP];
        var yWork = new T[m];
        Array.Copy(y, yWork, m);

        if (includeIntercept)
        {
            for (var i = 0; i < m; i++)
            {
                design[i, 0] = T.One;
                for (var j = 0; j < p; j++)
                {
                    design[i, j + 1] = x[i, j];
                }
            }
        }
        else
        {
            for (var i = 0; i < m; i++)
            {
                for (var j = 0; j < p; j++)
                {
                    design[i, j] = x[i, j];
                }
            }
        }

        var diagonal = new T[designP];
        var permutation = new int[designP];
        var factored = HouseholderQr<T>.Factor(design, yWork, diagonal, permutation);
        if (!factored)
        {
            throw new InvalidOperationException(
                "Design matrix is numerically rank-deficient; OLS is undefined. " +
                "Consider ridge regression or remove collinear columns before fitting.");
        }

        // beta-hat comes from back-substituting R . beta = c where c = Qty's leading designP entries.
        // SolveUpperTriangular un-permutes the solution so `coefficients` is in the caller's
        // original column order regardless of what column pivoting did internally.
        var coefficients = HouseholderQr<T>.SolveUpperTriangular(design, diagonal, yWork, designP, permutation);

        // Iterative refinement (Golub & Van Loan section 5.3.8): re-solve against the residual to
        // recover digits that catastrophic cancellation eats in the naive pipeline. When T is
        // decimal, the QR itself is 28-digit and refinement is unnecessary; otherwise, mixed-
        // precision refinement computes residuals in decimal to lift the stall floor.
        var residuals = new T[m];
        var rss = RefineAndComputeResiduals(x, y, coefficients, design, diagonal, permutation,
            residuals, includeIntercept, p, m, designP);

        var dof = m - designP;
        var sigmaSq = dof > 0 ? rss / T.CreateChecked(dof) : T.Zero;
        var sigma = NumericPrecision<T>.Sqrt(sigmaSq);

        // (XtX)^-1 via R^-1 . R^-t, then scale by sigma-hat-sq to get the estimator covariance.
        var xtxInverse = HouseholderQr<T>.BuildXtXInverse(design, diagonal, permutation);
        var covariance = new T[designP, designP];
        var standardErrors = new T[designP];
        for (var i = 0; i < designP; i++)
        {
            for (var j = 0; j < designP; j++)
            {
                covariance[i, j] = sigmaSq * xtxInverse[i, j];
            }

            // Diagonal entries can drift microscopically negative through accumulated
            // floating-point error on ill-conditioned designs; floor at zero before sqrt.
            standardErrors[i] = NumericPrecision<T>.Sqrt(T.Max(covariance[i, i], T.Zero));
        }

        // R-squared uses centred TSS when the model includes an intercept and uncentred TSS
        // otherwise (classical econometric and NIST convention). The centred form is
        // meaningful only when residuals are measured from the response mean; without
        // an intercept, the response mean is not a valid null model.
        T tss;
        if (includeIntercept)
        {
            var yMean = T.Zero;
            for (var i = 0; i < m; i++)
            {
                yMean += y[i];
            }

            yMean /= T.CreateChecked(m);

            tss = T.Zero;
            for (var i = 0; i < m; i++)
            {
                var d = y[i] - yMean;
                tss += d * d;
            }
        }
        else
        {
            tss = T.Zero;
            for (var i = 0; i < m; i++)
            {
                tss += y[i] * y[i];
            }
        }

        var rSquared = tss > T.Zero ? T.One - rss / tss : T.Zero;

        return new OlsResult<T>
        {
            Coefficients = coefficients,
            StandardErrors = standardErrors,
            Residuals = residuals,
            ResidualSumOfSquares = rss,
            ResidualStandardDeviation = sigma,
            DegreesOfFreedom = dof,
            RSquared = rSquared,
            CovarianceMatrix = covariance,
        };
    }

    private const int MaxRefinementIterations = 20;

    /// <summary>
    /// Applies iterative refinement to <paramref name="coefficients"/> (updated in-place),
    /// writes the final residuals into <paramref name="residuals"/>, and returns the
    /// residual sum of squares.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When <typeparamref name="T"/> is <see cref="decimal"/>, the QR is already 28-digit
    /// and no mixed-precision refinement is needed — residuals are computed directly in
    /// <typeparamref name="T"/>. For all other types, residuals are computed in
    /// <see cref="decimal"/> (~28 significant digits) and cast back to
    /// <typeparamref name="T"/> to lift the refinement stall floor from
    /// <c>kappa . u</c> to <c>u</c>.
    /// </para>
    /// </remarks>
    private static T RefineAndComputeResiduals(
        T[,] x,
        T[] y,
        T[] coefficients,
        T[,] qrFactored,
        T[] diagonal,
        int[] permutation,
        T[] residuals,
        bool includeIntercept,
        int p,
        int m,
        int designP)
    {
        // Iterative refinement (Golub & Van Loan §5.3.8): re-solve against the residual
        // to recover digits lost to catastrophic cancellation in the QR back-substitution.
        //
        // For T = decimal: refinement runs in T directly (already 28-digit precision).
        // For T = double/float/Half: refinement computes residuals in decimal (~28 digits)
        //   to lift the stall floor from κ·u to u. This is the same mixed-precision path
        //   the original OLS implementation used.
        var rhs = new T[m];

        // For non-decimal types, prepare decimal copies for mixed-precision residuals.
        decimal[,]? xDecimal = null;
        decimal[]? yDecimal = null;
        if (typeof(T) != typeof(decimal))
        {
            xDecimal = new decimal[m, p];
            yDecimal = new decimal[m];
            for (var i = 0; i < m; i++)
            {
                yDecimal[i] = decimal.CreateChecked(y[i]);
                for (var j = 0; j < p; j++)
                {
                    xDecimal[i, j] = decimal.CreateChecked(x[i, j]);
                }
            }
        }

        T rss;
        var bestCoefficients = (T[])coefficients.Clone();
        T bestRss;

        // Initial residual computation.
        if (typeof(T) == typeof(decimal))
        {
            bestRss = ComputeResidualsDirectlyWithRhs(x, y, coefficients, residuals, rhs, includeIntercept, p, m);
        }
        else
        {
            bestRss = ComputeResidualsDecimal(xDecimal!, yDecimal!, coefficients, residuals, rhs, includeIntercept, p, m);
        }

        rss = bestRss;
        var previousDeltaNormSq = rss + rss + T.One; // Sentinel larger than any initial RSS.

        for (var iter = 0; iter < MaxRefinementIterations; iter++)
        {
            // Solve for the correction δ via the stored QR: apply Qᵀ to rhs, back-substitute on R.
            HouseholderQr<T>.ApplyQTranspose(qrFactored, diagonal, rhs);
            var delta = HouseholderQr<T>.SolveUpperTriangular(qrFactored, diagonal, rhs, designP, permutation);

            // Apply the correction tentatively and measure its magnitude.
            var deltaNormSq = T.Zero;
            var betaNormSq = T.Zero;
            for (var k = 0; k < coefficients.Length; k++)
            {
                coefficients[k] += delta[k];
                deltaNormSq += delta[k] * delta[k];
                betaNormSq += coefficients[k] * coefficients[k];
            }

            // Recompute residuals with updated coefficients.
            if (typeof(T) == typeof(decimal))
            {
                rss = ComputeResidualsDirectlyWithRhs(x, y, coefficients, residuals, rhs, includeIntercept, p, m);
            }
            else
            {
                rss = ComputeResidualsDecimal(xDecimal!, yDecimal!, coefficients, residuals, rhs, includeIntercept, p, m);
            }

            if (rss < bestRss)
            {
                bestRss = rss;
                Array.Copy(coefficients, bestCoefficients, coefficients.Length);
            }

            // Converged: correction is at machine-epsilon floor; accept and stop.
            // Use 1e-26 (not 1e-30) because decimal can only represent down to ~1e-28;
            // 1e-30 truncates to 0m, making convergence impossible via this threshold.
            var convergenceThreshold = T.Max(betaNormSq, T.One) * T.CreateChecked(1e-26);
            if (deltaNormSq <= convergenceThreshold)
            {
                break;
            }

            // Stall: correction not shrinking AND RSS strictly worse than best.
            if (deltaNormSq >= previousDeltaNormSq && rss > bestRss)
            {
                Array.Copy(bestCoefficients, coefficients, coefficients.Length);
                if (typeof(T) == typeof(decimal))
                {
                    rss = ComputeResidualsDirectlyWithRhs(x, y, coefficients, residuals, rhs, includeIntercept, p, m);
                }
                else
                {
                    rss = ComputeResidualsDecimal(xDecimal!, yDecimal!, coefficients, residuals, rhs, includeIntercept, p, m);
                }

                break;
            }

            previousDeltaNormSq = deltaNormSq;
        }

        // Always return with β̂ set to the best iterate encountered.
        if (rss > bestRss)
        {
            Array.Copy(bestCoefficients, coefficients, coefficients.Length);
            if (typeof(T) == typeof(decimal))
            {
                rss = ComputeResidualsDirectlyWithRhs(x, y, coefficients, residuals, rhs, includeIntercept, p, m);
            }
            else
            {
                rss = ComputeResidualsDecimal(xDecimal!, yDecimal!, coefficients, residuals, rhs, includeIntercept, p, m);
            }
        }

        return rss;
    }

    /// <summary>
    /// Computes residuals directly in <typeparamref name="T"/>, writing both the
    /// <paramref name="residuals"/> and <paramref name="rhs"/> arrays for the iterative
    /// refinement loop. Used when <typeparamref name="T"/> is <see cref="decimal"/>
    /// (or any type where residuals should be computed natively in T).
    /// </summary>
    private static T ComputeResidualsDirectlyWithRhs(
        T[,] x,
        T[] y,
        T[] coefficients,
        T[] residuals,
        T[] rhs,
        bool includeIntercept,
        int p,
        int m)
    {
        var rss = T.Zero;
        for (var i = 0; i < m; i++)
        {
            var prediction = T.Zero;
            if (includeIntercept)
            {
                prediction = coefficients[0];
                for (var j = 0; j < p; j++)
                {
                    prediction += coefficients[j + 1] * x[i, j];
                }
            }
            else
            {
                for (var j = 0; j < p; j++)
                {
                    prediction += coefficients[j] * x[i, j];
                }
            }

            var r = y[i] - prediction;
            residuals[i] = r;
            rhs[i] = r;
            rss += r * r;
        }

        return rss;
    }

    /// <summary>
    /// Computes <c>r = y - X.beta-hat</c> in <see cref="decimal"/> precision, writes the
    /// <typeparamref name="T"/>-cast residuals to <paramref name="residuals"/> and the
    /// refinement RHS to <paramref name="rhs"/>, and returns the residual sum of squares
    /// in <typeparamref name="T"/>.
    /// </summary>
    private static T ComputeResidualsDecimal(
        decimal[,] xDecimal,
        decimal[] yDecimal,
        T[] coefficients,
        T[] residuals,
        T[] rhs,
        bool includeIntercept,
        int p,
        int m)
    {
        // Cast once per call. beta-hat mutates between calls, but the casts are cheap vs. the m.p
        // multiplies that follow.
        var betaDecimal = new decimal[coefficients.Length];
        for (var k = 0; k < coefficients.Length; k++)
        {
            betaDecimal[k] = decimal.CreateChecked(coefficients[k]);
        }

        var rss = 0.0m;
        for (var i = 0; i < m; i++)
        {
            var prediction = 0m;
            if (includeIntercept)
            {
                prediction = betaDecimal[0];
                for (var j = 0; j < p; j++)
                {
                    prediction += betaDecimal[j + 1] * xDecimal[i, j];
                }
            }
            else
            {
                for (var j = 0; j < p; j++)
                {
                    prediction += betaDecimal[j] * xDecimal[i, j];
                }
            }

            var rDec = yDecimal[i] - prediction;
            var rT = T.CreateChecked(rDec);
            residuals[i] = rT;
            rhs[i] = rT;
            rss += rDec * rDec;
        }

        return T.CreateChecked(rss);
    }
}

/// <summary>
/// Ordinary least-squares estimator solving <c>min_beta ||y - X.beta||2^2</c> via Householder
/// QR decomposition of the design matrix. Legacy facade delegating to
/// <see cref="OrdinaryLeastSquares{T}"/> instantiated at <c>T = double</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why QR and not the normal equations.</b> The classical closed form
/// <c>beta-hat = (XtX)^-1 . Xt.y</c> forms <c>XtX</c> explicitly, which squares the condition
/// number of the design matrix. For well-conditioned problems the two approaches agree,
/// but NIST StRD includes <c>Wampler5</c> (cond ~ 5e13) where the normal-equation
/// solver loses every digit of the response. Householder QR preserves the original
/// conditioning, so the estimator's accuracy floor is set by the problem, not by the
/// solver. References: Golub &amp; Van Loan, <i>Matrix Computations</i>, 4th ed., section 5.3;
/// Longley, J. W. (1967), <i>JASA</i> 62, 819-841; Wampler, R. H. (1970), <i>JASA</i>
/// 65, 549-565.
/// </para>
/// <para>
/// <b>Intercept handling.</b> When <c>includeIntercept</c> is
/// <see langword="true"/> (the default), a column of ones is prepended to <c>X</c>
/// before factorisation, and the resulting <see cref="OlsResult.Coefficients"/> vector
/// has length <c>p + 1</c> with the intercept in index <c>0</c>. Set to
/// <see langword="false"/> only when the modelling assumption genuinely has no
/// intercept (NIST's <c>NoInt1</c>/<c>NoInt2</c> are two such cases).
/// </para>
/// <para>
/// <b>Complexity.</b> <c>O(m . n^2)</c> flops for the QR, plus <c>O(n^3)</c> for the
/// covariance inversion. Memory: <c>O(m . n)</c> for the working copy of <c>X</c> and
/// <c>O(n^2)</c> for the covariance. Suitable for dense problems up to roughly
/// <c>n . p &lt; 10^6</c> entries; beyond that, use an iterative method.
/// </para>
/// </remarks>
public static class OrdinaryLeastSquares
{
    /// <summary>
    /// Fits <c>y = X . beta + epsilon</c> by minimising the residual sum of squares.
    /// Delegates to <see cref="OrdinaryLeastSquares{T}.Fit"/> at <c>T = double</c> and
    /// converts the result to the legacy <see cref="OlsResult"/> type.
    /// </summary>
    /// <param name="x">Design matrix, <c>m x p</c>, in row-major order. Not mutated.</param>
    /// <param name="y">Response vector, length <c>m</c>. Not mutated.</param>
    /// <param name="includeIntercept">
    /// When <see langword="true"/>, a column of ones is prepended to <paramref name="x"/>
    /// and the first returned coefficient is the intercept. Defaults to <see langword="true"/>.
    /// </param>
    /// <returns>
    /// An <see cref="OlsResult"/> with coefficients, standard errors, residuals, RSS, sigma-hat,
    /// degrees of freedom, R-squared, and the covariance matrix of the estimator.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="y"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// Dimensions are inconsistent: <paramref name="x"/> has no observations, <paramref name="y"/>
    /// length differs from <paramref name="x"/>'s row count, or the design matrix (after
    /// optional intercept prepending) has more columns than observations.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The design matrix is numerically rank-deficient (a QR column is zero or R has a
    /// zero diagonal). OLS is undefined on a rank-deficient design; use a regularised
    /// estimator such as ridge regression instead.
    /// </exception>
    public static OlsResult Fit(double[,] x, double[] y, bool includeIntercept = true)
    {
        var r = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept);
        return new OlsResult
        {
            Coefficients = r.Coefficients,
            StandardErrors = r.StandardErrors,
            Residuals = r.Residuals,
            ResidualSumOfSquares = r.ResidualSumOfSquares,
            ResidualStandardDeviation = r.ResidualStandardDeviation,
            DegreesOfFreedom = r.DegreesOfFreedom,
            RSquared = r.RSquared,
            CovarianceMatrix = r.CovarianceMatrix,
        };
    }
}
