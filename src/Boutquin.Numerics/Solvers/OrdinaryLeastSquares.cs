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
/// Householder QR decomposition of the design matrix. On the NIST StRD linear-regression
/// suite, <c>T = double</c> matches the certified coefficients to 1e-12 relative and the
/// standard errors to 1e-11, including the ill-conditioned <c>Longley</c>,
/// <c>Wampler4</c>, and <c>Wampler5</c> datasets, on every problem except <c>Filip</c>
/// (degree-10 polynomial, cond ≈ 10¹⁰), which holds 1e-7 on coefficients and 1e-6 on
/// standard errors.
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
/// but NIST's degree-five Wampler design has condition number about 6.4e6.
/// Wampler4 and Wampler5 use that same design with different response noise.
/// Householder QR avoids squaring the design condition number; arithmetic precision,
/// residual magnitude and input rounding still affect attainable coefficient accuracy. References: Golub &amp; Van Loan, <i>Matrix Computations</i>, 4th ed., section 5.3;
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
/// <b>Range.</b> A response or design column whose Euclidean norm exceeds the range of
/// <typeparamref name="T"/> cannot be transformed by the reflectors as given. It is factored after an exact
/// rescaling (a power of two for an IEEE 754 type, a power of ten for <c>decimal</c>), and the coefficients,
/// refinement steps and covariance are mapped back. Inputs outside the supported numeric operating
/// domain are handled on a best-effort basis, even when the requested outputs are representable.
/// </para>
/// <para>
/// <b>Mixed-precision refinement.</b> The QR solution is refined with Björck's
/// augmented-system iterative refinement (Björck, <i>Numerical Methods for Least Squares
/// Problems</i>, 1996, Algorithm 2.9.1; Golub &amp; Van Loan section 5.3.8), which corrects
/// the solution <c>beta</c> and the residual <c>r</c> together from the blocks
/// <c>f = y - r - X.beta</c> and <c>g = -Xt.r</c>. Unlike correcting <c>beta</c> alone,
/// which helps only nearly consistent systems, this makes <c>kappa(X)</c> rather than
/// <c>kappa(X)^2 . ||r||</c> govern the attainable accuracy. When <typeparamref name="T"/>
/// is not <see cref="decimal"/>, <c>f</c> and <c>g</c> are accumulated in <c>decimal</c>
/// after an exact power-of-two rescaling of each column, of <c>y</c> and, for <c>g</c>, of
/// <c>r</c>. Scaling reduces range loss, but extreme dynamic ranges can still lose
/// small terms. <c>decimal</c>'s 96-bit significand does not meet the doubled-precision
/// residual regime (106 bits for <c>double</c>) sufficient under the remaining hypotheses
/// of Higham (2002), section 20.5. That sufficient condition is not a necessary accuracy
/// threshold; attainable accuracy depends on the problem and arithmetic.
/// Values outside <c>decimal</c>'s range, and a block <c>g</c> whose restored scale
/// <c>|X|.|r|</c> exceeds the range of <typeparamref name="T"/>, fall back to
/// coefficient-only refinement in <typeparamref name="T"/>. When <typeparamref name="T"/>
/// is <c>decimal</c>, the QR is already 28-digit and coefficient-only refinement runs in
/// <c>T</c>.
/// </para>
/// <para>
/// <b>Complexity.</b> <c>O(m . n^2)</c> flops for the QR, plus <c>O(n^3)</c> for the
/// covariance inversion. Memory: <c>O(m . n)</c> for the working copy of <c>X</c> and
/// <c>O(n^2)</c> for the covariance. Suitable for dense problems up to roughly
/// <c>n . p &lt; 10^6</c> entries; beyond that, use an iterative method.
/// </para>
/// <para>
/// <b>Residual statistics.</b> The returned residuals are <c>y - X.beta-hat</c> evaluated
/// in the extended precision above and rounded to <typeparamref name="T"/>; the residual
/// sum of squares, and every quantity derived from it, is computed in
/// <typeparamref name="T"/> from those residuals. Where every raw square keeps its precision
/// (a normal IEEE 754 sum) the plain accumulation is returned; otherwise each residual is
/// divided by the largest magnitude before squaring, so the sum, the residual standard
/// deviation and the standard errors (the Euclidean norms of the rows of
/// <c>sigma-hat · R^-1</c>) preserve representable results within the supported operating domain. <c>decimal</c>,
/// whose fixed 28-place resolution rounds every square far from one, always takes the scaled
/// form, and its results carry that resolution (an absolute step of 1e-28). A covariance
/// entry that itself under- or overflows is returned as zero or infinity; for <c>decimal</c>
/// an overflowing entry raises <see cref="InvalidOperationException"/>.
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
    /// <exception cref="ArgumentNullException"><paramref name="x"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// Dimensions are inconsistent: <paramref name="x"/> has no observations, <paramref name="y"/>
    /// length differs from <paramref name="x"/>'s row count, or the design matrix (after
    /// optional intercept prepending) has more columns than observations; or an input value is non-finite.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The design matrix is numerically rank-deficient: at some QR step the pivot column's
    /// part orthogonal to the columns already factored is zero or at most
    /// <c>max(m, p) . eps</c> times that column's own norm. Small decimal columns are
    /// first scaled by exact powers of ten. The decision depends on the pivot order.
    /// OLS is undefined on a rank-deficient design; use a regularised estimator such as ridge regression instead.
    /// Also thrown when the QR solution is not finite in the working type. IEEE types return infinity
    /// for an unrepresentable residual sum of squares or covariance entry. For <c>T = decimal</c>,
    /// an unrepresentable coefficient or statistic is refused: an <see cref="OverflowException"/>
    /// raised inside the fit is reported as this exception with the original as its inner exception.
    /// </exception>
    public static OlsResult<T> Fit(T[,] x, ReadOnlySpan<T> y, bool includeIntercept = true)
    {
        try
        {
            return FitCore(x, y, includeIntercept);
        }
        catch (OverflowException ex)
        {
            // Only decimal throws on a range overflow (IEEE types saturate to infinity, which the
            // finiteness checks inside the fit already report as InvalidOperationException). A result
            // that cannot be represented is reported with the same documented exception.
            throw new InvalidOperationException(
                "OLS arithmetic exceeded the numeric type's range: the solution or a statistic derived " +
                "from it is not representable in the working type; rescale the problem.",
                ex);
        }
    }

    private static OlsResult<T> FitCore(T[,] x, ReadOnlySpan<T> y, bool includeIntercept)
    {
        ArgumentNullException.ThrowIfNull(x);

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

        for (var i = 0; i < m; i++)
        {
            if (!T.IsFinite(y[i]))
            {
                throw new ArgumentException("Response values must be finite.", nameof(y));
            }

            for (var j = 0; j < p; j++)
            {
                if (!T.IsFinite(x[i, j]))
                {
                    throw new ArgumentException("Design values must be finite.", nameof(x));
                }
            }
        }

        var design = new T[m, designP];
        var yWork = new T[m];
        y.CopyTo(yWork);

        // A response whose norm approaches the working range cannot be transformed by the
        // reflectors (the leading entry of Qt.y can equal ||y||2) although its least-squares
        // solution is representable. Such a response is factored as y . down, an exact rescaling
        // (a shift of the decimal point for decimal, a power of two for an IEEE 754 type), and the
        // solution is scaled back below; the refinement works from the original y. The same copy
        // feeds the mean and total sum of squares, whose centred deviations can exceed the range
        // of an opposite-sign response.
        var scaleResponse = HouseholderQr<T>.TryResponseScale(y, out var responseDown, out var responseUp);
        T[]? yScaled = null;
        if (scaleResponse)
        {
            yScaled = new T[m];
            for (var i = 0; i < m; i++)
            {
                yScaled[i] = y[i] * responseDown;
            }

            Array.Copy(yScaled, yWork, m);
        }

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

        // A design column whose norm exceeds the working range cannot be factored (R's diagonal is that
        // norm) although the solution can be representable. Such a column is factored as X_j . d_j, an
        // exact power of two (IEEE 754) or of ten (decimal); the solution, the refinement step and the
        // covariance are mapped back through D below. Small decimal columns also shift upward to
        // unit scale so their absolute rounding residues do not masquerade as independent columns.
        // columnScales stays null when no column needs it.
        T[]? columnScales = null;
        for (var j = 0; j < designP; j++)
        {
            if (!HouseholderQr<T>.TryColumnScale(design, j, out var columnDown, out _))
            {
                continue;
            }

            columnScales ??= CreateOnes(designP);
            columnScales[j] = columnDown;
            for (var i = 0; i < m; i++)
            {
                design[i, j] *= columnDown;
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
        if (scaleResponse || columnScales is not null)
        {
            for (var k = 0; k < designP; k++)
            {
                // The QR solves (X D) beta' = y . down, so beta = D . beta' . up. Both factors are exact
                // powers of the same base, so their product is exact and is applied once.
                // decimal throws here when the solution is beyond its range; an IEEE type saturates to
                // infinity, which the refinement's solves report as a non-finite QR solution.
                var factor = scaleResponse ? responseUp : T.One;
                if (columnScales is not null)
                {
                    factor *= columnScales[k];
                }

                coefficients[k] *= factor;
            }
        }

        // Augmented-system iterative refinement (Björck 1996, Alg. 2.9.1; Golub & Van Loan
        // section 5.3.8) with the residual blocks in extended precision when T is not decimal.
        var residuals = new T[m];
        var rss = RefineScaledAndComputeResiduals(x, y, coefficients, design, diagonal, permutation, columnScales,
            residuals, includeIntercept, p, m, designP);

        var dof = m - designP;
        var scaledRss = ScaledSumOfSquares(residuals, out var residualScale);
        var sigmaSq = dof > 0 ? rss / T.CreateChecked(dof) : T.Zero;
        var sigma = NumericPrecision<T>.Sqrt(sigmaSq);
        // The norm can fit even when its square cannot be represented, and a variance that is
        // positive but outside the normal range (or, for decimal, any variance) has already
        // lost digits to rounding, so the scaled form decides sigma whenever the raw
        // variance did not keep its precision.
        if (residualScale > T.Zero && dof > 0 && !HouseholderQr<T>.SquareSumKeepsPrecision(sigmaSq))
        {
            sigma = residualScale * NumericPrecision<T>.Sqrt(scaledRss / T.CreateChecked(dof));
        }

        var covariance = new T[designP, designP];
        var standardErrors = new T[designP];
        // A zero inference scale implies zero covariance. Avoid forming an inverse
        // that can overflow even though the scaled result is exactly zero.
        if (sigma != T.Zero)
        {
            // The standard errors are the Euclidean norms of the rows of sigma · R⁻¹, taken
            // from the same solve as the covariance: they stay representable when the squared
            // covariance diagonal under- or overflows.
            covariance = HouseholderQr<T>.BuildScaledXtXInverse(
                design, diagonal, permutation, sigma, columnScales, out standardErrors);
        }

        // R-squared uses centred TSS when the model includes an intercept and uncentred TSS
        // otherwise (classical econometric and NIST convention). The centred form is
        // meaningful only when residuals are measured from the response mean; without
        // an intercept, the response mean is not a valid null model.
        // yStat is the response in the units of the total sum of squares: the original response, or
        // its exact power-of-ten rescaling when the factorization needed one.
        ReadOnlySpan<T> yStat = yScaled is null ? y : yScaled;
        var yMean = T.Zero;
        var yOrigin = T.Zero;
        if (includeIntercept)
        {
            // Subtract a fixed observed origin before averaging so small offsets
            // are not repeatedly rounded against a large common translation.
            yOrigin = yStat[0];
            for (var i = 1; i < m; i++)
            {
                if (T.IsNegative(yStat[i]) != T.IsNegative(yOrigin))
                {
                    // Zero keeps opposite-sign extreme observations in range.
                    yOrigin = T.Zero;
                    break;
                }
            }
            for (var i = 0; i < m; i++)
            {
                var count = T.CreateChecked(i + 1);
                var centered = yStat[i] - yOrigin;
                // Same-sign differences are range-safe and preserve constant responses
                // exactly. For opposite signs, divide before subtracting instead.
                var increment = T.IsNegative(centered) == T.IsNegative(yMean)
                    ? (centered - yMean) / count
                    : centered / count - yMean / count;
                yMean += increment;
            }
        }

        // TSS is only a denominator: keep its scale separate so squaring a large
        // response cannot overflow when the resulting R-squared is representable.
        var tssScale = T.Zero;
        for (var i = 0; i < m; i++)
        {
            tssScale = T.Max(tssScale, T.Abs((yStat[i] - yOrigin) - yMean));
        }

        var scaledTss = T.Zero;
        if (tssScale > T.Zero)
        {
            for (var i = 0; i < m; i++)
            {
                var d = ((yStat[i] - yOrigin) - yMean) / tssScale;
                scaledTss += d * d;
            }
        }

        var residualToTotalScale = tssScale > T.Zero
            ? (scaleResponse ? residualScale * responseDown : residualScale) / tssScale
            : T.Zero;
        var rSquared = scaledTss > T.Zero
            ? T.One - residualToTotalScale * (scaledRss / scaledTss) * residualToTotalScale
            : T.Zero;

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

    private static T[] CreateOnes(int length)
    {
        var ones = new T[length];
        Array.Fill(ones, T.One);
        return ones;
    }

    /// <summary>
    /// Refines <paramref name="coefficients"/> in-place with Björck's augmented-system
    /// iterative refinement, writes the residuals <c>y - X.beta</c> of the refined
    /// coefficients into <paramref name="residuals"/>, and returns their sum of squares in
    /// <typeparamref name="T"/> (see <see cref="SumOfSquares"/>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each step evaluates <c>f = y - r - X.beta</c> and <c>g = -Xt.r</c> and solves
    /// <c>[I X; Xt 0][dr; dbeta] = [f; g]</c> with the stored QR factor <c>X.P = Q.R</c>:
    /// <c>Rt.h = Pt.g</c>, <c>d = Qt.f</c>, <c>dbeta = P.R^-1.(d1 - h)</c>,
    /// <c>dr = Q.[h; d2]</c> (Björck 1996, section 2.9.2). For <typeparamref name="T"/>
    /// other than <see cref="decimal"/>, <c>f</c> and <c>g</c> are accumulated in
    /// <see cref="decimal"/> on power-of-two-rescaled data. Without that extended
    /// precision (<typeparamref name="T"/> is <c>decimal</c>, or the rescaled data
    /// overflow <c>decimal</c>) <c>r</c> is held at zero and the step is the
    /// coefficient-only correction <c>dbeta = P.R^-1.(Qt.(y - X.beta))1</c> in
    /// <typeparamref name="T"/>.
    /// </para>
    /// <para>
    /// Termination: stop after applying a correction below one machine epsilon of
    /// <c>||beta||</c>, or before applying one that is not at least half the size of the
    /// previous correction (the iteration no longer contracts), or after
    /// <see cref="MaxRefinementIterations"/> steps.
    /// </para>
    /// </remarks>
    private static T RefineAndComputeResiduals(
        T[,] x,
        ReadOnlySpan<T> y,
        T[] coefficients,
        T[,] qrFactored,
        T[] diagonal,
        int[] permutation,
        T[] residuals,
        bool includeIntercept,
        int p,
        int m,
        int designP)
        => RefineScaledAndComputeResiduals(x, y, coefficients, qrFactored, diagonal, permutation, null,
            residuals, includeIntercept, p, m, designP);

    /// <summary>
    /// <see cref="RefineAndComputeResiduals"/> for a stored factor of <c>X . D</c>, where
    /// <paramref name="columnScales"/> holds <c>D</c> in original column order (<see langword="null"/>
    /// for <c>D = I</c>): the blocks are evaluated on the original <c>X</c>, <c>g</c> enters the
    /// transposed solve as <c>D . g</c>, and each step leaves the triangular solve as <c>D . dbeta'</c>.
    /// </summary>
    private static T RefineScaledAndComputeResiduals(
        T[,] x,
        ReadOnlySpan<T> y,
        T[] coefficients,
        T[,] qrFactored,
        T[] diagonal,
        int[] permutation,
        T[]? columnScales,
        T[] residuals,
        bool includeIntercept,
        int p,
        int m,
        int designP)
    {
        var extended = typeof(T) == typeof(decimal)
            ? null
            : ExtendedResidualBlocks.TryCreate(x, y, includeIntercept, p, m, designP);

        var r = new T[m];
        var f = new T[m];
        var g = new T[designP];

        // r0 = y - X.beta0: with r = 0 the f block is exactly that residual. Without an
        // extended-precision evaluator (T = decimal, or decimal overflow) the residual is
        // not refined: r stays 0, the g block is dropped, and each step is the x-only
        // correction in T that this path has always used.
        EvaluateBlocks(ref extended, x, y, coefficients, r, f, g, includeIntercept, p, m);
        if (extended is not null)
        {
            Array.Copy(f, r, m);
        }

        var epsilon = HouseholderQr<T>.MachineEpsilon;
        var two = T.CreateChecked(2);
        var previousDeltaNorm = T.Zero;
        var c = new T[designP];

        for (var iter = 0; iter < MaxRefinementIterations; iter++)
        {
            var augmented = extended is not null;
            EvaluateBlocks(ref extended, x, y, coefficients, r, f, g, includeIntercept, p, m);
            if (augmented && extended is null)
            {
                // Extended precision overflowed during this evaluation: continue x-only.
                augmented = false;
                Array.Clear(r);
                EvaluateBlocks(ref extended, x, y, coefficients, r, f, g, includeIntercept, p, m);
            }

            if (!augmented)
            {
                Array.Clear(g);
            }

            // The stored factor is of X . D: the system [I XD; (XD)t 0][dr; D^-1 dbeta] = [f; D g] has the
            // same dr, so g enters scaled by D and the step leaves it scaled by D.
            if (columnScales is not null)
            {
                for (var k = 0; k < designP; k++)
                {
                    g[k] *= columnScales[k];
                }
            }

            // Solve the augmented correction system with the stored QR factor.
            var h = HouseholderQr<T>.SolveUpperTriangularTranspose(qrFactored, diagonal, g, designP, permutation);
            HouseholderQr<T>.ApplyQTranspose(qrFactored, diagonal, f);
            for (var k = 0; k < designP; k++)
            {
                c[k] = f[k] - h[k];
                f[k] = h[k];
            }

            var deltaBeta = HouseholderQr<T>.SolveUpperTriangular(qrFactored, diagonal, c, designP, permutation);
            if (columnScales is not null)
            {
                for (var k = 0; k < designP; k++)
                {
                    deltaBeta[k] *= columnScales[k];
                }
            }

            HouseholderQr<T>.ApplyQ(qrFactored, diagonal, f);

            // Infinity norms keep convergence diagnostics representable whenever the
            // coefficients and corrections are: neither squares nor sums can overflow.
            var deltaNorm = T.Zero;
            for (var k = 0; k < designP; k++)
            {
                deltaNorm = T.Max(deltaNorm, T.Abs(deltaBeta[k]));
            }

            // Not contracting: the correction is at the noise floor (or diverging). Keep
            // the current iterate.
            if (iter > 0 && deltaNorm > previousDeltaNorm / two)
            {
                break;
            }

            var betaNorm = T.Zero;
            for (var k = 0; k < designP; k++)
            {
                coefficients[k] += deltaBeta[k];
                betaNorm = T.Max(betaNorm, T.Abs(coefficients[k]));
            }

            if (augmented)
            {
                for (var i = 0; i < m; i++)
                {
                    r[i] += f[i];
                }
            }

            if (deltaNorm <= epsilon * betaNorm)
            {
                break;
            }

            previousDeltaNorm = deltaNorm;
        }

        // Final residuals y - X.beta-hat (r = 0 makes f exactly that), rounded to T.
        Array.Clear(r);
        EvaluateBlocks(ref extended, x, y, coefficients, r, f, g, includeIntercept, p, m);
        for (var i = 0; i < m; i++)
        {
            residuals[i] = f[i];
        }

        return SumOfSquares(residuals);
    }

    /// <summary>
    /// <c>Σᵢ (vᵢ / s)²</c> for <c>s = max|vᵢ|</c>, returned as <paramref name="scale"/>; every
    /// term lies in <c>[0, 1]</c>, so the sum is in <c>[1, n]</c> and neither under- nor
    /// overflows. Both are zero for an all-zero vector.
    /// </summary>
    private static T ScaledSumOfSquares(T[] values, out T scale)
    {
        scale = T.Zero;
        foreach (var value in values)
        {
            scale = T.Max(scale, T.Abs(value));
        }

        var sum = T.Zero;
        if (scale > T.Zero)
        {
            foreach (var value in values)
            {
                var normalized = value / scale;
                sum += normalized * normalized;
            }
        }

        return sum;
    }

    /// <summary>
    /// <c>Σᵢ vᵢ²</c> of <paramref name="values"/> in <typeparamref name="T"/>. Where every raw
    /// square kept its precision (a normal IEEE 754 sum) the plain accumulation is returned
    /// unchanged; otherwise the sum is <c>(s · Σᵢ (vᵢ / s)²) · s</c> for <c>s = max|vᵢ|</c>,
    /// which squares nothing outside the working range and is rounded to
    /// <typeparamref name="T"/> only once, at its exact magnitude.
    /// </summary>
    private static T SumOfSquares(T[] values)
    {
        var scaledSum = ScaledSumOfSquares(values, out var scale);
        if (scale == T.Zero)
        {
            return T.Zero;
        }

        if (HouseholderQr<T>.HasFloatingExponent)
        {
            var raw = T.Zero;
            foreach (var value in values)
            {
                raw += value * value;
            }

            if (HouseholderQr<T>.SquareSumKeepsPrecision(raw))
            {
                return raw;
            }
        }

        return (scale * scaledSum) * scale;
    }

    /// <summary>
    /// Evaluates the augmented-system residual blocks <c>f = y - r - X.beta</c> and
    /// <c>g = -Xt.r</c> in extended precision when available, else in
    /// <typeparamref name="T"/>. An extended evaluation that overflows
    /// <see cref="decimal"/> permanently switches this fit to <typeparamref name="T"/>.
    /// </summary>
    private static void EvaluateBlocks(
        ref ExtendedResidualBlocks? extended,
        T[,] x,
        ReadOnlySpan<T> y,
        T[] coefficients,
        T[] r,
        T[] f,
        T[] g,
        bool includeIntercept,
        int p,
        int m)
    {
        if (extended is not null)
        {
            try
            {
                extended.Evaluate(coefficients, r, f, g);
                return;
            }
            catch (OverflowException)
            {
                extended = null;
            }
        }

        var offset = includeIntercept ? 1 : 0;
        Array.Clear(g);
        for (var i = 0; i < m; i++)
        {
            try
            {
                var prediction = includeIntercept ? coefficients[0] : T.Zero;
                for (var j = 0; j < p; j++)
                {
                    prediction += coefficients[j + offset] * x[i, j];
                }

                f[i] = y[i] - r[i] - prediction;
            }
            catch (OverflowException)
            {
                // Only decimal throws here. A term beta_j * x_ij beyond its range can still
                // belong to a representable residual, so evaluate this row on a power-of-ten
                // rescaling instead of refusing the fit.
                f[i] = ScaledRowResidual(x, y, coefficients, r, i, includeIntercept, p);
            }

            if (includeIntercept)
            {
                g[0] -= r[i];
            }

            for (var j = 0; j < p; j++)
            {
                g[j + offset] -= x[i, j] * r[i];
            }
        }
    }

    /// <summary>
    /// Residual <c>y_i - r_i - Σⱼ βⱼ·x_ij</c> of one row when forming a product
    /// <c>βⱼ·x_ij</c> overflows <see cref="decimal"/>. Every coefficient, the response and the
    /// iterate are multiplied by <c>10^-k</c> (an exact shift of the decimal point; the
    /// <c>k</c> that brings the largest term and the sum of all terms under 1e28 is chosen from
    /// <see cref="double"/> magnitudes, a power-of-ten decision only), the row is accumulated
    /// there, and the residual is shifted back by <c>10^k</c>. A residual that is itself
    /// beyond <see cref="decimal"/> still throws <see cref="OverflowException"/>.
    /// </summary>
    private static T ScaledRowResidual(
        T[,] x, ReadOnlySpan<T> y, T[] coefficients, T[] r, int row, bool includeIntercept, int p)
    {
        var offset = includeIntercept ? 1 : 0;
        var largestExponent = includeIntercept ? Log10Magnitude(coefficients[0]) : double.NegativeInfinity;
        for (var j = 0; j < p; j++)
        {
            largestExponent = Math.Max(largestExponent, Log10Magnitude(coefficients[j + offset]) + Log10Magnitude(x[row, j]));
        }

        // Headroom for the accumulated terms plus y and r, in decimal digits of decimal.MaxValue (7.9e28).
        var shift = double.IsFinite(largestExponent)
            ? (int)Math.Ceiling(largestExponent + Math.Log10(p + 3) - 28.0)
            : 1;
        shift = Math.Clamp(shift, 1, 28);

        var down = T.CreateChecked(new decimal(1, 0, 0, false, (byte)shift));
        var up = T.CreateChecked(decimal.One);
        var ten = T.CreateChecked(10);
        for (var k = 0; k < shift; k++)
        {
            up *= ten;
        }

        var prediction = includeIntercept ? coefficients[0] * down : T.Zero;
        for (var j = 0; j < p; j++)
        {
            prediction += (coefficients[j + offset] * down) * x[row, j];
        }

        return (((y[row] * down) - (r[row] * down)) - prediction) * up;
    }

    private static double Log10Magnitude(T value) => Math.Log10(Math.Abs(double.CreateChecked(value)));

    /// <summary>
    /// Residual-block evaluator in <see cref="decimal"/> for a non-decimal
    /// <typeparamref name="T"/>. Each design column <c>j</c> is stored as
    /// <c>X[:, j] . 2^-e_j</c> and the response as <c>y . 2^s</c>, with <c>e_j</c> the
    /// binary exponent of the column's largest entry and <c>s</c> minus that of the
    /// response's, so every stored value is O(1) and keeps ~28 significant digits
    /// in <c>decimal</c>'s fixed 28-place scale. The residual iterate <c>r</c> enters
    /// <c>g = -Xt.r</c> rescaled by its own largest entry's exponent, for the same reason
    /// (it shrinks far below <c>y</c> as the iteration converges). Power-of-two scaling is exact, and the
    /// conversions from <typeparamref name="T"/> go through the exact binary value rather
    /// than <c>decimal</c>'s 15-significant-digit conversion.
    /// </summary>
    private sealed class ExtendedResidualBlocks
    {
        private static readonly decimal[] s_powersOfTwo = BuildPowersOfTwo();

        private readonly decimal[,] _design;
        private readonly decimal[] _response;
        private readonly int[] _columnExponents;
        private readonly int _responseShift;
        private readonly decimal[] _beta;
        private readonly decimal[] _r;
        private readonly int _m;
        private readonly int _designP;

        private ExtendedResidualBlocks(decimal[,] design, decimal[] response, int[] columnExponents, int responseShift)
        {
            _design = design;
            _response = response;
            _columnExponents = columnExponents;
            _responseShift = responseShift;
            _m = response.Length;
            _designP = columnExponents.Length;
            _beta = new decimal[_designP];
            _r = new decimal[_m];
        }

        public static ExtendedResidualBlocks? TryCreate(
            T[,] x, ReadOnlySpan<T> y, bool includeIntercept, int p, int m, int designP)
        {
            var offset = includeIntercept ? 1 : 0;
            var columnExponents = new int[designP];
            for (var j = 0; j < p; j++)
            {
                var columnMax = 0.0;
                for (var i = 0; i < m; i++)
                {
                    columnMax = Math.Max(columnMax, Math.Abs(double.CreateChecked(x[i, j])));
                }

                if (!double.IsFinite(columnMax))
                {
                    return null;
                }

                columnExponents[j + offset] = columnMax == 0.0 ? 0 : Math.ILogB(columnMax);
            }

            var responseMax = 0.0;
            for (var i = 0; i < m; i++)
            {
                responseMax = Math.Max(responseMax, Math.Abs(double.CreateChecked(y[i])));
            }

            if (!double.IsFinite(responseMax))
            {
                return null;
            }

            var responseShift = responseMax == 0.0 ? 0 : -Math.ILogB(responseMax);

            try
            {
                var design = new decimal[m, designP];
                var response = new decimal[m];
                for (var i = 0; i < m; i++)
                {
                    response[i] = ToScaledDecimal(double.CreateChecked(y[i]), responseShift);
                    if (includeIntercept)
                    {
                        design[i, 0] = 1m;
                    }

                    for (var j = 0; j < p; j++)
                    {
                        design[i, j + offset] = ToScaledDecimal(double.CreateChecked(x[i, j]), -columnExponents[j + offset]);
                    }
                }

                return new ExtendedResidualBlocks(design, response, columnExponents, responseShift);
            }
            catch (OverflowException)
            {
                return null;
            }
        }

        /// <summary>Writes <c>f = y - r - X.beta</c> and <c>g = -Xt.r</c>; throws <see cref="OverflowException"/> on decimal overflow.</summary>
        public void Evaluate(T[] coefficients, T[] r, T[] f, T[] g)
        {
            // beta'_j = beta_j . 2^(e_j + s) and r' = r . 2^s, so X'.beta' = X.beta . 2^s.
            for (var k = 0; k < _designP; k++)
            {
                _beta[k] = ToScaledDecimal(double.CreateChecked(coefficients[k]), _columnExponents[k] + _responseShift);
            }

            // r enters f at the response's scale, but g = -Xt.r is stored at r's own binary
            // exponent: at the response's scale a small r (and the tiny g of a near-converged
            // iterate) would keep only decimal's fixed absolute resolution, which
            // (XtX)^-1 amplifies by 1/sigma_min^2.
            var residualMax = 0.0;
            for (var i = 0; i < _m; i++)
            {
                residualMax = Math.Max(residualMax, Math.Abs(double.CreateChecked(r[i])));
            }

            if (!double.IsFinite(residualMax))
            {
                throw new OverflowException("Non-finite residual has no decimal representation.");
            }

            var residualShift = residualMax == 0.0 ? 0 : -Math.ILogB(residualMax);
            for (var i = 0; i < _m; i++)
            {
                _r[i] = ToScaledDecimal(double.CreateChecked(r[i]), residualShift);
            }

            for (var i = 0; i < _m; i++)
            {
                var fi = _response[i] - ToScaledDecimal(double.CreateChecked(r[i]), _responseShift);
                for (var k = 0; k < _designP; k++)
                {
                    fi -= _design[i, k] * _beta[k];
                }

                f[i] = T.CreateChecked(Math.ScaleB((double)fi, -_responseShift));
            }

            for (var k = 0; k < _designP; k++)
            {
                var gk = 0m;
                for (var i = 0; i < _m; i++)
                {
                    gk -= _design[i, k] * _r[i];
                }

                g[k] = T.CreateChecked(Math.ScaleB((double)gk, _columnExponents[k] - residualShift));
                if (!T.IsFinite(g[k]))
                {
                    // |X_k| . |r| can exceed T's range although the correction h = R^-t . Pt . g has
                    // r's scale; restoring 2^(e_k + e_r) saturates instead of throwing, so report it
                    // as the overflow that switches this fit to coefficient-only refinement in T.
                    throw new OverflowException("The restored gradient block -Xt.r is beyond the working type's range.");
                }
            }
        }

        /// <summary>
        /// Converts <c>value . 2^shift</c> to <see cref="decimal"/> from the exact binary
        /// significand, rounding only in the final power-of-two divisions (~28 significant
        /// digits for values of order one). Throws <see cref="OverflowException"/> for a
        /// non-finite value or a result beyond <see cref="decimal"/>'s range.
        /// </summary>
        internal static decimal ToScaledDecimal(double value, int shift)
        {
            if (!double.IsFinite(value))
            {
                throw new OverflowException("Non-finite value has no decimal representation.");
            }

            if (value == 0.0)
            {
                return 0m;
            }

            var bits = BitConverter.DoubleToInt64Bits(value);
            var biasedExponent = (int)((bits >> 52) & 0x7FF);
            var significand = bits & 0xF_FFFF_FFFF_FFFFL;
            int exponent;
            if (biasedExponent == 0)
            {
                exponent = -1074;
            }
            else
            {
                significand |= 1L << 52;
                exponent = biasedExponent - 1075;
            }

            var trailingZeros = BitOperations.TrailingZeroCount(significand);
            significand >>= trailingZeros;
            exponent += trailingZeros + shift;

            decimal result = significand;
            while (exponent > 0)
            {
                var step = Math.Min(exponent, 64);
                result *= s_powersOfTwo[step];
                exponent -= step;
            }

            while (exponent < 0 && result != 0m)
            {
                var step = Math.Min(-exponent, 64);
                result /= s_powersOfTwo[step];
                exponent += step;
            }

            return bits < 0 ? -result : result;
        }

        private static decimal[] BuildPowersOfTwo()
        {
            var powers = new decimal[65];
            powers[0] = 1m;
            for (var k = 1; k < powers.Length; k++)
            {
                powers[k] = powers[k - 1] * 2m;
            }

            return powers;
        }
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
/// but NIST's degree-five Wampler design has condition number about 6.4e6.
/// Wampler4 and Wampler5 use that same design with different response noise.
/// Householder QR avoids squaring the design condition number; arithmetic precision,
/// residual magnitude and input rounding still affect attainable coefficient accuracy. References: Golub &amp; Van Loan, <i>Matrix Computations</i>, 4th ed., section 5.3;
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
[Obsolete(
    "Use OrdinaryLeastSquares<double> (or OrdinaryLeastSquares<decimal>). " +
    "This concrete-typed facade will be removed in v3.0. " +
    "See docs/migration-to-generic-math.md for migration examples.",
    DiagnosticId = "BNU0001",
    UrlFormat = "https://github.com/boutquin/Boutquin.Numerics/blob/main/docs/migration-to-generic-math.md")]
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
    /// <exception cref="ArgumentNullException"><paramref name="x"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// Dimensions are inconsistent: <paramref name="x"/> has no observations, <paramref name="y"/>
    /// length differs from <paramref name="x"/>'s row count, or the design matrix (after
    /// optional intercept prepending) has more columns than observations; or an input value is non-finite.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The design matrix is numerically rank-deficient (a QR column is zero or R has a
    /// zero diagonal). OLS is undefined on a rank-deficient design; use a regularised
    /// estimator such as ridge regression instead.
    /// Also thrown when the QR solution is not finite in the working type.
    /// </exception>
    public static OlsResult Fit(double[,] x, ReadOnlySpan<double> y, bool includeIntercept = true)
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
