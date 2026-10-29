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

namespace Boutquin.Numerics.Solvers;

/// <summary>
/// Generic output of an ordinary least-squares fit.
/// </summary>
/// <remarks>
/// <para><b>Generic-math tier:</b> A+√ (arithmetic + sqrt). Works for any <c>T</c>
/// implementing <see cref="IFloatingPoint{TSelf}"/>. At <c>T = decimal</c> the fit runs
/// in <c>decimal</c>'s 28-significant-digit arithmetic, which is not the accuracy it
/// delivers: on the NIST StRD linear-regression suite the tests hold its coefficients to
/// 1e-13 relative of the certified values on every problem except <c>Filip</c>
/// (degree-10 polynomial, cond ≈ 10¹⁰), where the 15-digit tabulated inputs limit it to
/// 1e-6.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed record OlsResult<T>
    where T : IFloatingPoint<T>
{
    private T[] _coefficients = [];
    private T[] _standardErrors = [];
    private T[] _residuals = [];
    private T[,] _covarianceMatrix = new T[0, 0];

    /// <summary>Coefficient vector <c>β̂</c>.</summary>
    public required T[] Coefficients
    {
        get => (T[])_coefficients.Clone();
        init => _coefficients = (T[])value.Clone();
    }

    /// <summary>Classical standard errors <c>√diag(σ̂² · (XᵀX)⁻¹)</c>.</summary>
    public required T[] StandardErrors
    {
        get => (T[])_standardErrors.Clone();
        init => _standardErrors = (T[])value.Clone();
    }

    /// <summary>Per-observation residuals <c>y − X·β̂</c>.</summary>
    public required T[] Residuals
    {
        get => (T[])_residuals.Clone();
        init => _residuals = (T[])value.Clone();
    }

    /// <summary>Residual sum of squares <c>Σᵢ rᵢ²</c>.</summary>
    public required T ResidualSumOfSquares { get; init; }

    /// <summary>Residual standard deviation <c>σ̂ = √(RSS / (n − p))</c>.</summary>
    public required T ResidualStandardDeviation { get; init; }

    /// <summary>
    /// Residual degrees of freedom <c>n − p</c>.
    /// </summary>
    /// <remarks>
    /// When this is zero (a saturated design where the observation count equals the
    /// coefficient count) the residual variance estimate <c>σ̂² = RSS / (n − p)</c> is
    /// undefined, so <see cref="StandardErrors"/>, <see cref="ResidualStandardDeviation"/>,
    /// and <see cref="CovarianceMatrix"/> are all returned as zero rather than throwing.
    /// A zeroed standard error is not "perfect precision" — it signals that no inferential
    /// content is available. Test <see cref="HasInferentialContent"/> before consuming any
    /// inference-bearing field.
    /// </remarks>
    public required int DegreesOfFreedom { get; init; }

    /// <summary>
    /// Whether the fit carries inferential content. <see langword="true"/> exactly when
    /// <see cref="DegreesOfFreedom"/> is positive. When <see langword="false"/> the design
    /// is saturated (<c>n = p</c>) and <see cref="StandardErrors"/>,
    /// <see cref="ResidualStandardDeviation"/>, and <see cref="CovarianceMatrix"/> are
    /// zeroed — point estimates in <see cref="Coefficients"/> remain valid, but no
    /// standard errors, confidence intervals, or hypothesis tests can be derived.
    /// </summary>
    public bool HasInferentialContent => DegreesOfFreedom > 0;

    /// <summary>Coefficient of determination <c>R²</c>.</summary>
    public required T RSquared { get; init; }

    /// <summary>Covariance matrix <c>Cov(β̂) = σ̂² · (XᵀX)⁻¹</c>.</summary>
    public required T[,] CovarianceMatrix
    {
        get => (T[,])_covarianceMatrix.Clone();
        init => _covarianceMatrix = (T[,])value.Clone();
    }
}

/// <summary>
/// Output of an ordinary least-squares fit: coefficient vector, classical standard
/// errors, fit residuals, residual sum of squares, residual standard deviation,
/// degrees of freedom, coefficient of determination, and the coefficient
/// covariance matrix already scaled by <c>σ̂²</c> for downstream inference.
/// </summary>
/// <remarks>
/// <para>
/// The contract is identical to what econometric and statistical packages publish
/// (NIST StRD, R's <c>lm()</c>, Python's <c>statsmodels.OLS</c>): given the model
/// <c>y = Xβ + ε</c> with <c>ε ∼ N(0, σ²·I)</c>, the OLS estimator is
/// <c>β̂ = (XᵀX)⁻¹ · Xᵀ · y</c>, the unbiased variance estimate is
/// <c>σ̂² = RSS / (n − p)</c>, and <c>Cov(β̂) = σ̂² · (XᵀX)⁻¹</c>.
/// Standard errors are the square roots of the diagonal of this covariance matrix.
/// </para>
/// <para>
/// When the fit included an intercept, <see cref="Coefficients"/>[0] is the intercept
/// and <see cref="Coefficients"/>[1..p] are the slopes matching the caller's column order
/// in <c>X</c>. When the fit omitted the intercept, <see cref="Coefficients"/> has the
/// same length as <c>X</c>'s column count.
/// </para>
/// <para>
/// <see cref="RSquared"/> uses the centred total sum of squares <c>Σᵢ(yᵢ − ȳ)²</c> when
/// the fit included an intercept and the uncentred <c>Σᵢ yᵢ²</c> when it did not,
/// matching NIST StRD's convention for the <c>NoInt1</c>/<c>NoInt2</c> problems.
/// </para>
/// </remarks>
public sealed record OlsResult
{
    private double[] _coefficients = [];
    private double[] _standardErrors = [];
    private double[] _residuals = [];
    private double[,] _covarianceMatrix = new double[0, 0];

    /// <summary>Coefficient vector <c>β̂</c>. Length <c>p</c> (or <c>p + 1</c> with intercept).</summary>
    public required double[] Coefficients
    {
        get => (double[])_coefficients.Clone();
        init => _coefficients = (double[])value.Clone();
    }

    /// <summary>
    /// Classical standard errors <c>√diag(σ̂² · (XᵀX)⁻¹)</c>. Indexed in the same order
    /// as <see cref="Coefficients"/>.
    /// </summary>
    public required double[] StandardErrors
    {
        get => (double[])_standardErrors.Clone();
        init => _standardErrors = (double[])value.Clone();
    }

    /// <summary>
    /// Per-observation residuals <c>y − X·β̂</c>. Length matches the caller's <c>y</c>.
    /// </summary>
    public required double[] Residuals
    {
        get => (double[])_residuals.Clone();
        init => _residuals = (double[])value.Clone();
    }

    /// <summary>Residual sum of squares <c>Σᵢ rᵢ²</c>.</summary>
    public required double ResidualSumOfSquares { get; init; }

    /// <summary>
    /// Residual standard deviation <c>σ̂ = √(RSS / (n − p))</c>. Returns zero when the
    /// residual degrees of freedom are zero or negative (saturated design); callers
    /// should treat that state as "no inferential content".
    /// </summary>
    public required double ResidualStandardDeviation { get; init; }

    /// <summary>Residual degrees of freedom <c>n − p</c>.</summary>
    public required int DegreesOfFreedom { get; init; }

    /// <summary>
    /// Coefficient of determination. Defined as <c>1 − RSS / TSS</c>, with <c>TSS</c>
    /// centred against the response mean when the fit included an intercept and
    /// uncentred otherwise.
    /// </summary>
    public required double RSquared { get; init; }

    /// <summary>
    /// <c>Cov(β̂) = σ̂² · (XᵀX)⁻¹</c>. Already scaled by the residual variance;
    /// symmetric positive semidefinite for a full-column-rank design, and positive
    /// definite only when that variance is positive. An exact or saturated fit returns
    /// the zero matrix. <c>Fit</c> throws for a rank-deficient design.
    /// </summary>
    public required double[,] CovarianceMatrix
    {
        get => (double[,])_covarianceMatrix.Clone();
        init => _covarianceMatrix = (double[,])value.Clone();
    }
}
