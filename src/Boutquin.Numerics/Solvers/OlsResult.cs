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
/// implementing <see cref="IFloatingPoint{TSelf}"/>. Instantiate at <c>T = decimal</c>
/// for uniform 28-digit precision on ill-conditioned problems.</para>
/// </remarks>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed record OlsResult<T>
    where T : IFloatingPoint<T>
{
    /// <summary>Coefficient vector <c>β̂</c>.</summary>
    public required T[] Coefficients { get; init; }

    /// <summary>Classical standard errors <c>√diag(σ̂² · (XᵀX)⁻¹)</c>.</summary>
    public required T[] StandardErrors { get; init; }

    /// <summary>Per-observation residuals <c>y − X·β̂</c>.</summary>
    public required T[] Residuals { get; init; }

    /// <summary>Residual sum of squares <c>Σᵢ rᵢ²</c>.</summary>
    public required T ResidualSumOfSquares { get; init; }

    /// <summary>Residual standard deviation <c>σ̂ = √(RSS / (n − p))</c>.</summary>
    public required T ResidualStandardDeviation { get; init; }

    /// <summary>Residual degrees of freedom <c>n − p</c>.</summary>
    public required int DegreesOfFreedom { get; init; }

    /// <summary>Coefficient of determination <c>R²</c>.</summary>
    public required T RSquared { get; init; }

    /// <summary>Covariance matrix <c>Cov(β̂) = σ̂² · (XᵀX)⁻¹</c>.</summary>
    public required T[,] CovarianceMatrix { get; init; }
}

/// <summary>
/// Output of an ordinary least-squares fit: coefficient vector, classical standard
/// errors, fit residuals, residual sum of squares, residual standard deviation,
/// degrees of freedom, coefficient of determination, and the unscaled inverse
/// normal-equation matrix that downstream inference routines (Wald tests,
/// F-tests, confidence ellipsoids) multiply by <c>σ̂²</c>.
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
    /// <summary>Coefficient vector <c>β̂</c>. Length <c>p</c> (or <c>p + 1</c> with intercept).</summary>
    public required double[] Coefficients { get; init; }

    /// <summary>
    /// Classical standard errors <c>√diag(σ̂² · (XᵀX)⁻¹)</c>. Indexed in the same order
    /// as <see cref="Coefficients"/>.
    /// </summary>
    public required double[] StandardErrors { get; init; }

    /// <summary>
    /// Per-observation residuals <c>y − X·β̂</c>. Length matches the caller's <c>y</c>.
    /// </summary>
    public required double[] Residuals { get; init; }

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
    /// <c>Cov(β̂) = σ̂² · (XᵀX)⁻¹</c>. Symmetric positive definite when the design matrix
    /// has full column rank; raised as an invalid-operation from <c>Fit</c> otherwise.
    /// </summary>
    public required double[,] CovarianceMatrix { get; init; }
}
