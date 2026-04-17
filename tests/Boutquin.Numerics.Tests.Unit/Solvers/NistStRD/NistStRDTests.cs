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

using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

/// <summary>
/// NIST Statistical Reference Datasets — nonlinear least-squares benchmark suite.
/// Each test runs <see cref="LevenbergMarquardtSolver"/> against a canonical problem
/// from <c>https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml</c> starting from
/// NIST's "Start 2" (easier initial guess) and compares the converged parameters and
/// residual sum of squares to the certified values.
/// </summary>
/// <remarks>
/// <para>
/// Certified SSE is the full residual sum of squares <c>Σ rᵢ²</c>. The solver reports
/// <c>FinalCost = ½·Σ rᵢ²</c>, so the assertion multiplies <c>FinalCost</c> by 2 before
/// comparing to the NIST SSE value.
/// </para>
/// <para>
/// Relative tolerances (1e-3 for parameters, 5% for SSE) are loose enough to accommodate
/// solver termination at a point slightly away from the NIST-certified optimum while
/// still failing loudly if the solver lands on a local minimum or stalls.
/// </para>
/// </remarks>
public sealed class NistStRDTests
{
    // ── Lanczos3 — Sum of three exponentials (24 obs, 6 params) ────────────

    [Fact]
    public void Lanczos3_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Lanczos3();

        double[] Residuals(double[] theta)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = (theta[0] * Math.Exp(-theta[1] * x))
                          + (theta[2] * Math.Exp(-theta[3] * x))
                          + (theta[4] * Math.Exp(-theta[5] * x));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var solver = new LevenbergMarquardtSolver(maxIterations: 500);
        var result = solver.Solve(Residuals, data.Start2);

        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    // ── Eckerle4 — Gaussian peak (35 obs, 3 params) ────────────────────────

    [Fact]
    public void Eckerle4_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Eckerle4();

        double[] Residuals(double[] theta)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var centered = (x - theta[2]) / theta[1];
                var model = (theta[0] / theta[1]) * Math.Exp(-0.5 * centered * centered);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var solver = new LevenbergMarquardtSolver(maxIterations: 500);
        var result = solver.Solve(Residuals, data.Start2);

        AssertNistConverged(result, data, parameterTolerance: 1e-3, sseTolerance: 0.05);
    }

    // ── Bennett5 — Superconductivity power law (154 obs, 3 params) ─────────

    [Fact]
    public void Bennett5_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Bennett5();

        double[] Residuals(double[] theta)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = theta[0] * Math.Pow(theta[1] + x, -1.0 / theta[2]);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var solver = new LevenbergMarquardtSolver(maxIterations: 1000);
        var result = solver.Solve(Residuals, data.Start2);

        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    // ── Hahn1 — Thermal-expansion rational fit (236 obs, 7 params) ─────────

    [Fact]
    public void Hahn1_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Hahn1();

        double[] Residuals(double[] theta)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var x2 = x * x;
                var x3 = x2 * x;
                var numerator = theta[0] + (theta[1] * x) + (theta[2] * x2) + (theta[3] * x3);
                var denominator = 1.0 + (theta[4] * x) + (theta[5] * x2) + (theta[6] * x3);
                r[i] = data.Y[i] - (numerator / denominator);
            }

            return r;
        }

        var solver = new LevenbergMarquardtSolver(maxIterations: 2000);
        var result = solver.Solve(Residuals, data.Start2);

        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    // ── Kirby2 — Optical-sensor rational fit (151 obs, 5 params) ───────────

    [Fact]
    public void Kirby2_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Kirby2();

        double[] Residuals(double[] theta)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var x2 = x * x;
                var numerator = theta[0] + (theta[1] * x) + (theta[2] * x2);
                var denominator = 1.0 + (theta[3] * x) + (theta[4] * x2);
                r[i] = data.Y[i] - (numerator / denominator);
            }

            return r;
        }

        var solver = new LevenbergMarquardtSolver(maxIterations: 1000);
        var result = solver.Solve(Residuals, data.Start2);

        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    // ── Phase 2 — Lower-difficulty tier (paramTol 1e-3, sseTol 0.01) ───────

    [Fact]
    public void Misra1a_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Misra1a();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var model = t[0] * (1.0 - Math.Exp(-t[1] * data.X[i]));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-3, sseTolerance: 0.01);
    }

    [Fact]
    public void Chwirut1_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Chwirut1();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = Math.Exp(-t[0] * x) / (t[1] + t[2] * x);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-3, sseTolerance: 0.01);
    }

    [Fact]
    public void Chwirut2_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Chwirut2();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = Math.Exp(-t[0] * x) / (t[1] + t[2] * x);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-3, sseTolerance: 0.01);
    }

    [Fact]
    public void Lanczos1_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Lanczos1();
        AssertSixExponential(data, paramTol: 1e-3, sseTol: 0.01, maxIterations: 500);
    }

    [Fact]
    public void Lanczos2_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Lanczos2();
        AssertSixExponential(data, paramTol: 1e-3, sseTol: 0.01, maxIterations: 500);
    }

    [Fact]
    public void Gauss1_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Gauss1();
        AssertEightParameterGaussianSum(data, paramTol: 1e-3, sseTol: 0.01, maxIterations: 500);
    }

    [Fact]
    public void Gauss2_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Gauss2();
        AssertEightParameterGaussianSum(data, paramTol: 1e-3, sseTol: 0.01, maxIterations: 500);
    }

    [Fact]
    public void DanWood_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.DanWood();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                r[i] = data.Y[i] - t[0] * Math.Pow(data.X[i], t[1]);
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-3, sseTolerance: 0.01);
    }

    [Fact]
    public void Misra1b_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Misra1b();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var denom = 1.0 + 0.5 * t[1] * data.X[i];
                var model = t[0] * (1.0 - 1.0 / (denom * denom));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-3, sseTolerance: 0.01);
    }

    // ── Phase 2 — Average-difficulty tier (paramTol 1e-2, sseTol 0.05) ─────

    [Fact]
    public void Nelson_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Nelson();
        // Nelson publishes a two-predictor log-response model: log(y) = b1 − b2·x1·exp(−b3·x2) + e.
        // NIST stores x1 (time) and x2 (temperature) as a flat row-major [n, 2] slab; the residual
        // uses log(y) so the model is linear-in-parameters in the conventional NLS sense.
        var n = data.Y.Length;

        double[] Residuals(double[] t)
        {
            var r = new double[n];
            for (var i = 0; i < n; i++)
            {
                var x1 = data.X[2 * i];
                var x2 = data.X[2 * i + 1];
                var model = t[0] - t[1] * x1 * Math.Exp(-t[2] * x2);
                r[i] = Math.Log(data.Y[i]) - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 1000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void MGH17_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.MGH17();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = t[0] + t[1] * Math.Exp(-x * t[3]) + t[2] * Math.Exp(-x * t[4]);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void Gauss3_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Gauss3();
        AssertEightParameterGaussianSum(data, paramTol: 1e-2, sseTol: 0.05, maxIterations: 2000);
    }

    [Fact]
    public void Misra1c_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Misra1c();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var model = t[0] * (1.0 - 1.0 / Math.Sqrt(1.0 + 2.0 * t[1] * data.X[i]));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void Misra1d_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Misra1d();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = t[0] * t[1] * x / (1.0 + t[1] * x);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 500).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void Roszman1_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Roszman1();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                // NIST defines the model with single-argument arctan (range (−π/2, π/2)),
                // not the two-argument form; Math.Atan matches the published contract.
                var model = t[0] - t[1] * x - Math.Atan(t[2] / (x - t[3])) / Math.PI;
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 1000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void ENSO_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.ENSO();
        var twoPi = 2.0 * Math.PI;

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var w12 = twoPi * x / 12.0;
                var w1 = twoPi * x / t[3];
                var w2 = twoPi * x / t[6];
                var model = t[0]
                    + t[1] * Math.Cos(w12) + t[2] * Math.Sin(w12)
                    + t[4] * Math.Cos(w1) + t[5] * Math.Sin(w1)
                    + t[7] * Math.Cos(w2) + t[8] * Math.Sin(w2);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    // ── Phase 2 — Higher-difficulty tier (paramTol 1e-2, sseTol 0.05) ──────

    [Fact]
    public void MGH09_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.MGH09();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var x2 = x * x;
                var model = t[0] * (x2 + x * t[1]) / (x2 + x * t[2] + t[3]);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void Thurber_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Thurber();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var x2 = x * x;
                var x3 = x2 * x;
                var numerator = t[0] + t[1] * x + t[2] * x2 + t[3] * x3;
                var denominator = 1.0 + t[4] * x + t[5] * x2 + t[6] * x3;
                r[i] = data.Y[i] - numerator / denominator;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void BoxBOD_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.BoxBOD();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                r[i] = data.Y[i] - t[0] * (1.0 - Math.Exp(-t[1] * data.X[i]));
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void Rat42_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Rat42();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var model = t[0] / (1.0 + Math.Exp(t[1] - t[2] * data.X[i]));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 1000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void MGH10_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.MGH10();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var model = t[0] * Math.Exp(t[1] / (data.X[i] + t[2]));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    [Fact]
    public void Rat43_ConvergesToCertifiedValues()
    {
        var data = NistStRDData.Rat43();

        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var denom = 1.0 + Math.Exp(t[1] - t[2] * data.X[i]);
                var model = t[0] / Math.Pow(denom, 1.0 / t[3]);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: 2000).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: 1e-2, sseTolerance: 0.05);
    }

    // ── Model helpers shared by multi-problem families ─────────────────────

    private static void AssertSixExponential(
        NistStRDData.Problem data,
        double paramTol,
        double sseTol,
        int maxIterations)
    {
        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var model = t[0] * Math.Exp(-t[1] * x)
                          + t[2] * Math.Exp(-t[3] * x)
                          + t[4] * Math.Exp(-t[5] * x);
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: maxIterations).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: paramTol, sseTolerance: sseTol);
    }

    private static void AssertEightParameterGaussianSum(
        NistStRDData.Problem data,
        double paramTol,
        double sseTol,
        int maxIterations)
    {
        // y = b1·exp(−b2·x) + b3·exp(−((x−b4)/b5)²) + b6·exp(−((x−b7)/b8)²)
        double[] Residuals(double[] t)
        {
            var r = new double[data.X.Length];
            for (var i = 0; i < data.X.Length; i++)
            {
                var x = data.X[i];
                var g1 = (x - t[3]) / t[4];
                var g2 = (x - t[6]) / t[7];
                var model = t[0] * Math.Exp(-t[1] * x)
                          + t[2] * Math.Exp(-(g1 * g1))
                          + t[5] * Math.Exp(-(g2 * g2));
                r[i] = data.Y[i] - model;
            }

            return r;
        }

        var result = new LevenbergMarquardtSolver(maxIterations: maxIterations).Solve(Residuals, data.Start2);
        AssertNistConverged(result, data, parameterTolerance: paramTol, sseTolerance: sseTol);
    }

    // ── Shared assertion helper ────────────────────────────────────────────

    private static void AssertNistConverged(
        MultivariateSolverResult result,
        NistStRDData.Problem data,
        double parameterTolerance,
        double sseTolerance)
    {
        var diag = $"[termination={result.TerminationReason}, iters={result.Iterations}, cost={result.FinalCost:G6}]";
        result.Converged.Should().BeTrue(because: diag);

        var sse = 2.0 * result.FinalCost;

        // NIST publishes two datasets whose certified SSE lives at or below the practical
        // double-precision floor for sum-of-squares computation — Lanczos1 at 1.4·10⁻²⁵ and
        // Lanczos2 at 2.2·10⁻¹¹ are two such problems — because the synthetic responses were
        // generated at machine epsilon. For those targets the solver's achievable SSE bottom
        // is dominated by ½·Σ(ε·yᵢ)² ≈ 10⁻²⁰, so a relative SSE check against a <10⁻¹⁸
        // certified value reduces to a comparison between two floating-point noise floors.
        // We assert an absolute ceiling of 10⁻¹⁸ instead; parameter agreement (checked below)
        // remains the authoritative convergence signal for those cases. Decision Rules §6 —
        // documented intrinsic precision issue, not algorithm bug.
        if (data.CertifiedSse < 1e-18)
        {
            sse.Should().BeLessThan(1e-18,
                because: $"NIST certified SSE = {data.CertifiedSse:G6} lies below double precision's practical floor; solver must at least reach 1e-18. Reached {sse:G6} {diag}");
        }
        else
        {
            var sseRelativeError = Math.Abs(sse - data.CertifiedSse) / data.CertifiedSse;
            sseRelativeError.Should().BeLessThan(sseTolerance,
                because: $"NIST certified SSE = {data.CertifiedSse:G6}; solver reached {sse:G6} (relative error {sseRelativeError:P2}) {diag}");
        }

        for (var i = 0; i < data.Certified.Length; i++)
        {
            var certified = data.Certified[i];
            var estimate = result.Parameters[i];
            // Guard against certified parameters that land exactly on zero — NIST publishes a
            // handful (e.g. ENSO's trigonometric offsets), and division-by-zero would mask the
            // real agreement. Compare absolutely when the target is zero; otherwise relatively.
            var rel = certified == 0.0
                ? Math.Abs(estimate)
                : Math.Abs((estimate - certified) / certified);
            rel.Should().BeLessThan(parameterTolerance,
                because: $"b{i + 1}: expected {certified:G6}, got {estimate:G6} (error {rel:G4}) {diag}");
        }
    }
}
