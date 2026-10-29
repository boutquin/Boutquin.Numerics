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

using System.Reflection;

using Boutquin.Numerics.Statistics;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

/// <summary>
/// Unit tests for <see cref="GraphicalLassoSolver{T}"/> covering all acceptance criteria
/// from spec N-3.
/// </summary>
public sealed class GraphicalLassoSolverTests
{
    [Theory]
    [InlineData(10000, 0.001)]
    [InlineData(0.0001, 1)]
    public void Solve_ConvergenceUsesRelativeMeanCovarianceChange(double scale, double tolerance)
    {
        var covariance = new double[,] { { scale, 0.3 * scale }, { 0.3 * scale, scale } };
        var result = GraphicalLassoSolver<double>.Solve(covariance, 0.1 * scale,
            new GraphicalLassoOptions<double>(MaxIterations: 1, Tolerance: tolerance));
        // Only the two offdiagonals change, by .1*scale each; mean=.05*scale.
        result.Converged.Should().Be(0.05 < tolerance * 0.3);
    }

    [Theory]
    [InlineData(70_000)]
    [InlineData(int.MaxValue)]
    public void SolveWithBic_HalfLargeObservationCount_SelectsUnpenalizedFit(int count)
    {
        Half[,] covariance = { { (Half)1, (Half)0.5 }, { (Half)0.5, (Half)1 } };
        var solve = () => GraphicalLassoSolver<Half>.SolveWithBic(
            covariance, count, [(Half)0, (Half)0.5]);

        solve.Should().NotThrow("positive int counts are valid for every supported numeric type")
            .Which.Lambda.Should().Be((Half)0);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void SolverEntryPoints_RejectNonFiniteTolerance(double tolerance)
    {
        double[,] sampleCovariance =
        {
            { 1.0, 0.2 },
            { 0.2, 1.0 },
        };
        var options = new GraphicalLassoOptions<double>(Tolerance: tolerance);

        FluentActions.Invoking(() => GraphicalLassoSolver<double>.Solve(sampleCovariance, 0.1, options))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName("options");
        FluentActions.Invoking(() => GraphicalLassoSolver<double>.SolveWithBic(
                sampleCovariance, nObservations: 20, lambdaGrid: [0.1], options))
            .Should().Throw<ArgumentOutOfRangeException>().WithParameterName("options");
    }

    // -----------------------------------------------------------------------
    //  AC-1: Known sparse-precision model — sparsity pattern recovery
    // -----------------------------------------------------------------------

    /// <summary>
    /// AC-1: On a 3×3 known sparse-precision Gaussian model, recovers the sparsity
    /// pattern (zero entries of Θ where true Θ has zeros) within PrecisionNumeric at
    /// appropriate λ.
    /// </summary>
    [Fact]
    public void Solve_RecoversSparsePatternOnKnown3x3Model()
    {
        // True precision matrix: diagonal entries 2.0, off-diagonal (0,1)=0.5 only.
        // S is the corresponding covariance (inverse of precision).
        // We use S directly as input sample covariance for this unit test.
        //
        // True precision: [[2.0, -0.5, 0.0],
        //                  [-0.5, 2.0, 0.0],
        //                  [0.0,  0.0, 2.0]]
        // True covariance = inv(Theta): close to [[0.5333, 0.1333, 0],
        //                                          [0.1333, 0.5333, 0],
        //                                          [0,      0,      0.5]]
        var s = new double[,]
        {
            { 0.5333, 0.1333, 0.0 },
            { 0.1333, 0.5333, 0.0 },
            { 0.0,    0.0,    0.5 }
        };

        // Use the double[,] and cast to avoid generic syntax noise in tests
        var sampleCov = ToDoubleArray(s);

        // At lambda=0.05 the (0,2) and (1,2) entries of true Theta are zero —
        // glasso should shrink them to exactly zero. Lambda must be below |s[0,1]|=0.1333
        // to preserve the non-zero entry but above |s[0,2]|=0.0 to zero the sparse entries.
        var result = GraphicalLassoSolver<double>.Solve(
            sampleCov, lambda: 0.05,
            new GraphicalLassoOptions<double>(MaxIterations: 200, Tolerance: 1e-8));

        result.Converged.Should().BeTrue("algorithm should converge in 200 iterations");

        // Off-diagonal entries (0,2) and (1,2) should be zeroed out by the penalty
        Math.Abs(result.Precision[0, 2]).Should().BeLessThan(1e-6, "theta[0,2] should be zero");
        Math.Abs(result.Precision[1, 2]).Should().BeLessThan(1e-6, "theta[1,2] should be zero");

        // The non-zero entry (0,1) should remain non-zero
        Math.Abs(result.Precision[0, 1]).Should().BeGreaterThan(0.01, "theta[0,1] should be non-zero");
    }

    // -----------------------------------------------------------------------
    //  AC-2: PD invariant — Θ ≻ 0 for λ ≥ 0
    // -----------------------------------------------------------------------

    /// <summary>
    /// AC-2: PD invariant — Θ̂ ≻ 0 (all eigenvalues > 0) for λ ≥ 0 on a 4×4 panel.
    /// </summary>
    [Fact]
    public void Solve_PrecisionIsPositiveDefinite_ForRepresentative4x4()
    {
        // A well-conditioned 4×4 sample covariance
        var s = new double[,]
        {
            { 1.0,  0.5,  0.2,  0.1 },
            { 0.5,  1.0,  0.3,  0.2 },
            { 0.2,  0.3,  1.0,  0.4 },
            { 0.1,  0.2,  0.4,  1.0 }
        };

        var result = GraphicalLassoSolver<double>.Solve(
            s, lambda: 0.1,
            new GraphicalLassoOptions<double>(MaxIterations: 200, Tolerance: 1e-8));

        result.Converged.Should().BeTrue();

        // All eigenvalues > 0 ⟹ PD. We compute eigenvalues via Jacobi.
        var n = 4;
        var eigenvalues = ComputeEigenvaluesSymmetric(result.Precision, n);
        eigenvalues.All(ev => ev > 0).Should().BeTrue(
            $"all precision eigenvalues must be positive; got [{string.Join(", ", eigenvalues.Select(v => v.ToString("G6", System.Globalization.CultureInfo.InvariantCulture)))}]");
    }

    // -----------------------------------------------------------------------
    //  AC-3: Large λ → approximately diagonal precision
    // -----------------------------------------------------------------------

    /// <summary>
    /// AC-3: λ → large produces approximately diagonal Θ (off-diagonal entries near 0).
    /// </summary>
    [Fact]
    public void Solve_LargeLambdaProducesDiagonalPrecision()
    {
        var s = new double[,]
        {
            { 1.0,  0.8,  0.6 },
            { 0.8,  1.0,  0.7 },
            { 0.6,  0.7,  1.0 }
        };

        var result = GraphicalLassoSolver<double>.Solve(
            s, lambda: 10.0,
            new GraphicalLassoOptions<double>(MaxIterations: 500, Tolerance: 1e-8));

        // All off-diagonal entries should be near zero
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                if (i != j)
                {
                    Math.Abs(result.Precision[i, j]).Should().BeLessThan(1e-6,
                        $"theta[{i},{j}] should be ~0 at very large lambda");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    //  AC-4: SolveWithBic selects valid λ minimising BIC
    // -----------------------------------------------------------------------

    /// <summary>
    /// AC-4: <see cref="GraphicalLassoSolver{T}.SolveWithBic"/> selects a valid λ from the grid
    /// (the one minimizing BIC).
    /// </summary>
    [Fact]
    public void SolveWithBic_SelectsValidLambdaFromGrid()
    {
        var s = new double[,]
        {
            { 1.0,  0.5,  0.1 },
            { 0.5,  1.0,  0.3 },
            { 0.1,  0.3,  1.0 }
        };

        var lambdaGrid = new[] { 0.01, 0.05, 0.1, 0.2, 0.5, 1.0 };
        var result = GraphicalLassoSolver<double>.SolveWithBic(
            s, nObservations: 50, lambdaGrid: lambdaGrid,
            new GraphicalLassoOptions<double>(MaxIterations: 200, Tolerance: 1e-8));

        // The selected lambda must come from the grid
        lambdaGrid.Should().Contain(result.Lambda,
            "BIC selection must pick a lambda from the provided grid");

        // The result must still be PD
        var n = 3;
        var eigenvalues = ComputeEigenvaluesSymmetric(result.Precision, n);
        eigenvalues.All(ev => ev > 0).Should().BeTrue("BIC-selected precision must be PD");
    }

    // -----------------------------------------------------------------------
    //  AC-5: Non-symmetric input → ArgumentException
    // -----------------------------------------------------------------------

    /// <summary>
    /// AC-5: Non-symmetric input → <see cref="ArgumentException"/>.
    /// </summary>
    [Fact]
    public void Solve_ThrowsArgumentException_ForNonSymmetricInput()
    {
        var s = new double[,]
        {
            { 1.0, 0.5 },
            { 0.2, 1.0 }  // not symmetric: s[1,0]=0.2 ≠ s[0,1]=0.5
        };

        var act = () => GraphicalLassoSolver<double>.Solve(s, lambda: 0.1);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*symmetric*");
    }

    [Fact]
    public void HalfEntryPoints_RejectMaterialSmallScaleAsymmetry()
    {
        var covariance = new Half[,]
        {
            { (Half)0.00002, (Half)0.0000005 },
            { (Half)0,       (Half)0.00002 }
        };

        foreach (var reverseTriangles in new[] { false, true })
        {
            if (reverseTriangles)
            {
                (covariance[0, 1], covariance[1, 0]) = (covariance[1, 0], covariance[0, 1]);
            }

            FluentActions.Invoking(() => GraphicalLassoSolver<Half>.Solve(covariance, (Half)0))
                .Should().Throw<ArgumentException>().WithMessage("*symmetric*")
                .WithParameterName("sampleCovariance");
            FluentActions.Invoking(() => GraphicalLassoSolver<Half>.SolveWithBic(
                    covariance, nObservations: 20, lambdaGrid: [(Half)0]))
                .Should().Throw<ArgumentException>().WithMessage("*symmetric*")
                .WithParameterName("sampleCovariance");
        }
    }

    // -----------------------------------------------------------------------
    //  AC-6: λ < 0 → ArgumentOutOfRangeException
    // -----------------------------------------------------------------------

    /// <summary>
    /// AC-6: λ &lt; 0 → <see cref="ArgumentOutOfRangeException"/>.
    /// </summary>
    [Fact]
    public void Solve_ThrowsArgumentOutOfRangeException_ForNegativeLambda()
    {
        var s = new double[,]
        {
            { 1.0, 0.3 },
            { 0.3, 1.0 }
        };

        var act = () => GraphicalLassoSolver<double>.Solve(s, lambda: -0.1);
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("lambda");
    }

    // -----------------------------------------------------------------------
    //  Additional: zero lambda → result ≈ inverse of sample covariance
    // -----------------------------------------------------------------------

    /// <summary>
    /// At λ = 0 the graphical lasso is unconstrained and should return Θ ≈ S⁻¹.
    /// </summary>
    [Fact]
    public void Solve_ZeroLambda_ReturnsSampleCovarianceInverse()
    {
        // Simple 2×2 case: S = [[2, 1],[1, 2]], S⁻¹ = [[2/3, -1/3],[-1/3, 2/3]]
        var s = new double[,]
        {
            { 2.0, 1.0 },
            { 1.0, 2.0 }
        };

        var result = GraphicalLassoSolver<double>.Solve(
            s, lambda: 0.0,
            new GraphicalLassoOptions<double>(MaxIterations: 500, Tolerance: 1e-9));

        // Expected: 1/3 ≈ 0.3333, 2/3 ≈ 0.6667
        result.Precision[0, 0].Should().BeApproximately(2.0 / 3.0, 1e-5, "theta[0,0]");
        result.Precision[0, 1].Should().BeApproximately(-1.0 / 3.0, 1e-5, "theta[0,1]");
        result.Precision[1, 0].Should().BeApproximately(-1.0 / 3.0, 1e-5, "theta[1,0]");
        result.Precision[1, 1].Should().BeApproximately(2.0 / 3.0, 1e-5, "theta[1,1]");
    }

    // -----------------------------------------------------------------------
    //  Additional: Covariance field = inverse of Precision
    // -----------------------------------------------------------------------

    [Fact]
    public void Solve_CovarianceIsInverseOfPrecision()
    {
        var s = new double[,]
        {
            { 1.0,  0.4 },
            { 0.4,  1.0 }
        };

        var result = GraphicalLassoSolver<double>.Solve(
            s, lambda: 0.05,
            new GraphicalLassoOptions<double>(MaxIterations: 200, Tolerance: 1e-8));

        // Θ · Σ should equal I
        var n = 2;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                double sum = 0;
                for (var k = 0; k < n; k++)
                {
                    sum += result.Precision[i, k] * result.Covariance[k, j];
                }
                var expected = i == j ? 1.0 : 0.0;
                Math.Abs(sum - expected).Should().BeLessThan(1e-5,
                    $"(Theta*Sigma)[{i},{j}] should equal {expected}");
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Additional: Weighted penalty respected
    // -----------------------------------------------------------------------

    [Fact]
    public void Solve_WeightedPenalty_ReducesShrinkageOnLowerWeightEntry()
    {
        // S has strong correlation between (0,1). With uniform weights, (0,1) gets shrunk.
        // With weight[0,1]=0.01 (very small) and weight[0,2]=1.0, entry (0,1) should stay
        // larger than with uniform weights.
        var s = new double[,]
        {
            { 1.0,  0.7,  0.1 },
            { 0.7,  1.0,  0.1 },
            { 0.1,  0.1,  1.0 }
        };

        var weights = new double[,]
        {
            { 0.0, 0.01, 1.0 },
            { 0.01, 0.0, 1.0 },
            { 1.0,  1.0, 0.0 }
        };

        var resultUniform = GraphicalLassoSolver<double>.Solve(
            s, lambda: 0.3,
            new GraphicalLassoOptions<double>(MaxIterations: 200, Tolerance: 1e-8));

        var resultWeighted = GraphicalLassoSolver<double>.Solve(
            s, lambda: 0.3,
            new GraphicalLassoOptions<double>(MaxIterations: 200, Tolerance: 1e-8, Weights: weights));

        // With very small weight on (0,1), the weighted solver should leave theta[0,1]
        // less shrunk (larger magnitude) than the uniform solver
        Math.Abs(resultWeighted.Precision[0, 1]).Should().BeGreaterThan(
            Math.Abs(resultUniform.Precision[0, 1]),
            "lower penalty weight on (0,1) means less shrinkage → larger magnitude");
    }

    // -----------------------------------------------------------------------
    //  F-L5-1: nObservations ≤ 0 → ArgumentOutOfRangeException
    // -----------------------------------------------------------------------

    /// <summary>
    /// F-L5-1: nObservations=0 must throw ArgumentOutOfRangeException.
    /// Without the guard, Log(0)=-∞ corrupts BIC and the k=0 path yields NaN,
    /// leaving bestResult null → misleading InvalidOperationException.
    /// </summary>
    [Fact]
    public void SolveWithBic_ThrowsArgumentOutOfRangeException_ForZeroObservations()
    {
        var s = new double[,]
        {
            { 1.0, 0.3 },
            { 0.3, 1.0 }
        };
        var lambdaGrid = new[] { 0.1, 0.5 };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(s, nObservations: 0, lambdaGrid: lambdaGrid);
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("nObservations");
    }

    /// <summary>
    /// F-L5-1: nObservations=-1 must throw ArgumentOutOfRangeException.
    /// Log(-1) = NaN, which silently picks the first k>0 lambda.
    /// </summary>
    [Fact]
    public void SolveWithBic_ThrowsArgumentOutOfRangeException_ForNegativeObservations()
    {
        var s = new double[,]
        {
            { 1.0, 0.3 },
            { 0.3, 1.0 }
        };
        var lambdaGrid = new[] { 0.1, 0.5 };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(s, nObservations: -1, lambdaGrid: lambdaGrid);
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("nObservations");
    }

    // -----------------------------------------------------------------------
    //  F-L1-c3-01: zero diagonal → non-positive-definite input rejection
    // -----------------------------------------------------------------------

    /// <summary>
    /// F-L1-c3-01: SolveWithBic must reject a zero-variance column as a
    /// non-positive-definite covariance matrix, with a public argument error rather
    /// than a low-level failure from inversion.
    /// </summary>
    [Fact]
    public void SolveWithBic_ThrowsArgumentException_ForZeroVarianceColumn()
    {
        // Column 1 has zero variance: S[1,1] = 0
        var s = new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 1.0 }
        };
        var lambdaGrid = new[] { 0.1, 0.5 };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(s, nObservations: 50, lambdaGrid: lambdaGrid);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*positive definite*")
            .WithParameterName("sampleCovariance");
    }

    [Fact]
    public void Solve_SmallScalePositiveDefiniteCovariance_IsAccepted()
    {
        var covariance = new double[,]
        {
            { 1e-12, 2e-13 },
            { 2e-13, 1e-12 }
        };

        var result = GraphicalLassoSolver<double>.Solve(
            covariance,
            lambda: 0.0,
            new GraphicalLassoOptions<double>(Tolerance: 1e-6));

        result.Converged.Should().BeTrue();
        result.Precision[0, 0].Should().BePositive();
        result.Precision[1, 1].Should().BePositive();
        result.Precision.Cast<double>().Should().AllSatisfy(
            value => double.IsFinite(value).Should().BeTrue());
    }

    // -----------------------------------------------------------------------
    //  F-L5-2: Weights wrong dimension → ArgumentException from Solve/SolveWithBic
    // -----------------------------------------------------------------------

    /// <summary>
    /// F-L5-2: Solve with a weights matrix smaller than p×p must throw ArgumentException,
    /// not IndexOutOfRangeException from SolveLassoSubproblem.
    /// </summary>
    [Fact]
    public void Solve_ThrowsArgumentException_ForWrongSizedWeights()
    {
        var s = new double[,]
        {
            { 1.0, 0.3 },
            { 0.3, 1.0 }
        };
        // Weights is 1×1 but p=2 — accessing weights[1, 0] would go out of bounds
        var weights = new double[,] { { 1.0 } };
        var options = new GraphicalLassoOptions<double>(Weights: weights);
        var act = () => GraphicalLassoSolver<double>.Solve(s, lambda: 0.1, options);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*Weights*");
    }

    /// <summary>
    /// F-L5-2: SolveWithBic with a weights matrix smaller than p×p must throw ArgumentException.
    /// </summary>
    [Fact]
    public void SolveWithBic_ThrowsArgumentException_ForWrongSizedWeights()
    {
        var s = new double[,]
        {
            { 1.0, 0.3 },
            { 0.3, 1.0 }
        };
        var weights = new double[,] { { 1.0 } };
        var options = new GraphicalLassoOptions<double>(Weights: weights);
        var lambdaGrid = new[] { 0.1 };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(s, nObservations: 10, lambdaGrid: lambdaGrid, options: options);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*Weights*");
    }

    // -----------------------------------------------------------------------
    //  F-L5-3: p=0 → ArgumentException on both Solve and SolveWithBic
    // -----------------------------------------------------------------------

    /// <summary>
    /// F-L5-3: An empty (0×0) covariance matrix passed to Solve must throw
    /// ArgumentException, not OverflowException from new T[-1] in SolveCore.
    /// </summary>
    [Fact]
    public void Solve_ThrowsArgumentException_ForEmptyMatrix()
    {
        var s = new double[0, 0];
        var act = () => GraphicalLassoSolver<double>.Solve(s, lambda: 0.1);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*empty*");
    }

    /// <summary>
    /// F-L5-3: An empty (0×0) covariance matrix passed to SolveWithBic must throw
    /// ArgumentException, not OverflowException from new T[-1] in SolveCore.
    /// </summary>
    [Fact]
    public void SolveWithBic_ThrowsArgumentException_ForEmptyMatrix()
    {
        var s = new double[0, 0];
        var lambdaGrid = new[] { 0.1 };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(s, nObservations: 10, lambdaGrid: lambdaGrid);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*empty*");
    }

    // -----------------------------------------------------------------------
    //  M-2 AC-1: BIC selection uses the documented (n/2)·log-likelihood scaling
    // -----------------------------------------------------------------------

    /// <summary>
    /// M-2 AC-1: <see cref="GraphicalLassoSolver{T}.SolveWithBic"/> must weight the
    /// Gaussian log-likelihood by n/2 — i.e. select λ minimising
    /// BIC = −n·(log det Θ − tr(SΘ)) + k·ln(n) — rather than the defective
    /// unscaled criterion −2·(log det Θ − tr(SΘ)) + k·ln(n), which over-weights the
    /// k·ln(n) penalty by a factor of n and systematically over-sparsifies λ.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Independent golden — derived externally with scikit-learn's low-level
    /// <c>sklearn.covariance.graphical_lasso(S, alpha=λ)</c> (v1.8.0), NOT from the C#
    /// under test. For the 3×3 sample covariance below and n = 500, grid
    /// {0.05, 0.1, 0.2} (ln n = 6.214608), each fit gives (log det Θ, tr(SΘ), k) where
    /// k counts BOTH off-diagonal triangles (unique pairs × 2, accepted decision
    /// 2026-06-29-r1-660796ca):
    /// </para>
    /// <para>
    ///   λ=0.05: (1.971786, 2.955056, 2);  λ=0.1: (1.936134, 2.946645, 2);  λ=0.2: (1.915039, 3.000000, 0).
    /// </para>
    /// <para>
    /// CORRECT BIC = −500·(logDet − tr) + k·ln n → {504.064, 517.684, 542.480} → argmin = λ=0.05.
    /// BUGGY  BIC = −2·(logDet − tr) + k·ln n → {14.396, 14.450, 2.170} → argmin = λ=0.2 (over-sparsified).
    /// The C# BCD solver reproduces sklearn's precision at these λ to machine precision
    /// (max abs diff 0.0 at λ=0.05 on this input), so the selected index is robust.
    /// </para>
    /// </remarks>
    [Fact]
    public void SolveWithBic_SelectsNOver2ScaledOptimum_NotOverSparsifiedLambda()
    {
        var s = new double[,]
        {
            { 0.5527969051613634, 0.17991301348430638, -0.04434057971149475 },
            { 0.17991301348430638, 0.5534318751652731, -0.02724217038309842 },
            { -0.04434057971149475, -0.02724217038309842, 0.4815919080349265 }
        };

        var lambdaGrid = new[] { 0.05, 0.1, 0.2 };
        var result = GraphicalLassoSolver<double>.SolveWithBic(
            s, nObservations: 500, lambdaGrid: lambdaGrid,
            new GraphicalLassoOptions<double>(MaxIterations: 500, Tolerance: 1e-8));

        // The n/2-scaled BIC selects λ=0.05 (retains the (0,1) edge, k=2).
        // The defective unscaled BIC selects λ=0.2 (over-sparsified, k=0) — this
        // assertion is RED before the fix and GREEN after.
        result.Lambda.Should().Be(0.05,
            "the documented n/2-scaled BIC criterion selects the least-sparse well-fitting λ, "
            + "not the over-sparsified λ=0.2 the unscaled criterion picks");
    }

    // -----------------------------------------------------------------------
    //  M-2 AC-2: LogDet catch is narrowed to InvalidOperationException
    // -----------------------------------------------------------------------

    /// <summary>
    /// M-2 AC-2 (sentinel preserved): a genuinely non-positive-definite matrix makes
    /// <c>CholeskyDecomposition.Decompose</c> throw <see cref="InvalidOperationException"/>,
    /// which <c>LogDet</c> must still catch and translate to the
    /// <see cref="double.NegativeInfinity"/> sentinel (⇒ BIC = +∞ ⇒ that λ is never
    /// selected). Characterisation test — green before and after the narrowing.
    /// </summary>
    [Fact]
    public void LogDet_NonPdIndefiniteMatrix_ReturnsNegativeInfinitySentinel()
    {
        // [[1,2],[2,1]] is symmetric but indefinite (eigenvalues 3 and −1);
        // Cholesky hits a non-positive Schur diagonal and throws InvalidOperationException.
        var indefinite = new double[,] { { 1.0, 2.0 }, { 2.0, 1.0 } };

        var logDet = InvokeLogDet(indefinite, p: 2);

        logDet.Should().Be(double.NegativeInfinity,
            "a non-PD iterate must yield the NegativeInfinity sentinel so its λ is never BIC-selected");
    }

    /// <summary>
    /// M-2 AC-2 (narrowing discriminator): a programmer error inside the <c>LogDet</c>
    /// try-body — here an <see cref="IndexOutOfRangeException"/> from passing p greater
    /// than the matrix rank — must PROPAGATE, not be swallowed into the sentinel. The
    /// pre-fix bare <c>catch</c> swallows it (returns NegativeInfinity, no throw) → RED;
    /// the narrowed <c>catch (InvalidOperationException)</c> lets it propagate → GREEN.
    /// </summary>
    [Fact]
    public void LogDet_ProgrammerErrorInTryBody_Propagates_NotSwallowedToSentinel()
    {
        // A valid 2×2 PD matrix but p=3: Decompose returns a 2×2 factor, then the
        // log-diagonal loop reads lower[2,2] → IndexOutOfRangeException (a programmer
        // error, NOT InvalidOperationException).
        var pd2x2 = new double[,] { { 2.0, 0.0 }, { 0.0, 2.0 } };

        var act = () => InvokeLogDet(pd2x2, p: 3);

        act.Should().Throw<IndexOutOfRangeException>(
            "a non-InvalidOperationException must propagate once the bare catch is narrowed, "
            + "rather than being silently swallowed into the NegativeInfinity sentinel");
    }

    // -----------------------------------------------------------------------
    //  M-2 AC-3: non-finite covariance entry → ArgumentException naming the parameter
    // -----------------------------------------------------------------------

    /// <summary>
    /// M-2 AC-3: <see cref="GraphicalLassoSolver{T}.Solve"/> must reject a covariance
    /// containing a non-finite (NaN/±∞) entry with an <see cref="ArgumentException"/>
    /// naming <c>sampleCovariance</c>, rather than silently returning a NaN-poisoned result.
    /// </summary>
    [Fact]
    public void Solve_ThrowsArgumentException_ForNonFiniteCovarianceEntry()
    {
        var s = new double[,]
        {
            { 1.0, double.NaN },
            { double.NaN, 1.0 }
        };

        var act = () => GraphicalLassoSolver<double>.Solve(s, lambda: 0.1);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*not finite*")
            .WithParameterName("sampleCovariance");
    }

    /// <summary>
    /// M-2 AC-3: <see cref="GraphicalLassoSolver{T}.SolveWithBic"/> must apply the same
    /// non-finite guard, naming <c>sampleCovariance</c>.
    /// </summary>
    [Fact]
    public void SolveWithBic_ThrowsArgumentException_ForNonFiniteCovarianceEntry()
    {
        var s = new double[,]
        {
            { 1.0, 0.3, 0.0 },
            { 0.3, 1.0, double.PositiveInfinity },
            { 0.0, double.PositiveInfinity, 1.0 }
        };
        var lambdaGrid = new[] { 0.1, 0.5 };

        var act = () => GraphicalLassoSolver<double>.SolveWithBic(
            s, nObservations: 50, lambdaGrid: lambdaGrid);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*not finite*")
            .WithParameterName("sampleCovariance");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Solve_NonFiniteLambda_ThrowsArgumentOutOfRangeException(double lambda)
    {
        var covariance = new double[,] { { 1.0, 0.2 }, { 0.2, 1.0 } };
        var act = () => GraphicalLassoSolver<double>.Solve(covariance, lambda);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName(nameof(lambda));
    }

    [Fact]
    public void Solve_IndefiniteCovariance_ThrowsArgumentException()
    {
        var indefinite = new double[,] { { 1.0, 2.0 }, { 2.0, 1.0 } };
        var act = () => GraphicalLassoSolver<double>.Solve(indefinite, lambda: 0.0);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*positive definite*")
            .WithParameterName("sampleCovariance");
    }

    [Fact]
    public void SolveWithBic_IndefiniteCovariance_ThrowsArgumentException()
    {
        var indefinite = new double[,] { { 1.0, 2.0 }, { 2.0, 1.0 } };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(
            indefinite, nObservations: 100, lambdaGrid: [0.1, 0.2]);

        act.Should().Throw<ArgumentException>()
            .WithMessage("*positive definite*")
            .WithParameterName("sampleCovariance");
    }

    [Theory]
    [InlineData(-1.0, -1.0)]
    [InlineData(double.NaN, double.NaN)]
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData(0.1, 0.2)]
    public void Solve_InvalidPenaltyWeights_ThrowsArgumentException(double upper, double lower)
    {
        var covariance = new double[,] { { 1.0, 0.2 }, { 0.2, 1.0 } };
        var weights = new double[,] { { 0.0, upper }, { lower, 0.0 } };
        var options = new GraphicalLassoOptions<double>(Weights: weights);
        var act = () => GraphicalLassoSolver<double>.Solve(covariance, 0.1, options);

        act.Should().Throw<ArgumentException>()
            .WithParameterName("options");
    }

    [Theory]
    [InlineData(0.0, 100)]
    [InlineData(-1e-4, 100)]
    [InlineData(1e-4, 0)]
    [InlineData(1e-4, -1)]
    public void Solve_NonPositiveToleranceOrIterationLimit_ThrowsArgumentOutOfRangeException(
        double tolerance,
        int maxIterations)
    {
        var covariance = new double[,] { { 1.0, 0.2 }, { 0.2, 1.0 } };
        var options = new GraphicalLassoOptions<double>(
            MaxIterations: maxIterations,
            Tolerance: tolerance);
        var act = () => GraphicalLassoSolver<double>.Solve(covariance, 0.1, options);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("options");
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-0.1)]
    public void SolveWithBic_InvalidLambdaGridEntry_ThrowsArgumentOutOfRangeException(double lambda)
    {
        var covariance = new double[,] { { 1.0, 0.2 }, { 0.2, 1.0 } };
        var act = () => GraphicalLassoSolver<double>.SolveWithBic(
            covariance, nObservations: 100, lambdaGrid: [0.1, lambda]);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*index 1*")
            .WithParameterName("lambdaGrid");
    }

    // -----------------------------------------------------------------------
    //  Helpers
    // -----------------------------------------------------------------------

    private static double[,] ToDoubleArray(double[,] src) => src;

    /// <summary>
    /// Invokes the private <c>GraphicalLassoSolver&lt;double&gt;.LogDet(double[,], int)</c>
    /// via reflection and unwraps <see cref="TargetInvocationException"/> so the
    /// underlying exception surfaces directly to the assertion.
    /// </summary>
    private static double InvokeLogDet(double[,] a, int p)
    {
        var method = typeof(GraphicalLassoSolver<double>).GetMethod(
            "LogDet", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("LogDet method not found via reflection.");

        try
        {
            return (double)method.Invoke(null, [a, p])!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            throw ex.InnerException;
        }
    }

    [Fact]
    public void PsdOracle_EqualDiagonalIndefiniteMatrix_HasNegativeEigenvalue()
    {
        double[,] matrix = { { 1, 2 }, { 2, 1 } };
        ComputeEigenvaluesSymmetric(matrix, 2).Min().Should().BeApproximately(-1, 1e-12);
    }

    /// <summary>
    /// Power-iteration based eigenvalue estimation for small symmetric matrices.
    /// Uses Jacobi sweeps for correctness.
    /// </summary>
    private static double[] ComputeEigenvaluesSymmetric(double[,] m, int n)
    {
        // Copy matrix for in-place Jacobi
        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                a[i, j] = m[i, j];
            }
        }

        // Simple Jacobi iteration (sufficient for small n)
        for (var sweep = 0; sweep < 100; sweep++)
        {
            for (var p = 0; p < n - 1; p++)
            {
                for (var q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p, q]) < 1e-15)
                    {
                        continue;
                    }

                    var theta = (a[q, q] - a[p, p]) / (2.0 * a[p, q]);
                    var t = (theta >= 0 ? 1.0 : -1.0) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    var c = 1.0 / Math.Sqrt(1.0 + t * t);
                    var s2 = t * c;

                    // Update rows/cols p and q
                    var app = a[p, p];
                    var aqq = a[q, q];
                    var apq = a[p, q];

                    a[p, p] = c * c * app - 2 * s2 * c * apq + s2 * s2 * aqq;
                    a[q, q] = s2 * s2 * app + 2 * s2 * c * apq + c * c * aqq;
                    a[p, q] = 0.0;
                    a[q, p] = 0.0;

                    for (var r = 0; r < n; r++)
                    {
                        if (r == p || r == q)
                        {
                            continue;
                        }

                        var arp = a[r, p];
                        var arq = a[r, q];
                        a[r, p] = c * arp - s2 * arq;
                        a[p, r] = a[r, p];
                        a[r, q] = s2 * arp + c * arq;
                        a[q, r] = a[r, q];
                    }
                }
            }
        }

        var eigenvalues = new double[n];
        for (var i = 0; i < n; i++)
        {
            eigenvalues[i] = a[i, i];
        }

        return eigenvalues;
    }
}
