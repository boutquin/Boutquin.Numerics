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

using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// TDD review-fix batch for the LinearAlgebra and Interpolation clusters
/// (findings S1, S2, S19, S20; coverage gaps B2, B15–B19, B24, B25).
/// </summary>
public sealed class LinAlgInterpReviewFixTests
{
    // ───────────────────────────────────────────────────────────────────────
    // S1 — GaussianElimination singularity tested against column MAX, not the
    // randomly-selected pivot row.
    // ───────────────────────────────────────────────────────────────────────

    /// <summary>
    /// A non-singular system whose pivot column maximum sits just above the
    /// relative pivot floor must solve deterministically for every seed. Before
    /// the fix, <c>SelectPivotRow</c> could randomly choose a candidate as low as
    /// <c>0.5·maxVal</c> — below the pivot floor — and the singularity test on
    /// that randomly-chosen row threw "singular" for some seeds only.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(11)]
    [InlineData(42)]
    [InlineData(99)]
    [InlineData(123)]
    [InlineData(2024)]
    public void GaussianElimination_ColumnMaxJustAbovePivotFloor_SolvesForAllSeeds(int seed)
    {
        // matrixScale = 1.0 (set by the [1] entry below) => pivotFloor = 1e-14.
        // Column 0 below the diagonal holds two eligible candidates whose magnitudes
        // straddle pivotFloor: max = 1.5e-14 (> floor), the other = 0.9e-14 (< floor)
        // but >= 0.5·max = 0.75e-14, so it is in the GERCP candidate set. A pre-fix
        // SelectPivotRow that returns the small candidate trips the singularity guard.
        const double a00 = 1.5e-14;
        const double a10 = 0.9e-14;

        var a = new double[,]
        {
            { a00, 0.0, 0.0 },
            { a10, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
        };
        var b = new double[] { a00, a10 + 2.0, 3.0 };

        var act = () => GaussianElimination<double>.Solve(a, b, seed);

        act.Should().NotThrow(
            $"the column max {a00} exceeds the pivot floor, so the system is non-singular regardless of seed {seed}");

        // The system solves for every seed (the bug was a seed-dependent "singular" throw). The
        // first unknown rides on a near-floor pivot so its conditioning is poor — a loose bound is
        // appropriate; x[1] and x[2] are well-conditioned and must be exact.
        var x = GaussianElimination<double>.Solve(a, b, seed);
        x.Should().OnlyContain(v => double.IsFinite(v));
        x[0].Should().BeApproximately(1.0, 0.1);
        x[1].Should().BeApproximately(2.0, 1e-9);
        x[2].Should().BeApproximately(3.0, 1e-9);
    }

    /// <summary>
    /// A genuinely singular column (column max below the pivot floor) must still throw.
    /// </summary>
    [Fact]
    public void GaussianElimination_ColumnMaxBelowPivotFloor_ThrowsSingular()
    {
        // matrixScale = 1.0 => pivotFloor = 1e-14. Column 0 max is 1e-15 < floor.
        var a = new double[,]
        {
            { 1e-15, 0.0, 0.0 },
            { 1e-16, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
        };
        var b = new double[] { 1e-15, 1.0, 1.0 };

        var act = () => GaussianElimination<double>.Solve(a, b, seed: 1);
        act.Should().Throw<InvalidOperationException>();
    }

    // ───────────────────────────────────────────────────────────────────────
    // S2 — DecomposePivoted accepts the pivot diagonal from the maintained
    // Schur value (diagRemaining[j]), not the un-updated work[j,j].
    // ───────────────────────────────────────────────────────────────────────

    /// <summary>
    /// A genuinely rank-2 PSD matrix requiring a non-identity permutation. The accepted pivot
    /// diagonal must be the maintained Schur value <c>diagRemaining[j]</c>; reading the un-updated
    /// <c>work[j,j]</c> redundantly re-derives it from the original diagonal and risks a roundoff
    /// divergence in the rank decision. The factor must report rank 2 and reconstruct
    /// <c>L·Lᵀ == PᵀAP</c> exactly. (The two diagonal expressions are algebraically identical, so
    /// a deterministic RED is not constructible; this test pins the post-fix contract.)
    /// </summary>
    [Fact]
    public void DecomposePivoted_RankDeficientNonIdentityPermutation_ReconstructsInput()
    {
        // A = B·Bᵀ with B = [[1,0],[1,1],[1,2]] (3×2, rank 2). diag (1, 2, 5) is non-monotone, so
        // the first pivot selects index 2 (largest) — a non-identity permutation.
        var a = new double[,]
        {
            { 1.0, 1.0, 1.0 },
            { 1.0, 2.0, 3.0 },
            { 1.0, 3.0, 5.0 },
        };

        var result = CholeskyDecomposition<double>.DecomposePivoted(a, tolerance: 1e-10, maxRank: null);

        result.Rank.Should().Be(2, "the matrix is exactly rank-2");
        result.Permutation.Should().NotEqual(new[] { 0, 1, 2 }, "the pivot order is non-identity");

        // Reconstruct L·Lᵀ and compare against the permuted A.
        const int n = 3;
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var sum = 0.0;
                for (var k = 0; k < result.Rank; k++)
                {
                    sum += result.Lower[i, k] * result.Lower[j, k];
                }

                var pi = result.Permutation[i];
                var pj = result.Permutation[j];
                sum.Should().BeApproximately(a[pi, pj], 1e-9, $"reconstruction at [{i},{j}]");
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // S19 — Cholesky.Solve guards against a near-zero factor diagonal.
    // ───────────────────────────────────────────────────────────────────────

    /// <summary>
    /// <c>Solve</c> divides by <c>lower[i,i]</c>. A zero on the factor diagonal must
    /// produce a clear exception rather than a silent NaN/Infinity solution.
    /// </summary>
    [Fact]
    public void CholeskySolve_ZeroFactorDiagonal_ThrowsClearException()
    {
        // A lower-triangular factor with a zero diagonal entry (rank-deficient factor).
        var lower = new double[,]
        {
            { 2.0, 0.0, 0.0 },
            { 1.0, 0.0, 0.0 },
            { 1.0, 1.0, 3.0 },
        };
        var b = new double[] { 2.0, 1.0, 5.0 };

        var act = () => CholeskyDecomposition<double>.Solve(lower, b);
        act.Should().Throw<InvalidOperationException>();
    }

    // ───────────────────────────────────────────────────────────────────────
    // S20 — Jacobi "already-diagonal" fast path threshold is a real one (~eps).
    // Verify eigen results are unchanged for representative matrices.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Jacobi_NearlyDiagonalMatrix_ProducesCorrectEigenvalues()
    {
        // Diagonal-dominant with tiny off-diagonals; the fast path must not corrupt results.
        var matrix = new double[,]
        {
            { 5.0, 1e-13, 0.0 },
            { 1e-13, 3.0, 1e-13 },
            { 0.0, 1e-13, 1.0 },
        };

        var result = JacobiEigenDecomposition<double>.Decompose(matrix);

        result.Values[0].Should().BeApproximately(5.0, 1e-9);
        result.Values[1].Should().BeApproximately(3.0, 1e-9);
        result.Values[2].Should().BeApproximately(1.0, 1e-9);
    }

    // ───────────────────────────────────────────────────────────────────────
    // B2 — InterpolatorFactory<T>.Create coverage.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Factory_Generic_Linear_ReturnsLinearInterpolator()
    {
        var interp = InterpolatorFactory<double>.Create(InterpolatorKind.Linear);
        interp.Should().BeSameAs(LinearInterpolator<double>.Instance);
        interp.Name.Should().Be("Linear");
    }

    [Fact]
    public void Factory_Generic_MonotoneCubic_ReturnsMonotoneCubicInterpolator()
    {
        var interp = InterpolatorFactory<double>.Create(InterpolatorKind.MonotoneCubic);
        interp.Should().BeSameAs(MonotoneCubicInterpolator<double>.Instance);
    }

    [Fact]
    public void Factory_Generic_MonotoneConvex_ReturnsMonotoneConvexInterpolator()
    {
        var interp = InterpolatorFactory<double>.Create(InterpolatorKind.MonotoneConvex);
        interp.Should().BeSameAs(MonotoneConvexInterpolator<double>.Instance);
    }

    [Fact]
    public void Factory_Generic_LogLinear_Double_ReturnsLogLinearInterpolator()
    {
        var interp = InterpolatorFactory<double>.Create(InterpolatorKind.LogLinear);
        interp.Name.Should().Be("LogLinear");
    }

    [Fact]
    public void Factory_Generic_FlatForward_Float_ReturnsFlatForwardInterpolator()
    {
        var interp = InterpolatorFactory<float>.Create(InterpolatorKind.FlatForward);
        interp.Name.Should().Be("FlatForward");
    }

    [Fact]
    public void Factory_Generic_LogLinear_NonIeee_Throws()
    {
        // decimal does not implement IFloatingPointIeee754<T>.
        var act = () => InterpolatorFactory<decimal>.Create(InterpolatorKind.LogLinear);
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void Factory_Generic_FlatForward_NonIeee_Throws()
    {
        var act = () => InterpolatorFactory<decimal>.Create(InterpolatorKind.FlatForward);
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void Factory_Generic_CubicSpline_ThrowsNotSupported()
    {
        var act = () => InterpolatorFactory<double>.Create(InterpolatorKind.CubicSpline);
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void Factory_Generic_UnknownKind_Throws()
    {
        var act = () => InterpolatorFactory<double>.Create((InterpolatorKind)999);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ───────────────────────────────────────────────────────────────────────
    // B15 — CholeskyDecomposition<T> guards: non-square, length mismatch, nulls.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Cholesky_Decompose_NonSquare_Throws()
    {
        var a = new double[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
        var act = () => CholeskyDecomposition<double>.Decompose(a);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Cholesky_Decompose_Null_Throws()
    {
        var act = () => CholeskyDecomposition<double>.Decompose(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Cholesky_Solve_LengthMismatch_Throws()
    {
        var lower = new double[,] { { 2.0, 0.0 }, { 1.0, 3.0 } };
        var b = new double[] { 1.0, 2.0, 3.0 };
        var act = () => CholeskyDecomposition<double>.Solve(lower, b);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Cholesky_Solve_NullLower_Throws()
    {
        var act = () => CholeskyDecomposition<double>.Solve(null!, new double[] { 1.0 });
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Cholesky_Solve_NullB_Throws()
    {
        var lower = new double[,] { { 1.0 } };
        var act = () => CholeskyDecomposition<double>.Solve(lower, null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Cholesky_DecomposePivoted_Null_Throws()
    {
        var act = () => CholeskyDecomposition<double>.DecomposePivoted(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Cholesky_DecomposePivoted_NonSquare_Throws()
    {
        var a = new double[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
        var act = () => CholeskyDecomposition<double>.DecomposePivoted(a, 0.0, null);
        act.Should().Throw<ArgumentException>();
    }

    // ───────────────────────────────────────────────────────────────────────
    // B16 — GaussianElimination<T> non-square throw; 1×1 and all-zero paths.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Gaussian_NonSquare_Throws()
    {
        var a = new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
        var b = new double[] { 1.0, 2.0 };
        var act = () => GaussianElimination<double>.Solve(a, b, seed: 1);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Gaussian_LengthMismatch_Throws()
    {
        var a = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var b = new double[] { 1.0 };
        var act = () => GaussianElimination<double>.Solve(a, b, seed: 1);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Gaussian_Null_Throws()
    {
        var actA = () => GaussianElimination<double>.Solve(null!, new double[] { 1.0 }, seed: 1);
        var actB = () => GaussianElimination<double>.Solve(new double[,] { { 1.0 } }, null!, seed: 1);
        actA.Should().Throw<ArgumentNullException>();
        actB.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Gaussian_OneByOne_Solves()
    {
        var a = new double[,] { { 4.0 } };
        var b = new double[] { 8.0 };
        var x = GaussianElimination<double>.Solve(a, b, seed: 1);
        x[0].Should().BeApproximately(2.0, 1e-12);
    }

    [Fact]
    public void Gaussian_AllZeroMatrix_ThrowsSingular()
    {
        var a = new double[,] { { 0.0, 0.0 }, { 0.0, 0.0 } };
        var b = new double[] { 0.0, 0.0 };
        var act = () => GaussianElimination<double>.Solve(a, b, seed: 1);
        act.Should().Throw<InvalidOperationException>();
    }

    // ───────────────────────────────────────────────────────────────────────
    // B17 — PivotedCholeskyResult.Lower and .Permutation VALUES asserted.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void PivotedCholesky_Identity_LowerAndPermutationValues()
    {
        var a = new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
        };

        var result = CholeskyDecomposition<double>.DecomposePivoted(a);

        result.Rank.Should().Be(3);
        result.Permutation.Should().BeEquivalentTo(new[] { 0, 1, 2 });

        // L of identity (any pivot order) is identity on the first Rank columns.
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var expected = i == j ? 1.0 : 0.0;
                result.Lower[i, j].Should().BeApproximately(expected, 1e-12, $"L[{i},{j}]");
            }
        }
    }

    [Fact]
    public void PivotedCholesky_DiagonalMatrix_PermutesLargestFirst()
    {
        // Diagonal (1, 9, 4): pivot order must select the largest diagonal first → index 1.
        var a = new double[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 9.0, 0.0 },
            { 0.0, 0.0, 4.0 },
        };

        var result = CholeskyDecomposition<double>.DecomposePivoted(a);

        result.Rank.Should().Be(3);
        result.Permutation[0].Should().Be(1, "the largest diagonal (9) sits at original index 1");
        // L[0,0] = sqrt of the largest pivot (9) = 3.
        result.Lower[0, 0].Should().BeApproximately(3.0, 1e-12);
    }

    // ───────────────────────────────────────────────────────────────────────
    // B18 — Jacobi orthonormality (VᵀV ≈ I) asserted directly; non-convergence path.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Jacobi_EigenvectorMatrix_IsOrthonormal()
    {
        var matrix = new double[,]
        {
            { 4.0, 2.0, 1.0 },
            { 2.0, 5.0, 3.0 },
            { 1.0, 3.0, 6.0 },
        };

        var result = JacobiEigenDecomposition<double>.Decompose(matrix);
        var v = result.Vectors;
        const int n = 3;

        // VᵀV must equal the identity.
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var dot = 0.0;
                for (var k = 0; k < n; k++)
                {
                    dot += v[k, i] * v[k, j];
                }

                var expected = i == j ? 1.0 : 0.0;
                dot.Should().BeApproximately(expected, 1e-10, $"(VᵀV)[{i},{j}]");
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // B19 — NearestPsdProjection.Higham correctness (nearest correlation matrix).
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Higham_KnownInvalidCorrelation_MatchesReferenceNearest()
    {
        // Higham (2002) canonical example: the nearest correlation matrix to
        // [[1, 1, 0], [1, 1, 1], [0, 1, 1]] is approximately
        // [[1, 0.7607, 0.1573], [0.7607, 1, 0.7607], [0.1573, 0.7607, 1]].
        var bad = new double[,]
        {
            { 1.0, 1.0, 0.0 },
            { 1.0, 1.0, 1.0 },
            { 0.0, 1.0, 1.0 },
        };

        var ncm = NearestPsdProjection<double>.Higham(bad, 200, 1e-13);

        NearestPsdProjection<double>.IsPsd(ncm).Should().BeTrue();
        ncm[0, 1].Should().BeApproximately(0.7607, 1e-3);
        ncm[0, 2].Should().BeApproximately(0.1573, 1e-3);
        ncm[1, 2].Should().BeApproximately(0.7607, 1e-3);
        for (var i = 0; i < 3; i++)
        {
            ncm[i, i].Should().BeApproximately(1.0, 1e-12);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // B24 — guard tests for stateless interpolators.
    // ───────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("linear")]
    [InlineData("monotonecubic")]
    public void StatelessInterpolator_EmptyArray_Throws(string kind)
    {
        var interp = Resolve(kind);
        var act = () => interp.Interpolate(1.0, Array.Empty<double>(), Array.Empty<double>());
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData("linear")]
    [InlineData("loglinear")]
    [InlineData("flatforward")]
    [InlineData("monotonecubic")]
    public void StatelessInterpolator_LengthMismatch_Throws(string kind)
    {
        var interp = Resolve(kind);
        double[] xs = [1.0, 2.0, 3.0];
        double[] ys = [1.0, 2.0];
        var act = () => interp.Interpolate(1.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData("linear")]
    [InlineData("loglinear")]
    [InlineData("flatforward")]
    [InlineData("monotonecubic")]
    public void StatelessInterpolator_UnsortedX_Throws(string kind)
    {
        var interp = Resolve(kind);
        double[] xs = [1.0, 3.0, 2.0];
        double[] ys = [1.0, 2.0, 3.0];
        var act = () => interp.Interpolate(2.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Theory]
    [InlineData("linear")]
    [InlineData("loglinear")]
    [InlineData("flatforward")]
    [InlineData("monotonecubic")]
    public void StatelessInterpolator_DuplicateX_Throws(string kind)
    {
        var interp = Resolve(kind);
        double[] xs = [1.0, 2.0, 2.0];
        double[] ys = [1.0, 2.0, 3.0];
        var act = () => interp.Interpolate(1.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void MonotoneConvex_OriginXNotZero_Throws()
    {
        var interp = MonotoneConvexInterpolator<double>.Instance;
        double[] xs = [0.5, 1.0, 2.0];
        double[] ys = [0.0, 0.1, 0.2];
        var act = () => interp.Interpolate(1.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void MonotoneConvex_OriginYNotZero_Throws()
    {
        var interp = MonotoneConvexInterpolator<double>.Instance;
        double[] xs = [0.0, 1.0, 2.0];
        double[] ys = [0.05, 0.1, 0.2];
        var act = () => interp.Interpolate(1.5, xs, ys);
        act.Should().Throw<ArgumentException>();
    }

    private static IInterpolator<double> Resolve(string kind) => kind switch
    {
        "linear" => LinearInterpolator<double>.Instance,
        "loglinear" => LogLinearInterpolator<double>.Instance,
        "flatforward" => FlatForwardInterpolator<double>.Instance,
        "monotonecubic" => MonotoneCubicInterpolator<double>.Instance,
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    // ───────────────────────────────────────────────────────────────────────
    // B25 — T=float / T=Half parity/smoke for a representative LinAlg type and interpolator.
    // ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void Cholesky_Float_DecomposesAndSolves()
    {
        var a = new float[,]
        {
            { 4f, 12f, -16f },
            { 12f, 37f, -43f },
            { -16f, -43f, 98f },
        };

        var lower = CholeskyDecomposition<float>.Decompose(a);
        lower[0, 0].Should().BeApproximately(2f, 1e-3f);
        lower[1, 1].Should().BeApproximately(1f, 1e-3f);
        lower[2, 2].Should().BeApproximately(3f, 1e-3f);

        var b = new float[] { 4f * 1 + 12f * 2 - 16f * 3, 12f * 1 + 37f * 2 - 43f * 3, -16f * 1 - 43f * 2 + 98f * 3 };
        var x = CholeskyDecomposition<float>.Solve(lower, b);
        x[0].Should().BeApproximately(1f, 1e-2f);
        x[1].Should().BeApproximately(2f, 1e-2f);
        x[2].Should().BeApproximately(3f, 1e-2f);
    }

    [Fact]
    public void LinearInterpolator_Float_Interpolates()
    {
        var interp = LinearInterpolator<float>.Instance;
        float[] xs = [0f, 1f, 2f];
        float[] ys = [0f, 10f, 20f];
        interp.Interpolate(0.5f, xs, ys).Should().BeApproximately(5f, 1e-4f);
        interp.Interpolate(1.5f, xs, ys).Should().BeApproximately(15f, 1e-4f);
    }

    [Fact]
    public void LinearInterpolator_Half_Interpolates()
    {
        var interp = LinearInterpolator<Half>.Instance;
        Half[] xs = [(Half)0, (Half)1, (Half)2];
        Half[] ys = [(Half)0, (Half)10, (Half)20];
        ((double)interp.Interpolate((Half)0.5, xs, ys)).Should().BeApproximately(5.0, 0.1);
    }
}
