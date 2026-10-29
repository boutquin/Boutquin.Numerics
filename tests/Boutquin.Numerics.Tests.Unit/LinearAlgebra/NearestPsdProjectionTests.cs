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

using Boutquin.Numerics.LinearAlgebra;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class NearestPsdProjectionTests
{
    [Fact]
    public void Higham_LargestFiniteTolerance_ReturnsExactFixedPoint()
    {
        decimal[,]? result = null;
        var act = () => result = NearestPsdProjection<decimal>.Higham(
            new[,] { { 1m, 0m }, { 0m, 1m } }, maxIterations: 2, tolerance: decimal.MaxValue);

        act.Should().NotThrow("every finite non-negative tolerance is accepted");
        result![0, 0].Should().Be(1m);
        result[1, 1].Should().Be(1m);
        result[0, 1].Should().Be(0m);
    }

    [Fact]
    public void EigenClip_PassesPsdInputThrough()
    {
        var psd = new decimal[,]
        {
            { 2m, 1m, 0m },
            { 1m, 2m, 1m },
            { 0m, 1m, 2m },
        };

        var projected = NearestPsdProjection.EigenClip(psd);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                // T = decimal: tightened from the previous double-cast 1e-10. The reconstruction
                // error is bounded by the Jacobi off-diagonal convergence threshold (~1e-13 here),
                // not the full decimal regime — EigenClip's accuracy is eigensolver-limited, so 1e-12
                // is the honest tightened bound.
                Math.Abs(projected[i, j] - psd[i, j]).Should().BeLessThan(1e-12m);
            }
        }
    }

    [Fact]
    public void EigenClip_RemovesNegativeEigenvalues()
    {
        // Indefinite matrix: λ = {3, 1, −1} on a perturbation of identity.
        // Construct via eigendecomp: V·diag(3,1,-1)·V^T with random orthogonal V.
        var indefinite = new decimal[,]
        {
            { 1m, 2m, 0m },
            { 2m, 1m, 0m },
            { 0m, 0m, 1m },
        };

        var projected = NearestPsdProjection.EigenClip(indefinite);
        // The 2×2 block has eigenpairs (3, [1,1]/√2) and (-1, [1,-1]/√2).
        // Clipping -1 gives [[1.5, 1.5], [1.5, 1.5]]; the third eigenvalue is 1.
        var expected = new decimal[,]
        {
            { 1.5m, 1.5m, 0m },
            { 1.5m, 1.5m, 0m },
            { 0m, 0m, 1m },
        };

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Math.Abs(projected[i, j] - expected[i, j]).Should().BeLessThan(1e-12m);
            }
        }

        NearestPsdProjection.IsPsd(projected).Should().BeTrue();
    }

    [Fact]
    public void EigenClip_AlreadyPositiveDefiniteSixBySixMatrix_ReproducesNearMachinePrecision()
    {
        // Regression test: EigenClip's class doc claims it is "exact (closest PSD in Frobenius
        // norm)" — for an already-PD input, the closest PSD matrix is the input itself. Under
        // the loose Jacobi convergence threshold, this 6x6 PD matrix (B*B^T + 0.5I) was
        // perturbed by ~3e-8 relative, far looser than floating-point rounding.
        double[,] matrix =
        {
            { 1.069031501167347, -0.1474906577398295, -0.06018170477633033, 0.12263704961573811, -0.040312988955308326, 0.27724366073432183 },
            { -0.1474906577398295, 0.8092803531100647, -0.05491673299281633, 0.16217255498980696, 0.336288997205411, -0.1369048939674599 },
            { -0.06018170477633033, -0.05491673299281633, 0.6301809931885348, 0.021519703130660046, -0.09130170269560402, -0.12559125513077662 },
            { 0.12263704961573811, 0.16217255498980696, 0.021519703130660046, 0.7751854857880425, 0.2300268574707456, -0.1492297208932007 },
            { -0.040312988955308326, 0.336288997205411, -0.09130170269560402, 0.2300268574707456, 0.9481865960249922, -0.2450613490688443 },
            { 0.27724366073432183, -0.1369048939674599, -0.12559125513077662, -0.1492297208932007, -0.2450613490688443, 1.296703521259973 },
        };

        var clipped = NearestPsdProjection<double>.EigenClip(matrix);

        for (var i = 0; i < 6; i++)
        {
            for (var j = 0; j < 6; j++)
            {
                clipped[i, j].Should().BeApproximately(matrix[i, j], 1e-10,
                    $"EigenClip should reproduce an already-PD input at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void Higham_AlreadyValidCorrelationMatrix_DoesNotThrow()
    {
        // Regression test: this 5x5 correlation matrix is already PSD (min eigenvalue > 0), so
        // Higham's fixed point is itself and the method should return without iterating far.
        // Under the loose Jacobi convergence threshold, EigenClip's residual noise (~2e-8 per
        // iterate) never let the iterate-to-iterate delta fall below the default 1e-12
        // tolerance, so this already-valid input spuriously threw "did not converge".
        double[,] correlation =
        {
            { 1, -0.07540292352970832, 0.8912896920364966, 0.5635615505868771, 0.16620558731650392 },
            { -0.07540292352970832, 1, -0.37073086497421565, 0.07640240540295276, -0.06576622351983928 },
            { 0.8912896920364966, -0.37073086497421565, 1, 0.3112615150169278, -0.08857491817492236 },
            { 0.5635615505868771, 0.07640240540295276, 0.3112615150169278, 1, 0.495576722612021 },
            { 0.16620558731650392, -0.06576622351983928, -0.08857491817492236, 0.495576722612021, 1 },
        };

        var projected = NearestPsdProjection<double>.Higham(correlation);
        for (var i = 0; i < correlation.GetLength(0); i++)
        {
            projected[i, i].Should().Be(1.0);
            for (var j = 0; j < correlation.GetLength(1); j++)
            {
                projected[i, j].Should().BeApproximately(correlation[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void IsPsd_NonSquareMatrix_Throws()
    {
        // Sibling-contract alignment: EigenClip and Higham both throw ArgumentException on a
        // non-square matrix. IsPsd used to silently return false instead, hiding the caller's
        // malformed-matrix bug behind a "not PSD" result.
        var nonSquare = new double[2, 3];

        var act = () => NearestPsdProjection<double>.IsPsd(nonSquare);

        act.Should().Throw<ArgumentException>().WithParameterName("a");
    }

    [Fact]
    public void IsPsdWithTolerance_NonSquareMatrix_Throws()
    {
        var nonSquare = new double[2, 3];

        var act = () => NearestPsdProjection<double>.IsPsd(nonSquare, -1e-12);

        act.Should().Throw<ArgumentException>().WithParameterName("a");
    }

    [Fact]
    public void IsPsd_ExactlyPsdMatrixAtLargeScale_ReturnsTrue()
    {
        // Regression test: v*v^T is exactly PSD (rank-1) for any v, in exact arithmetic. At
        // large scale, floating-point rounding in the eigendecomposition produces a smallest
        // eigenvalue slightly below zero (order scale * machine epsilon); the default absolute
        // tolerance (-1e-12) does not absorb this once the matrix's own magnitude exceeds
        // ~10, so exactly-PSD singular matrices at scale ~1e6 were reported non-PSD.
        double[] v = [1000.1, -999.7, 1000.3, -1000.9];
        var n = v.Length;
        var matrix = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                matrix[i, j] = v[i] * v[j];
            }
        }

        NearestPsdProjection<double>.IsPsd(matrix).Should().BeTrue();
    }

    [Fact]
    public void Higham_ConvergesToValidCorrelation()
    {
        // Invalid correlation: triangle inequality violated.
        var bad = new decimal[,]
        {
            { 1.0m, 0.9m, -0.9m },
            { 0.9m, 1.0m, 0.9m },
            { -0.9m, 0.9m, 1.0m },
        };

        var fixed_ = NearestPsdProjection.Higham(bad);
        NearestPsdProjection.IsPsd(fixed_).Should().BeTrue();

        for (var i = 0; i < 3; i++)
        {
            // Higham's unit-diagonal projection assigns the diagonal exactly T.One each iteration,
            // so the returned correlation matrix has an exactly-unit diagonal — assert equality
            // rather than the previous double-cast 1e-10.
            fixed_[i, i].Should().Be(1m);
        }
    }

    [Fact]
    public void IsPsd_RejectsIndefinite()
    {
        var indefinite = new decimal[,]
        {
            { 1m, 2m },
            { 2m, 1m },
        };
        NearestPsdProjection.IsPsd(indefinite).Should().BeFalse();
    }

    [Fact]
    public void IsPsd_NonFiniteTolerance_Throws()
    {
        var negative = new double[,] { { -1 } };

        var act = () => NearestPsdProjection<double>.IsPsd(negative, double.NaN);

        act.Should().Throw<ArgumentOutOfRangeException>()
            .Which.ParamName.Should().Be("tolerance");
    }

    [Fact]
    public void Higham_DoesNotConvergeWithinMaxIterations_Throws()
    {
        // Invalid correlation that needs several alternating projections to reach PSD.
        // Capping iterations at 1 prevents convergence; the method must NOT silently return a
        // non-PSD iterate (which would violate the <returns> PSD guarantee) — it must throw, the
        // same way JacobiEigenDecomposition signals non-convergence.
        var bad = new decimal[,]
        {
            { 1.0m, 0.9m, -0.9m },
            { 0.9m, 1.0m, 0.9m },
            { -0.9m, 0.9m, 1.0m },
        };

        var act = () => NearestPsdProjection.Higham(bad, maxIterations: 1, tolerance: 1e-12m);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Higham_ZeroTolerance_AcceptsExactFixedPoint()
    {
        var identity = new decimal[,] { { 1m } };

        decimal[,]? projected = null;
        var act = () => projected = NearestPsdProjection<decimal>.Higham(identity, maxIterations: 2, tolerance: 0m);

        act.Should().NotThrow();
        projected![0, 0].Should().Be(1m);
    }

    [Fact]
    public void Higham_LooseTolerance_NeverReturnsIndefiniteMatrix()
    {
        var bad = new decimal[,]
        {
            { 1m, 0.9m, -0.9m },
            { 0.9m, 1m, 0.9m },
            { -0.9m, 0.9m, 1m },
        };

        var projected = NearestPsdProjection<decimal>.Higham(bad, maxIterations: 100, tolerance: 100m);

        NearestPsdProjection<decimal>.IsPsd(projected).Should().BeTrue();
    }

    [Fact]
    public void EigenClipAndIsPsd_TinyScaleCovariance_DoNotSpuriouslyThrow()
    {
        // Propagation guard for the Jacobi scale-relative rotation-skip fix: EigenClip and IsPsd
        // both delegate to JacobiEigenDecomposition. A tiny-magnitude (~1e-22) PSD covariance must
        // diagonalize without a spurious "did not converge" throw, and IsPsd must report it PSD.
        var tinyPsd = new double[,]
        {
            { 2e-22, 1e-22 },
            { 1e-22, 3e-22 },
        };

        NearestPsdProjection<double>.IsPsd(tinyPsd).Should().BeTrue();

        var clipped = NearestPsdProjection<double>.EigenClip(tinyPsd);
        // Already PSD, so EigenClip reproduces the input.
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                clipped[i, j].Should().BeApproximately(tinyPsd[i, j], 1e-30);
            }
        }
    }
}
