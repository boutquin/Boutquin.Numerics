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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

/// <summary>
/// Regression pins for the Jacobi eigendecomposition's stopping and rotation-skip rules: an
/// absolute convergence floor below the working type's resolution (clusters of exactly equal
/// diagonals in float, or in double above unit scale), and absolute skip and stopping thresholds
/// that leave the small block of a graded matrix unrotated; plus the NERCOME route that feeds a
/// graded first-subset moment to it.
/// </summary>
public sealed class Arena414074R2JacobiTests
{
    // ------------------------------------------------------------------ working-precision clusters

    [Fact]
    public void Qis_FloatIsotropicHadamardPanel_ReturnsTheIsotropicMatrix()
    {
        // Distinct, sign-flipped Sylvester-Hadamard columns of order 16: every column has zero mean
        // and the columns are exactly orthogonal, so the sample covariance is sigma^2 * 16/15 * I and
        // QIS of an isotropic matrix is that matrix. In float the sample covariance carries rounding
        // noise below half an ulp of the equal diagonals.
        int[] columns = [5, 10, 1, 9, 2, 3];
        int[] signs = [-1, -1, -1, -1, 1, -1];
        int[] rows = [11, 4, 12, 15, 0, 1, 10, 9, 3, 2, 13, 7, 8, 5, 14, 6];
        const double sigma = 1.6614963415805854E+19;
        var x = new float[rows.Length, columns.Length];
        for (var i = 0; i < rows.Length; i++)
        {
            for (var j = 0; j < columns.Length; j++)
            {
                x[i, j] = (float)(sigma * Hadamard(rows[i], columns[j]) * signs[j]);
            }
        }

        float[,]? actual = null;
        var estimate = () => actual = new QuadraticInverseShrinkageEstimator<float>().Estimate(x);
        estimate.Should().NotThrow("an isotropic sample covariance is diagonal to working precision");

        var sigmaFloat = (double)(float)sigma;
        var expected = sigmaFloat * sigmaFloat * 16 / 15;
        for (var i = 0; i < columns.Length; i++)
        {
            for (var j = 0; j < columns.Length; j++)
            {
                var want = i == j ? expected : 0;
                Math.Abs(actual![i, j] - want).Should().BeLessThan(1e-4 * expected, "entry [{0},{1}] of an isotropic QIS estimate", i, j);
            }
        }
    }

    [Fact]
    public void Jacobi_FloatEqualDiagonalCluster_ConvergesToWorkingPrecision()
    {
        // The normalized 7x7 matrix QIS forms from a float isotropic Hadamard panel (t = 32): exactly
        // equal diagonals and off-diagonal rounding noise of at most 0.13 float epsilon, so every
        // rotation's diagonal update is below half an ulp of the diagonals.
        const float a = 5.1336815E-09f;
        const float b = 1.5401044E-08f;
        float[,] matrix =
        {
            { 1, -a, a, -a, -b, a, a },
            { -a, 1, a, a, -b, -a, -a },
            { a, a, 1, -a, -a, a, a },
            { -a, a, -a, 1, -a, a, a },
            { -b, -b, -a, -a, 1, a, a },
            { a, -a, a, a, a, 1, b },
            { a, -a, a, a, a, b, 1 },
        };

        AssertBackwardStable(matrix, 8 * MachineEpsilon<float>());
    }

    [Fact]
    public void Jacobi_DoubleEqualDiagonalClusterAboveUnitScale_Converges()
    {
        // The same mechanism in double above unit scale: diagonals exactly 2^30 and off-diagonals below
        // half an ulp of them, so the absolute 1e-15 floor is far below the resolution of the cluster.
        var u = Math.ScaleB(1.0, 30 - 55);
        var a = u;
        var b = 3 * u;
        var d = Math.ScaleB(1.0, 30);
        double[,] matrix =
        {
            { d, -a, a, -a, -b, a, a },
            { -a, d, a, a, -b, -a, -a },
            { a, a, d, -a, -a, a, a },
            { -a, a, -a, d, -a, a, a },
            { -b, -b, -a, -a, d, a, a },
            { a, -a, a, a, a, d, b },
            { a, -a, a, a, a, b, d },
        };

        AssertBackwardStable(matrix, 8 * MachineEpsilon<double>());
    }

    // ---------------------------------------------------------------------------- graded matrices

    [Fact]
    public void Jacobi_GradedBlockDiagonal_RotatesTheSmallBlock()
    {
        // blockdiag(s * [[1, 1/2], [1/2, 1]], [[4, 1], [1, 3]]) with s = 2^-600. The small block's
        // eigenvalues are s * 3/2 and s / 2 with 45-degree eigenvectors; its off-diagonal is far below
        // the absolute rotation-skip floor although it is half of its own diagonal.
        var s = Math.ScaleB(1.0, -600);
        double[,] matrix =
        {
            { s, s / 2, 0, 0 },
            { s / 2, s, 0, 0 },
            { 0, 0, 4, 1 },
            { 0, 0, 1, 3 },
        };

        var result = Decompose(matrix);
        var values = result.Values;
        var vectors = result.Vectors;

        (values[2] / s).Should().BeApproximately(1.5, 1e-14, "the larger eigenvalue of the small block is 3s/2");
        (values[3] / s).Should().BeApproximately(0.5, 1e-14, "the smaller eigenvalue of the small block is s/2");
        for (var k = 2; k < 4; k++)
        {
            Math.Abs(vectors[0, k]).Should().BeApproximately(1 / Math.Sqrt(2), 1e-14);
            Math.Abs(vectors[1, k]).Should().BeApproximately(1 / Math.Sqrt(2), 1e-14);
        }
    }

    [Fact]
    public void Jacobi_CoupledGradedMatrix_SmallEigenpairsHaveRelativeAccuracy()
    {
        // A = D C D with C positive definite and D = diag(2^-300, 2^-299, 1, 2). Expected values: 500-digit
        // mpmath eigsy on the exact double entries (80 digits are not enough: the small eigenvalues sit
        // 181 decimal orders below the norm). The small eigenvalues are the Schur complement's,
        // about 1e-181, and differ from the reduced diagonal unless the small pair is rotated.
        double[] d = [Math.ScaleB(1.0, -300), Math.ScaleB(1.0, -299), 1, 2];
        double[,] c = { { 1, .5, .3, .2 }, { .5, 1, .4, .1 }, { .3, .4, 1, .5 }, { .2, .1, .5, 1 } };
        var matrix = new double[4, 4];
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                matrix[i, j] = c[i, j] * d[i] * d[j];
            }
        }

        var result = Decompose(matrix);
        var values = result.Values;
        var vectors = result.Vectors;

        const double Large1 = 4.3027756377319946466;
        const double Large2 = 0.69722436226800535344;
        const double Small1 = 8.5173017143645448154e-181;
        const double Small2 = 1.6364939839356067592e-181;
        values[0].Should().BeApproximately(Large1, 1e-14);
        values[1].Should().BeApproximately(Large2, 1e-14);
        (values[2] / Small1).Should().BeApproximately(1, 1e-12, "the larger small eigenvalue needs relative accuracy");
        (values[3] / Small2).Should().BeApproximately(1, 1e-12, "the smallest eigenvalue needs relative accuracy");

        // Eigenvector components on the small coordinates, sign-normalized by the first component.
        double[][] expected = [[0.28233753284532270504, 0.95931512942662189764], [0.95931512942662189764, -0.28233753284532270504]];
        for (var k = 0; k < 2; k++)
        {
            var sign = Math.Sign(vectors[0, k + 2]);
            (sign * vectors[0, k + 2]).Should().BeApproximately(expected[k][0], 1e-12);
            (sign * vectors[1, k + 2]).Should().BeApproximately(expected[k][1], 1e-12);
        }
    }

    // ------------------------------------------------------------------- NERCOME on coupled columns

    [Fact]
    public void Nercome_CoupledColumnsBeyondTheSquaredRange_KeepEveryEntry()
    {
        // Every column is nonzero in every row (one coupled group); columns at 2^-240, 2^-235, 2^330
        // and 2^328 span 2^570, so the small columns' products underflow at the group maximum.
        // Expected: 500-digit mpmath NERCOME (exact eigenvectors of the first-subset moment, whose
        // entries span 342 decimal orders) on the first subset that Xoshiro256** seed 7 yields for 24 rows and size 6.
        var rows = CoupledPanel(-240, -235, 330, 328);
        double[,] expected =
        {
            { 1.0380602684182512e-143, -2.2606652651064658e-143, -9.6202380599486309e+27, 1.7735620680269496e+27 },
            { -2.2606652651064658e-143, 5.0728551710974476e-141, -2.7612595952099677e+28, -9.9560741100190574e+27 },
            { -9.6202380599486309e+27, -2.7612595952099677e+28, 4.4653460014411574e+199, -3.0419087662573448e+198 },
            { 1.7735620680269496e+27, -9.9560741100190574e+27, -3.0419087662573448e+198, 3.967930265719588e+198 },
        };

        AssertRelative(Estimate(new NercomeCovarianceEstimator<double>(0.25, 1, 7), rows), expected, 1e-9);
    }

    [Fact]
    public void Nercome_DefaultSelectionOnCoupledColumnsBeyondTheSquaredRange_ScoresEveryCandidate()
    {
        // The same panel with the seven-candidate selector (sizes 9, 4, 14, 19, 11, 16) and two
        // replications from seed 7. The 500-digit oracle's squared averaged residuals are 3.67e396,
        // 1.97e397, 7.68e397, 1.50e395, 4.31e397 and 9.15e396 in original units, so it selects size 19,
        // not the first candidate; a score whose squares leave the range of double cannot order them.
        var rows = CoupledPanel(-240, -235, 330, 328);
        double[,] expected =
        {
            { 6.5544154582503424e-144, 6.5383587952156413e-144, 4.1647044114472849e+26, -2.7009437761499663e+26 },
            { 6.5383587952156413e-144, 6.0649463857766522e-141, -1.6123446064098361e+28, -1.4291318992180269e+28 },
            { 4.1647044114472849e+26, -1.6123446064098361e+28, 4.141007329777185e+199, -2.4460170611055683e+198 },
            { -2.7009437761499663e+26, -1.4291318992180269e+28, -2.4460170611055683e+198, 5.2046671887686578e+198 },
        };

        AssertRelative(Estimate(new NercomeCovarianceEstimator<double>(null, 2, 7), rows), expected, 1e-9);
    }

    [Fact]
    public void Nercome_FloatCoupledColumnsBeyondTheSquaredRange_KeepEveryEntry()
    {
        // The same integers at 2^-62, 2^-58, 2^28 and 2^26 in float: the small columns' products at the
        // group maximum (about 2^-172) are below float's normal range. Expected: 500-digit mpmath NERCOME
        // on the exact float entries and the same first subset (seed 7, size 6).
        var source = CoupledPanel(-62, -58, 28, 26);
        var rows = new float[source.GetLength(0), source.GetLength(1)];
        for (var i = 0; i < source.GetLength(0); i++)
        {
            for (var j = 0; j < source.GetLength(1); j++)
            {
                rows[i, j] = (float)source[i, j];
            }
        }

        double[,] expected =
        {
            { 1.5236777881091906e-36, -1.6519657038773748e-36, -4.5234141972140714e-10, 8.3392487671934913e-11 },
            { -1.6519657038773748e-36, 1.8608387960127563e-34, -6.4916901106463067e-10, -2.3406617745390871e-10 },
            { -4.5234141972140714e-10, -6.4916901106463067e-10, 6.7257037708942357e+17, -4.5817227272713631e+16 },
            { 8.3392487671934913e-11, -2.3406617745390871e-10, -4.5817227272713631e+16, 5.9764962316878754e+16 },
        };

        float[,]? actual = null;
        var estimate = () => actual = new NercomeCovarianceEstimator<float>(0.25f, 1, 7).Estimate(rows);
        estimate.Should().NotThrow("the covariance of this panel is representable in float");
        var widened = new double[4, 4];
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                widened[i, j] = actual![i, j];
            }
        }

        AssertRelative(widened, expected, 1e-4);
    }

    [Fact]
    public void Nercome_CoupledGradedColumnsWithinRange_ResolvesTheSmallBlock()
    {
        // Columns at 2^-200, 2^-195, 2^200 and 2^198: the products at the group maximum stay normal
        // (about 2^-800), so only the eigensolver's treatment of the graded moment decides the small
        // block. Expected: 500-digit mpmath NERCOME on the same first subset (seed 7, size 6).
        var rows = CoupledPanel(-200, -195, 200, 198);
        double[,] expected =
        {
            { 1.2549378608069163e-119, -2.7329766084931571e-119, -7.7711664173186124, 1.4326720291321745 },
            { -2.7329766084931571e-119, 6.1327055955052918e-117, -2.2305277376794007e+1, -8.0424526181141638 },
            { -7.7711664173186124, -2.2305277376794007e+1, 2.4102175453289273e+121, -1.6419023021636136e+120 },
            { 1.4326720291321745, -8.0424526181141638, -1.6419023021636136e+120, 2.1417321618509406e+120 },
        };

        AssertRelative(Estimate(new NercomeCovarianceEstimator<double>(0.25, 1, 7), rows), expected, 1e-9);
    }

    // -------------------------------------------------------------------------------------- helpers

    private static EigenResult<double> Decompose(double[,] matrix)
    {
        EigenResult<double>? result = null;
        var decompose = () => result = JacobiEigenDecomposition<double>.Decompose(matrix);
        decompose.Should().NotThrow("a graded symmetric matrix is diagonalizable within the sweep cap");
        return result!;
    }

    private static double[,] Estimate(NercomeCovarianceEstimator<double> estimator, double[,] rows)
    {
        double[,]? result = null;
        var estimate = () => result = estimator.Estimate(rows);
        estimate.Should().NotThrow("the covariance of these panels is representable in double");
        return result!;
    }

    private static readonly int[][] s_coupledIntegers =
    [
        [-2, 1, -4, -1], [5, 5, 5, -4], [-1, -3, 4, 4], [-4, -4, 3, -5], [-2, -4, 4, -3], [-3, -4, 1, -2],
        [1, 5, 3, -2], [-5, 5, 1, -1], [4, -5, -2, 4], [-5, -5, -4, -5], [-5, -5, 4, -5], [3, -4, -1, -5],
        [1, -5, -5, 3], [-2, 4, 2, 4], [2, -3, -2, 4], [4, 2, -3, -4], [4, 4, 1, -5], [3, -1, 1, 3],
        [5, -5, 2, 1], [1, -2, 3, 3], [-2, -1, 2, 2], [-4, 4, 2, -5], [-5, -3, 1, 4], [-4, 3, -5, 5],
    ];

    private static double[,] CoupledPanel(params int[] exponents)
    {
        var rows = new double[s_coupledIntegers.Length, exponents.Length];
        for (var i = 0; i < s_coupledIntegers.Length; i++)
        {
            for (var j = 0; j < exponents.Length; j++)
            {
                rows[i, j] = Math.ScaleB(s_coupledIntegers[i][j], exponents[j]);
            }
        }

        return rows;
    }

    private static void AssertRelative(double[,] actual, double[,] expected, double tolerance)
    {
        var n = expected.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var error = Math.Abs(actual[i, j] - expected[i, j]) / Math.Abs(expected[i, j]);
                error.Should().BeLessThan(tolerance, "entry [{0},{1}] expected {2:R}, actual {3:R}", i, j, expected[i, j], actual[i, j]);
            }
        }
    }

    private static int Hadamard(int i, int j) => (BitOperations.PopCount((uint)(i & j)) & 1) == 0 ? 1 : -1;

    private static T MachineEpsilon<T>()
        where T : IFloatingPoint<T>
    {
        var two = T.One + T.One;
        var epsilon = T.One;
        while (T.One + (epsilon / two) != T.One)
        {
            epsilon /= two;
        }

        return epsilon;
    }

    // A converged decomposition reconstructs the input to a small multiple of the working epsilon times
    // its largest entry and returns orthonormal eigenvectors.
    private static void AssertBackwardStable<T>(T[,] matrix, T tolerance)
        where T : IFloatingPoint<T>
    {
        EigenResult<T>? result = null;
        var decompose = () => result = JacobiEigenDecomposition<T>.Decompose(matrix);
        decompose.Should().NotThrow("a cluster with sub-ulp off-diagonals is diagonalizable within the sweep cap");

        var n = matrix.GetLength(0);
        var values = result!.Values;
        var vectors = result.Vectors;
        var largest = 0.0;
        foreach (var entry in matrix)
        {
            largest = Math.Max(largest, Math.Abs(double.CreateChecked(entry)));
        }

        var bound = double.CreateChecked(tolerance);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var reconstructed = 0.0;
                var gram = 0.0;
                for (var k = 0; k < n; k++)
                {
                    reconstructed += double.CreateChecked(vectors[i, k]) * double.CreateChecked(values[k]) * double.CreateChecked(vectors[j, k]);
                    gram += double.CreateChecked(vectors[k, i]) * double.CreateChecked(vectors[k, j]);
                }

                Math.Abs(reconstructed - double.CreateChecked(matrix[i, j])).Should().BeLessThanOrEqualTo(bound * largest, "reconstruction entry [{0},{1}]", i, j);
                Math.Abs(gram - (i == j ? 1 : 0)).Should().BeLessThanOrEqualTo(bound, "orthogonality entry [{0},{1}]", i, j);
            }
        }
    }
}
