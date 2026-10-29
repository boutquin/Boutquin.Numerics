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

using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

// Independently expressed analytic cases. Upstream test identifiers describe the
// motivating failure classes; upstream code and random fixtures are not copied.
[Trait("Tier", "Unit")]
public sealed class UpstreamInspiredContractCasesTests
{
    // SciPy test_zeros.py: test_brent_underflow_in_root_bracketing, test_gh5584.
    // Both scales stay inside the documented central range. The exact dyadic
    // zero prevents a residual criterion from hiding an inaccurate coordinate.
    [Theory]
    [InlineData(false, -490)]
    [InlineData(true, -490)]
    [InlineData(false, 490)]
    [InlineData(true, 490)]
    public void ScaledSignBracket_FindsExactDyadicZero(bool useBisection, int exponent)
    {
        var scale = Math.ScaleB(1d, exponent);
        var solver = CreateSolver(useBisection);
        var result = solver.Solve(x => scale * (x - 1.25), 1d, 2d);

        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Root - 1.25), 0d, 1e-12);
        Assert.Equal(scale * (result.Root - 1.25), result.FinalResidual);
    }

    [Theory]
    [InlineData(false, -490)]
    [InlineData(true, -490)]
    [InlineData(false, 490)]
    [InlineData(true, 490)]
    public void ScaledSameSignEndpoints_AreRejected(bool useBisection, int exponent)
    {
        var scale = Math.ScaleB(1d, exponent);
        var solver = CreateSolver(useBisection);
        Assert.Throws<InvalidOperationException>(() => solver.Solve(x => scale * (x - 1.25), 2d, 3d));
    }

    // SciPy test_zeros.py: test_gh_5557. A residual is small throughout the
    // plateau, but its endpoint remains farther from the zero than tolerance.
    [Theory]
    [InlineData(-300)]
    [InlineData(0)]
    [InlineData(300)]
    public void Brent_FlatShoulder_UsesCoordinateTolerance(int exponent)
    {
        var scale = Math.ScaleB(1d, exponent);
        var tolerance = scale / 8d;
        var root = 3d * scale / 4d;
        var solver = new BrentSolver<double>(tolerance, 128);
        var result = solver.Solve(x => x < scale / 2d ? -scale / 4d : x - root, 0d, scale);

        Assert.True(result.Converged);
        Assert.InRange(Math.Abs(result.Root - root), 0d, tolerance + 6d * Math.ScaleB(1d, -52) * root);
    }

    // SciPy TestPCHIP.test_overshoot/test_monotone motivate properties only.
    // Numerics uses Fritsch-Carlson, so no PCHIP derivative/value is an oracle.
    [Theory]
    [InlineData(false, false, 1)]
    [InlineData(false, true, 1)]
    [InlineData(true, false, 1)]
    [InlineData(true, true, 1)]
    [InlineData(false, false, -1)]
    [InlineData(false, true, -1)]
    [InlineData(true, false, -1)]
    [InlineData(true, true, -1)]
    public void IrregularGrid_PlateauAndReversals_PreserveEverySegment(bool useDecimal, bool cached, int sign)
    {
        if (useDecimal)
        {
            CheckSegmentShapes<decimal>(cached, sign, 1e-24m);
        }
        else
        {
            CheckSegmentShapes<double>(cached, sign, 2e-13);
        }
    }

    // scikit-learn test_covariance.py:test_oas has a one-feature case.
    // Nonzero mean distinguishes Numerics' known-zero-mean second moment from
    // sklearn's default centering. Its target equals the input moment for p=1.
    [Fact]
    public void Oas_OneFeatureNonzeroMean_DoublePreservesRawSecondMoment()
    {
        var result = new OracleApproximatingShrinkageEstimator<double>().Estimate(new double[,] { { 1d }, { 2d }, { 4d } });
        Assert.InRange(Math.Abs(result[0, 0] - 7d), 0d, 1e-13);
    }

    [Fact]
    public void Oas_OneFeatureNonzeroMean_DecimalPreservesRawSecondMoment()
    {
        var result = new OracleApproximatingShrinkageEstimator<decimal>().Estimate(new decimal[,] { { 1m }, { 2m }, { 4m } });
        Assert.InRange(Math.Abs(result[0, 0] - 7m), 0m, 1e-25m);
    }

    // Independent rational oracle: S=[[5,5/2],[5/2,5/2]], tr(S)=15/2,
    // tr(S^2)=175/4, dispersion=125/8, n=4, p=2. Original OAS gives
    // rho=(225/4)/(125/2)=9/10; sklearn's simplified rho clips to 1.
    [Fact]
    public void Oas_TwoFeatures_DoubleRetainsOriginalFiniteDimensionCorrection()
    {
        CheckOriginalOas<double>(1e-13);
    }

    [Fact]
    public void Oas_TwoFeatures_DecimalRetainsOriginalFiniteDimensionCorrection()
    {
        CheckOriginalOas<decimal>(1e-25m);
    }

    private static IBracketedRootSolver<double> CreateSolver(bool useBisection) => useBisection
        ? new BisectionSolver<double>(double.Epsilon, 1e-12, 200)
        : new BrentSolver<double>(1e-12, 200);

    private static void CheckSegmentShapes<T>(bool cached, int sign, T tolerance)
        where T : unmanaged, IFloatingPoint<T>
    {
        T[] xs = [T.Zero, T.One / T.CreateChecked(1024), T.One, T.CreateChecked(17), T.CreateChecked(18), T.CreateChecked(1000)];
        T[] ys = [T.CreateChecked(2 * sign), T.CreateChecked(5 * sign), T.CreateChecked(5 * sign), T.CreateChecked(-3 * sign), T.CreateChecked(7 * sign), T.CreateChecked(sign)];
        var spline = new MonotoneCubicSpline<T>(xs, ys);
        T Evaluate(T x) => cached ? spline.Interpolate(x) : MonotoneCubicInterpolator<T>.Instance.Interpolate(x, xs, ys);

        for (var segment = 0; segment < xs.Length - 1; segment++)
        {
            var previous = ys[segment];
            var direction = T.Sign(ys[segment + 1] - ys[segment]);
            for (var step = 1; step <= 64; step++)
            {
                var x = xs[segment] + (xs[segment + 1] - xs[segment]) * T.CreateChecked(step) / T.CreateChecked(64);
                var value = Evaluate(x);
                Assert.True(T.IsFinite(value));
                Assert.True(value >= T.Min(ys[segment], ys[segment + 1]) - tolerance);
                Assert.True(value <= T.Max(ys[segment], ys[segment + 1]) + tolerance);
                Assert.True(T.CreateChecked(direction) * (value - previous) >= -tolerance);
                if (direction == 0)
                {
                    Assert.True(T.Abs(value - ys[segment]) <= tolerance);
                }

                previous = value;
            }
        }
    }

    private static void CheckOriginalOas<T>(T tolerance)
        where T : IFloatingPoint<T>
    {
        T[,] observations =
        {
            { T.CreateChecked(3), T.One },
            { T.CreateChecked(-3), -T.One },
            { T.One, T.CreateChecked(2) },
            { -T.One, T.CreateChecked(-2) },
        };
        var result = new OracleApproximatingShrinkageEstimator<T>().Estimate(observations);
        T[,] expected =
        {
            { T.CreateChecked(31) / T.CreateChecked(8), T.One / T.CreateChecked(4) },
            { T.One / T.CreateChecked(4), T.CreateChecked(29) / T.CreateChecked(8) },
        };
        for (var row = 0; row < 2; row++)
        {
            for (var column = 0; column < 2; column++)
            {
                Assert.True(T.Abs(result[row, column] - expected[row, column]) <= tolerance);
            }
        }
    }
}
