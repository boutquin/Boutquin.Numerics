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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Tests that absolute (scale-blind) tolerances were replaced with scale-relative ones
/// (review findings #15, #16): large-scale Jacobi inputs and tiny-scale Gaussian systems
/// must no longer be mis-handled by a fixed absolute threshold.
/// </summary>
public sealed class ScaleRelativeToleranceTests
{
    [Fact]
    public void Jacobi_LargeScaleMatrix_Converges()
    {
        // Tridiagonal [[2,1,0],[1,2,1],[0,1,2]] scaled by 1e10. Eigenvalues of the base are
        // 2-√2, 2, 2+√2; an absolute 1e-15 off-diagonal-sum threshold is unreachable at this
        // scale (the rounding floor of the off-diagonals is ~ (1e10·eps)² ≫ 1e-15).
        const double s = 1e10;
        var matrix = new double[,]
        {
            { 2 * s, 1 * s, 0 },
            { 1 * s, 2 * s, 1 * s },
            { 0, 1 * s, 2 * s },
        };

        var result = JacobiEigenDecomposition<double>.Decompose(matrix);

        var expected = new[] { (2 + Math.Sqrt(2)) * s, 2 * s, (2 - Math.Sqrt(2)) * s };
        for (var i = 0; i < 3; i++)
        {
            result.Values[i].Should().BeApproximately(expected[i], expected[i] * 1e-9);
        }
    }

    [Fact]
    public void Jacobi_TinyScaleMatrix_StillCorrect()
    {
        // Guard against over-loosening: a small-magnitude matrix must remain accurate.
        const double s = 1e-6;
        var matrix = new double[,]
        {
            { 2 * s, 1 * s, 0 },
            { 1 * s, 2 * s, 1 * s },
            { 0, 1 * s, 2 * s },
        };

        var result = JacobiEigenDecomposition<double>.Decompose(matrix);
        var expected = new[] { (2 + Math.Sqrt(2)) * s, 2 * s, (2 - Math.Sqrt(2)) * s };
        for (var i = 0; i < 3; i++)
        {
            result.Values[i].Should().BeApproximately(expected[i], Math.Abs(expected[i]) * 1e-9);
        }
    }

    [Fact]
    public void GaussianElimination_TinyScaleSystem_NotFalselySingular()
    {
        // A·x = b with A = 1e-15·I is perfectly well-conditioned (x = b), but an absolute
        // 1e-14 pivot threshold flags every pivot (1e-15) as "singular".
        const double s = 1e-15;
        var a = new double[,]
        {
            { 2 * s, 0, 0 },
            { 0, 3 * s, 0 },
            { 0, 0, 5 * s },
        };
        var b = new double[] { 2 * s, 6 * s, 20 * s };

        var x = GaussianElimination<double>.Solve(a, b, seed: 7);

        x[0].Should().BeApproximately(1.0, 1e-6);
        x[1].Should().BeApproximately(2.0, 1e-6);
        x[2].Should().BeApproximately(4.0, 1e-6);
    }

    [Fact]
    public void OracleApproximatingShrinkage_IsScaleInvariant()
    {
        // The OAS shrinkage intensity ρ is a scale-free ratio, so the correlation structure
        // of the estimate must not depend on the magnitude of the returns. An absolute
        // denominator tolerance (1e-28) breaks this: at small magnitude the denominator drops
        // below it and ρ is forced to 1 (pure scaled-identity target → zero off-diagonals).
        var x = new double[,]
        {
            { 0.020, 0.018, -0.010 },
            { -0.010, -0.012, 0.020 },
            { 0.030, 0.028, 0.010 },
            { 0.000, 0.002, -0.020 },
            { 0.010, 0.011, 0.030 },
            { -0.020, -0.019, -0.010 },
            { 0.015, 0.014, 0.000 },
            { -0.005, -0.006, 0.020 },
        };
        const double c = 1e-7;
        var scaled = new double[x.GetLength(0), x.GetLength(1)];
        for (var i = 0; i < x.GetLength(0); i++)
        {
            for (var j = 0; j < x.GetLength(1); j++)
            {
                scaled[i, j] = c * x[i, j];
            }
        }

        var estimator = new OracleApproximatingShrinkageEstimator<double>();
        var big = estimator.Estimate(x);
        var small = estimator.Estimate(scaled);

        // Correlation-like ratio cov(0,1)/var(0) is scale-invariant; it collapses to 0 for the
        // small input if ρ was wrongly forced to 1.
        var ratioBig = big[0, 1] / big[0, 0];
        var ratioSmall = small[0, 1] / small[0, 0];
        ratioSmall.Should().BeApproximately(ratioBig, Math.Abs(ratioBig) * 1e-6);
    }

    [Theory]
    [InlineData(1e100)]
    [InlineData(1e78)]
    [InlineData(2e78)]
    [InlineData(1e-100)]
    public void ScaledIdentityShrinkage_PreservesCovarianceUnitsAtExtremeScales(double scale)
    {
        AssertShrinkageScaleEquivariance(new LedoitWolfShrinkageEstimator<double>(), scale);
    }

    [Theory]
    [InlineData(1e100)]
    [InlineData(1e-100)]
    public void ConstantCorrelationShrinkage_PreservesCovarianceUnitsAtExtremeScales(double scale)
    {
        AssertShrinkageScaleEquivariance(new LedoitWolfConstantCorrelationEstimator<double>(), scale);
    }

    [Theory]
    [InlineData(1e100)]
    [InlineData(1e-100)]
    public void SingleFactorShrinkage_PreservesCovarianceUnitsAtExtremeScales(double scale)
    {
        AssertShrinkageScaleEquivariance(new LedoitWolfSingleFactorEstimator<double>(), scale);
    }

    [Theory]
    [InlineData(1e100)]
    [InlineData(1e-100)]
    public void OracleApproximatingShrinkage_PreservesCovarianceUnitsAtExtremeScales(double scale)
    {
        AssertShrinkageScaleEquivariance(new OracleApproximatingShrinkageEstimator<double>(), scale);
    }

    [Fact]
    public void ShrinkageEstimators_PreserveCovarianceUnits_ForFloatAndDecimal()
    {
        AssertShrinkageScaleEquivariance(new LedoitWolfShrinkageEstimator<float>(), 1e10f, 1e-4);
        AssertShrinkageScaleEquivariance(new LedoitWolfConstantCorrelationEstimator<float>(), 1e10f, 1e-4);
        AssertShrinkageScaleEquivariance(new LedoitWolfSingleFactorEstimator<float>(), 1e10f, 1e-4);
        AssertShrinkageScaleEquivariance(new OracleApproximatingShrinkageEstimator<float>(), 1e10f, 1e-4);

        AssertShrinkageScaleEquivariance(new LedoitWolfShrinkageEstimator<decimal>(), 1_000_000_000m, 1e-8);
        AssertShrinkageScaleEquivariance(new LedoitWolfConstantCorrelationEstimator<decimal>(), 1_000_000_000m, 1e-8);
        AssertShrinkageScaleEquivariance(new LedoitWolfSingleFactorEstimator<decimal>(), 1_000_000_000m, 1e-8);
        AssertShrinkageScaleEquivariance(new OracleApproximatingShrinkageEstimator<decimal>(), 1_000_000_000m, 1e-8);
    }

    [Fact]
    public void ShrinkageEstimators_KeepAggregateIntensityFiniteNearOverflowBoundary()
    {
        ICovarianceEstimator<double>[] estimators =
        [
            new LedoitWolfShrinkageEstimator<double>(),
            new LedoitWolfConstantCorrelationEstimator<double>(),
            new LedoitWolfSingleFactorEstimator<double>(),
            new OracleApproximatingShrinkageEstimator<double>(),
        ];
        var returns = new double[20, 3];
        var scaled = new double[20, 3];
        const double scale = 1e77;
        for (var k = 0; k < 20; k++)
        {
            for (var i = 0; i < 3; i++)
            {
                returns[k, i] = Math.Sin((k + 1) * (i + 2) * 0.71);
                scaled[k, i] = scale * returns[k, i];
            }
        }

        foreach (var estimator in estimators)
        {
            var baseline = estimator.Estimate(returns);
            var actual = estimator.Estimate(scaled);
            double.IsFinite(actual[0, 0]).Should().BeTrue($"{estimator.GetType().Name} must retain finite covariance");
            (actual[0, 0] / (scale * scale)).Should().BeApproximately(baseline[0, 0], 1e-8);
        }
    }

    private static void AssertShrinkageScaleEquivariance<T>(ICovarianceEstimator<T> estimator, T scale, double tolerance)
        where T : IFloatingPoint<T>
    {
        var returns = new T[20, 3];
        var scaled = new T[20, 3];
        for (var k = 0; k < 20; k++)
        {
            for (var i = 0; i < 3; i++)
            {
                returns[k, i] = T.CreateChecked(0.02 * Math.Sin((k + 1) * (i + 2) * 0.71) + 0.001 * (i + 1) * k);
                scaled[k, i] = scale * returns[k, i];
            }
        }

        var expected = estimator.Estimate(returns);
        var actual = estimator.Estimate(scaled);
        var scaleSquared = scale * scale;
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                T.IsFinite(actual[i, j]).Should().BeTrue($"covariance [{i},{j}] must remain finite");
                var expectedEntry = double.CreateChecked(expected[i, j]);
                var actualEntry = double.CreateChecked(actual[i, j] / scaleSquared);
                actualEntry.Should().BeApproximately(expectedEntry, Math.Max(1e-10, Math.Abs(expectedEntry) * tolerance));
            }
        }
    }

    private static void AssertShrinkageScaleEquivariance(ICovarianceEstimator<double> estimator, double scale)
    {
        var returns = new double[20, 3];
        var scaled = new double[20, 3];
        for (var k = 0; k < 20; k++)
        {
            for (var i = 0; i < 3; i++)
            {
                returns[k, i] = 0.02 * Math.Sin((k + 1) * (i + 2) * 0.71) + 0.001 * (i + 1) * k;
                scaled[k, i] = scale * returns[k, i];
            }
        }

        var expected = estimator.Estimate(returns);
        var actual = estimator.Estimate(scaled);
        var scaleSquared = scale * scale;
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                double.IsFinite(actual[i, j]).Should().BeTrue($"covariance [{i},{j}] must remain finite");
                (actual[i, j] / scaleSquared).Should().BeApproximately(
                    expected[i, j], Math.Max(1e-15, Math.Abs(expected[i, j]) * 1e-8));
            }
        }
    }
}
