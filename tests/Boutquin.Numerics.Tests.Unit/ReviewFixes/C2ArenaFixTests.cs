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

using Boutquin.Numerics.Distributions;
using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.LinearAlgebra;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class C2ArenaFixTests
{
    [Fact]
    public void GaussianElimination_HalfZeroMatrix_IsRejectedAsSingular()
    {
        var act = () => GaussianElimination<Half>.Solve(new Half[1, 1], new Half[1], seed: 7);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Jacobi_LowerTriangleIsIgnoredByTheUpperTriangleContract()
    {
        double[,] packed = { { 2e-22, 1e-22 }, { 1.0, 3e-22 } };
        double[,] mirrored = { { 2e-22, 1e-22 }, { 1e-22, 3e-22 } };

        var packedResult = JacobiEigenDecomposition<double>.Decompose(packed);
        var mirroredResult = JacobiEigenDecomposition<double>.Decompose(mirrored);

        packedResult.Values.Should().Equal(mirroredResult.Values);
        Flatten(packedResult.Vectors).Should().Equal(Flatten(mirroredResult.Vectors));
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    public void EigenClip_InvalidFloor_IsRejected(double eigenvalueFloor)
    {
        double[,] matrix = { { 1.0, 2.0 }, { 2.0, 1.0 } };

        var act = () => NearestPsdProjection<double>.EigenClip(matrix, eigenvalueFloor);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName(nameof(eigenvalueFloor));
    }

    [Fact]
    public void CholeskySolve_RectangularFactor_IsRejectedAtTheBoundary()
    {
        double[,] lower = { { 1.0 }, { 0.0 } };

        var act = () => CholeskyDecomposition<double>.Solve(lower, [1.0, 0.0]);

        act.Should().Throw<ArgumentException>().WithParameterName(nameof(lower));
    }

    [Fact]
    public void PivotedCholesky_NegativeMaximumRank_IsRejected()
    {
        double[,] matrix = { { 1.0 } };

        var act = () => CholeskyDecomposition<double>.DecomposePivoted(matrix, 0.0, -1);

        act.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("maxRank");
    }

    [Fact]
    public void LegacyJacobi_NullMatrix_UsesTheGenericArgumentContract()
    {
#pragma warning disable BNU0001
        var act = () => JacobiEigenDecomposition.Decompose(null!);
#pragma warning restore BNU0001

        act.Should().Throw<ArgumentNullException>().WithParameterName("matrix");
    }

    [Theory]
    [InlineData(double.MaxValue, 1.0)]
    [InlineData(double.MinValue, 0.0)]
    public void CumulativeNormal_FiniteExtreme_SaturatesAtAnalyticLimit(double x, double expected)
    {
        CumulativeNormal<double>.Evaluate(x).Should().Be(expected);
        NormalDistribution<double>.Cdf(x).Should().Be(expected);
    }

    [Fact]
    public void Interpolators_NonFiniteAbscissae_AreRejectedAcrossTheFamily()
    {
        double[] xs = [0.0, double.NaN, 1.0];
        double[] ys = [0.0, 10.0, 20.0];

        Action[] calls =
        [
            () => LinearInterpolator<double>.Instance.Interpolate(0.5, xs, ys),
            () => LogLinearInterpolator<double>.Instance.Interpolate(0.5, xs, [1.0, 2.0, 3.0]),
            () => FlatForwardInterpolator<double>.Instance.Interpolate(0.5, xs, ys),
            () => MonotoneCubicInterpolator<double>.Instance.Interpolate(0.5, xs, ys),
            () => new MonotoneCubicSpline<double>(xs, ys),
            () => new CubicSplineInterpolator<double>(xs, ys),
        ];

        calls.Should().AllSatisfy(call => call.Should().Throw<ArgumentException>());
    }

    [Fact]
    public void PcaResult_NaNThreshold_IsRejectedAcrossGenericAndLegacySurfaces()
    {
        var generic = new PcaResult<double>
        {
            Eigenvalues = [2.0, 1.0],
            Eigenvectors = new double[2, 2],
            ExplainedVarianceRatio = [2.0 / 3.0, 1.0 / 3.0],
            CumulativeExplainedVariance = [2.0 / 3.0, 1.0],
            Mean = [],
        };
        var legacy = new PcaResult
        {
            Eigenvalues = [2m, 1m],
            Eigenvectors = new decimal[2, 2],
            ExplainedVarianceRatio = [2m / 3m, 1m / 3m],
            CumulativeExplainedVariance = [2m / 3m, 1m],
            Mean = [],
        };

        Action genericCall = () => generic.NumComponentsForExplainedVariance(double.NaN);
        Action legacyCall = () => legacy.NumComponentsForExplainedVariance(double.NaN);

        genericCall.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("threshold");
        legacyCall.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("threshold");
    }

    [Fact]
    public void InverseNormal_NaNProbability_IsRejectedAcrossDelegatingSurfaces()
    {
        Action[] calls =
        [
            () => InverseNormal<double>.Evaluate(double.NaN),
            () => NormalDistribution<double>.InverseCdf(double.NaN),
#pragma warning disable BNU0001
            () => InverseNormal.Evaluate(double.NaN),
            () => NormalDistribution.InverseCdf(double.NaN),
#pragma warning restore BNU0001
        ];

        calls.Should().AllSatisfy(call =>
            call.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("p"));
    }

    private static IEnumerable<double> Flatten(double[,] matrix)
    {
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var column = 0; column < matrix.GetLength(1); column++)
            {
                yield return matrix[row, column];
            }
        }
    }
}
