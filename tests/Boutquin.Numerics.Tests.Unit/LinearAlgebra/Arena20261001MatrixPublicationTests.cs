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
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class Arena20261001MatrixPublicationTests
{
    [Fact]
    public void Higham2002_PrintedFourByFour_MatrixDistanceAndRank()
    {
        double[,] a = { { 2, -1, 0, 0 }, { -1, 2, -1, 0 }, { 0, -1, 2, -1 }, { 0, 0, -1, 2 } };
        double[,] printed = { { 1, -.8084, .1916, .1068 }, { -.8084, 1, -.6562, .1916 }, { .1916, -.6562, 1, -.8084 }, { .1068, .1916, -.8084, 1 } };
        var x = NearestPsdProjection<double>.Higham(a, tolerance: 1e-10, maxIterations: 1000);
        var squaredDistance = 0d;
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                x[i, j].Should().BeApproximately(printed[i, j], 5.01e-5);
                squaredDistance += Math.Pow(a[i, j] - x[i, j], 2);
            }
        }
        Math.Sqrt(squaredDistance).Should().BeApproximately(2.13, .005);
        var (values, _) = JacobiEigenDecomposition<double>.Decompose(x);
        values.Count(v => v > 1e-7).Should().Be(3);
        values.Min().Should().BeApproximately(0, 1e-7);
    }

    [Fact]
    public void Higham1989_NonsymmetricProjection_IncludesSkewDistance()
    {
        double[,] a = { { 1, 4 }, { 0, -1 } };
        var x = NearestPsdProjection<double>.EigenClip(a, 0);
        x[0, 0].Should().BeApproximately((1 + Math.Sqrt(5)) / 2, 2e-14);
        x[1, 1].Should().BeApproximately((-1 + Math.Sqrt(5)) / 2, 2e-14);
        x[0, 1].Should().BeApproximately(1, 2e-14);
        x[1, 0].Should().BeApproximately(1, 2e-14);
        var distance = Math.Pow(1 - x[0, 0], 2) + Math.Pow(4 - x[0, 1], 2) + x[1, 0] * x[1, 0] + Math.Pow(-1 - x[1, 1], 2);
        distance.Should().BeApproximately(13, 1e-13);
    }

    [Fact]
    public void Hull2021_PrintedSpectrum_ExplainedVarianceArithmetic()
    {
        // Table22.10, Global11th edition. This is printed-spectrum arithmetic,
        // not a reconstruction of the unavailable 2631-by-8 yield panel.
        double[] deviations = [11.54, 3.55, 1.78, 1.25, .91, .69, .62, .57];
        var diagonal = new double[8, 8];
        for (var i = 0; i < 8; i++) { diagonal[i, i] = deviations[i] * deviations[i]; }
        var pca = PrincipalComponentAnalysis<double>.Decompose(diagonal);
        pca.Eigenvalues.Sum().Should().BeApproximately(152.5185, 1e-12);
        pca.ExplainedVarianceRatio[0].Should().BeApproximately(133.1716 / 152.5185, 1e-14);
        pca.CumulativeExplainedVariance[1].Should().BeApproximately(145.7741 / 152.5185, 1e-14);
        pca.ExplainedVarianceRatio[2].Should().BeApproximately(3.1684 / 152.5185, 1e-14);
        pca.NumComponentsForExplainedVariance(.95).Should().Be(2);
        pca.NumComponentsForExplainedVariance(.96).Should().Be(3);
    }

    [Fact]
    public void Jolliffe_TruncationError_EqualsDiscardedSpectrum()
    {
        // Centered rows with independent columns: X'X=diag(18,8,2).
        double[,] data = { { 3, 0, 0 }, { -3, 0, 0 }, { 0, 2, 0 }, { 0, -2, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
        double[,] covariance = { { 3.6, 0, 0 }, { 0, 1.6, 0 }, { 0, 0, .4 } };
        var pca = PrincipalComponentAnalysis<double>.Decompose(covariance);
        var scores = pca.Project(data, 1);
        var vectors = pca.Eigenvectors;
        var error = 0d;
        for (var i = 0; i < 6; i++)
        {
            for (var j = 0; j < 3; j++) { error += Math.Pow(data[i, j] - scores[i, 0] * vectors[j, 0], 2); }
        }
        error.Should().BeApproximately(10, 1e-13);
        error.Should().BeApproximately(5 * pca.Eigenvalues.Skip(1).Sum(), 1e-13);
    }

    [Fact]
    public void Cholesky_ComponentwiseFactorAndSolveBounds()
    {
        // Higham(2002) Th10.3/10.4. Exact binary integer arithmetic evaluates
        // residuals independently, so checker rounding cannot hide a violation.
        double[,] a = { { 6, 2, 1 }, { 2, 5, -1 }, { 1, -1, 4 } };
        double[] b = [1, -3, 7];
        var l = CholeskyDecomposition<double>.Decompose(a);
        var solution = CholeskyDecomposition<double>.Solve(l, b);
        const int N = 3;
        var factorGamma = Gamma(N + 1);
        var solveGamma = Gamma(3 * N + 1);
        var scale = BigInteger.One << 1074;
        for (var i = 0; i < N; i++)
        {
            var residual = -Exact(b[i]) * scale;
            var solveBound = 0d;
            for (var j = 0; j < N; j++)
            {
                var product = BigInteger.Zero;
                var absoluteProduct = 0d;
                for (var k = 0; k < N; k++)
                {
                    product += Exact(l[i, k]) * Exact(l[j, k]);
                    absoluteProduct += Math.Abs(l[i, k] * l[j, k]);
                }
                var error = BigInteger.Abs(product - Exact(a[i, j]) * scale);
                (error <= Exact(factorGamma * absoluteProduct) * scale).Should().BeTrue($"factor component ({i},{j})");
                residual += Exact(a[i, j]) * Exact(solution[j]);
                solveBound += absoluteProduct * Math.Abs(solution[j]);
            }
            (BigInteger.Abs(residual) <= Exact(solveGamma * solveBound) * scale).Should().BeTrue($"solve component {i}");
        }
    }

    [Fact]
    public void Higham_WBound_AndKahanFamilyLimit()
    {
        const int Rank = 3;
        const int N = 5;
        const double Theta = .1;
        var a = KahanGram(N, Rank, Theta);
        var result = CholeskyDecomposition<double>.DecomposePivoted(a, 1e-12, null);
        result.Rank.Should().Be(Rank);
        result.Permutation.Take(Rank).Should().Equal(0, 1, 2);
        var l = result.Lower;
        var frobeniusSquared = 0d;
        for (var j = Rank; j < N; j++)
        {
            var w = new double[Rank];
            for (var i = Rank - 1; i >= 0; i--)
            {
                var rhs = l[j, i];
                for (var k = i + 1; k < Rank; k++) { rhs -= l[k, i] * w[k]; }
                w[i] = rhs / l[i, i];
                // Independent solution of the unit upper triangular Kahan block.
                w[i].Should().BeApproximately(-Math.Cos(Theta) * Math.Pow(1 + Math.Cos(Theta), Rank - i - 1), 1e-9);
                frobeniusSquared += w[i] * w[i];
            }
        }
        var boundSquared = (N - Rank) * (Math.Pow(4, Rank) - 1) / 3;
        frobeniusSquared.Should().BeLessThanOrEqualTo(boundSquared);
        frobeniusSquared.Should().BeGreaterThan(.98 * boundSquared);
    }

    [Fact]
    public void Higham1990_CTheta_RankEstimateIsThresholdDependent()
    {
        // Example2.1/section6 family, n=r=10, theta=.38 radians. Independent
        // 100-digit eigenanalysis gives lambda_min/lambda_max=4.362314e-16.
        // The historical double experiment's 3.7e-16 is not an exact eigenvalue.
        var a = KahanGram(10, 10, .38);
        const double SpectralNorm = 927.80415392799891144980378029325;
        for (var i = 0; i < 10; i++)
        {
            for (var j = 0; j < 10; j++) { a[i, j] /= SpectralNorm; }
        }
        CholeskyDecomposition<double>.DecomposePivoted(a).Rank.Should().Be(10);
        CholeskyDecomposition<double>.DecomposePivoted(a, 1e-10, null).Rank.Should().Be(9);
    }

    private static double[,] KahanGram(int n, int rank, double theta)
    {
        var upper = new double[rank, n];
        for (var i = 0; i < rank; i++)
        {
            for (var j = i; j < n; j++) { upper[i, j] = (rank - i) * Math.Pow(Math.Sin(theta), i) * (i == j ? 1 : -Math.Cos(theta)); }
        }
        var a = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                for (var k = 0; k < rank; k++) { a[i, j] += upper[k, i] * upper[k, j]; }
            }
        }
        return a;
    }

    private static double Gamma(int n) => n * Math.ScaleB(1, -53) / (1 - n * Math.ScaleB(1, -53));

    private static BigInteger Exact(double value)
    {
        var bits = BitConverter.DoubleToInt64Bits(value);
        var magnitude = bits & long.MaxValue;
        var exponent = (int)(magnitude >> 52);
        var significand = magnitude & ((1L << 52) - 1);
        if (exponent != 0) { significand |= 1L << 52; }
        var integer = new BigInteger(significand) << Math.Max(0, exponent - 1);
        return bits < 0 ? -integer : integer;
    }
}
