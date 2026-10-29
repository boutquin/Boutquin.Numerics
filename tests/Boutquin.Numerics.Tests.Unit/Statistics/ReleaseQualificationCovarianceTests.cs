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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class ReleaseQualificationCovarianceTests
{
    [Fact]
    public void GraphicalLasso_OuterToleranceWithExhaustedInnerSolve_StopsRepeatingWork()
    {
        const int count = 5;
        var sample = new double[count, count];
        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < count; j++)
            {
                sample[i, j] = i == j ? 1 : .99;
            }
        }

        // The finite inner budget cannot finish these correlated regressions.
        // Once the outer iterate meets its requested tolerance, stop spending
        // outer sweeps while retaining the honest inner-exhaustion status.
        var result = GraphicalLassoSolver<double>.Solve(sample, .001,
            new GraphicalLassoOptions<double>(MaxIterations: 400));
        result.Converged.Should().BeFalse();
        result.Iterations.Should().BeLessThan(20,
            "an outer iterate within tolerance must stop instead of consuming the outer budget");
        var precision = result.Precision;
        var covariance = result.Covariance;
        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < count; j++)
            {
                var product = 0.0;
                for (var k = 0; k < count; k++)
                {
                    product += precision[i, k] * covariance[k, j];
                }

                product.Should().BeApproximately(i == j ? 1 : 0, 1e-11);
            }
        }
    }

    [Fact]
    public void GraphicalLasso_ZeroPenalty_MatchesShermanMorrisonInverse()
    {
        double[] direction = [1, 2, -1, 3, -2, 4];
        const double ridge = 1e-8;
        var sample = new double[6, 6];
        for (var i = 0; i < 6; i++)
        {
            for (var j = 0; j < 6; j++)
            {
                sample[i, j] = direction[i] * direction[j] + (i == j ? ridge : 0);
            }
        }

        var result = GraphicalLassoSolver<double>.Solve(sample, 0);
        result.Converged.Should().BeTrue();
        for (var i = 0; i < 6; i++)
        {
            for (var j = 0; j < 6; j++)
            {
                result.Covariance[i, j].Should().Be(sample[i, j]);
                var expected = ((i == j ? 1.0 : 0) - direction[i] * direction[j] / (35 + ridge)) / ridge;
                result.Precision[i, j].Should().BeApproximately(expected, 30);
            }
        }

        var bic = GraphicalLassoSolver<double>.SolveWithBic(sample, 50, [0, 1e-9, 1e-6]);
        CholeskyDecomposition<double>.Decompose(bic.Precision).Should().NotBeNull();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphicalLasso_DiagonalProblem_PreservesIndependentSmallVariance(bool diagonalPenalty)
    {
        var small = Math.ScaleB(1.0, -300);
        var weight = Math.ScaleB(1.0, 400);
        double[,] sample = { { 1, 0 }, { 0, small } };
        double[,] weights = diagonalPenalty ? new double[,] { { weight, 0 }, { 0, 0 } }
            : new double[,] { { 0, weight }, { weight, 0 } };
        var actual = GraphicalLassoSolver<double>.Solve(sample, weight,
            new GraphicalLassoOptions<double>(Weights: weights));
        actual.Covariance[1, 1].Should().Be(small);
        actual.Precision[1, 1].Should().Be(1 / small);
        actual.Precision[0, 0].Should().Be(diagonalPenalty ? Math.ScaleB(1.0, -800) : 1);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(1e10)]
    [InlineData(1e50)]
    public void GraphicalLasso_BicCountsSmallPrecisionBlockEdges(double scale)
    {
        double[,] sample = { { scale, .1 * scale, 0, 0 }, { .1 * scale, scale, 0, 0 },
            { 0, 0, 1, .3 }, { 0, 0, .3, 1 } };
        double[,] weights = { { 1, 1, 0, 0 }, { 1, 1, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
        // Analytic 2x2 block inverses give BIC(lambda0)-BIC(lambda1)=7.15312354,
        // independent of scale. There are four directed edges at zero and two at .11*scale.
        var result = GraphicalLassoSolver<double>.SolveWithBic(sample, 100, [0, .11 * scale],
            new GraphicalLassoOptions<double>(Weights: weights));
        result.Lambda.Should().Be(.11 * scale);
    }

    [Fact]
    public void GraphicalLasso_AnalyticSparseKktPoint_RetainsExactZeroEdge()
    {
        double[,] sample = { { 141.0 / 257, 50.0 / 257 + .05, -15.0 / 257 + .025 },
            { 50.0 / 257 + .05, 200.0 / 257, -60.0 / 257 - .15 },
            { -15.0 / 257 + .025, -60.0 / 257 - .15, 275.0 / 257 } };
        double[,] weights = { { 0, 1, 2 }, { 1, 0, 3 }, { 2, 3, 0 } };
        var result = GraphicalLassoSolver<double>.Solve(sample, .05,
            new GraphicalLassoOptions<double>(Weights: weights));
        result.Precision[0, 2].Should().Be(0);
        result.Precision[2, 0].Should().Be(0);
        result.Precision[0, 1].Should().BeApproximately(-.5, .001);
        result.Precision[1, 2].Should().BeApproximately(.3, .001);
    }

    [Fact]
    public void GraphicalLasso_TenVariableKktChain_PreservesSparseOptimumAndInverse()
    {
        const int count = 10;
        var determinants = new double[count + 1];
        determinants[0] = 1;
        determinants[1] = 2;
        for (var i = 2; i <= count; i++)
        {
            determinants[i] = 2 * determinants[i - 1] - .16 * determinants[i - 2];
        }

        var sample = new double[count, count];
        var weights = new double[count, count];
        for (var i = 0; i < count; i++)
        {
            for (var j = i; j < count; j++)
            {
                // Cofactor formula for inverse of Theta=tridiag(-.4,2,-.4).
                var covariance = Math.Pow(.4, j - i) * determinants[i] * determinants[count - j - 1] / determinants[count];
                sample[i, j] = sample[j, i] = covariance + (j == i + 1 ? .05 : 0);
                weights[i, j] = weights[j, i] = i == j ? 0 : 1;
            }
        }

        // W-S=-.05*sign(Theta) on the chain and zero on absent edges: exact KKT certificate.
        var result = GraphicalLassoSolver<double>.Solve(sample, .05,
            new GraphicalLassoOptions<double>(Weights: weights));
        result.Converged.Should().BeTrue();
        var precision = result.Precision;
        var estimatedCovariance = result.Covariance;
        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < count; j++)
            {
                var expected = i == j ? 2 : Math.Abs(i - j) == 1 ? -.4 : 0;
                precision[i, j].Should().BeApproximately(expected, .001);
                if (Math.Abs(i - j) > 1)
                {
                    precision[i, j].Should().Be(0);
                }

                var product = 0.0;
                for (var k = 0; k < count; k++)
                {
                    product += precision[i, k] * estimatedCovariance[k, j];
                }

                product.Should().BeApproximately(i == j ? 1 : 0, 1e-12);
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CorrelationEstimatorOutput_IsExactlySymmetricAndAcceptedByPca(int family)
    {
        var random = new System.Random(7);
        var rows = new double[50, 5];
        for (var i = 0; i < 50; i++)
        {
            for (var j = 0; j < 5; j++)
            {
                rows[i, j] = (random.NextDouble() - .5) * (j + 1);
            }
        }

        ICovarianceEstimator<double> estimator = family switch
        {
            0 => new DenoisedCovarianceEstimator<double>(),
            1 => new DetonedCovarianceEstimator<double>(),
            2 => new TracyWidomDenoisedCovarianceEstimator<double>(),
            _ => new DoublySparseEstimator<double>(),
        };
        var actual = estimator.Estimate(rows);
        for (var i = 0; i < 5; i++)
        {
            for (var j = 0; j < 5; j++)
            {
                actual[i, j].Should().Be(actual[j, i]);
            }
        }

        PrincipalComponentAnalysis<double>.Decompose(actual).Should().NotBeNull();
    }

    [Fact]
    public void Nercome_GradedCoupledColumns_SelectsExactProjectorMinimum()
    {
        decimal[,] rows =
        {
            { -248e7m, -2874e-11m }, { -863e7m, -679e-11m }, { -2934e7m, -2090e-11m },
            { 3708e7m, 2261e-11m }, { -802e7m, 1494e-11m }, { -1702e7m, 2614e-11m }, { 687e7m, 1887e-11m },
        };
        // Exact rational spectral-projector enumeration over all subsets gives scores
        // m=5:10531.91, m=1:852945.97, m=2:2945.90, m=4:16865.86, m=3:16925.16.
        var actual = new NercomeCovarianceEstimator<decimal>().Estimate(rows);
        actual[0, 1].Should().BeApproximately(125.38852836696228m, 1e-7m);
    }

    [Fact]
    public void LedoitWolf_OrdinarySmallDecimalReturns_PreservesScaleEquivariance()
    {
        var random = new System.Random(31);
        var rows = new decimal[500, 20];
        var scaled = new decimal[500, 20];
        for (var i = 0; i < 500; i++)
        {
            var factor = random.Next(-1000, 1001) / 1000m;
            for (var j = 0; j < 20; j++)
            {
                rows[i, j] = (.4m * factor + random.Next(-1000, 1001) / 1000m) * .000007m;
                scaled[i, j] = rows[i, j] * 1000;
            }
        }

        var small = new LedoitWolfShrinkageEstimator<decimal>().Estimate(rows);
        var large = new LedoitWolfShrinkageEstimator<decimal>().Estimate(scaled);
        for (var i = 0; i < 20; i++)
        {
            for (var j = 0; j < 20; j++)
            {
                small[i, j].Should().BeApproximately(large[i, j] / 1000000, 1e-25m);
            }
        }
    }

    [Fact]
    public void Poet_ThresholdAboveDecimalRange_RemovesRepresentableResidual()
    {
        const decimal scale = 1e12m;
        decimal[,] rows = { { 2 * scale, 2 * scale, 0 }, { -2 * scale, -2 * scale, 0 },
            { scale, -scale, 0 }, { -scale, scale, 0 } };
        var actual = new PoetCovarianceEstimator<decimal>(1, 1e14m).Estimate(rows);
        actual[0, 0].Should().BeApproximately(10m / 3 * scale * scale, 1e8m);
        actual[0, 1].Should().BeApproximately(8m / 3 * scale * scale, 1e8m);
        actual[2, 2].Should().Be(0);
    }

    [Fact]
    public void Detoned_ExactlyCollinearColumns_UsesIdentityResidualFallback()
    {
        var rows = new double[50, 5];
        for (var i = 0; i < 50; i++)
        {
            for (var j = 0; j < 5; j++)
            {
                rows[i, j] = (i - 25) * (j + 1);
            }
        }

        var actual = new DetonedCovarianceEstimator<double>(1).Estimate(rows);
        for (var i = 0; i < 5; i++)
        {
            for (var j = 0; j < 5; j++)
            {
                if (i != j)
                {
                    actual[i, j].Should().BeApproximately(0, 1e-10);
                }
            }
        }
    }

    [Fact]
    public void Qis_ResolvedSmallPositiveEigenvalue_DoubleMatchesAppendixA()
    {
        // Disjoint centered rows give S=diag(.4,.1,4e-15) exactly in real arithmetic.
        // Independent 100-digit evaluation of Ledoit-Wolf Appendix A, p=3,n=5:
        // eigenvalues after shrinkage = (.34035567241337946,.15964432758661493,9.6054223848763454e-15).
        double[,] rows = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, .5, 0 }, { 0, -.5, 0 }, { 0, 0, 1e-7 }, { 0, 0, -1e-7 } };
        var actual = new QuadraticInverseShrinkageEstimator<double>().Estimate(rows);

        actual[0, 0].Should().BeApproximately(.3403556724133794636, 1e-12);
        actual[1, 1].Should().BeApproximately(.1596443275866149310, 1e-12);
        actual[2, 2].Should().BeApproximately(9.605422384876345409e-15, 1e-25);
        actual[0, 2].Should().Be(0);
    }

    [Fact]
    public void Qis_ResolvedSmallPositiveEigenvalue_DecimalMatchesAppendixA()
    {
        decimal[,] rows = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, .5m, 0 }, { 0, -.5m, 0 }, { 0, 0, .0000001m }, { 0, 0, -.0000001m } };
        var actual = new QuadraticInverseShrinkageEstimator<decimal>().Estimate(rows);

        actual[0, 0].Should().BeApproximately(.3403556724133794636042962525m, 1e-20m);
        actual[1, 1].Should().BeApproximately(.1596443275866149309733188711m, 1e-20m);
        actual[2, 2].Should().BeApproximately(.0000000000000096054223848763m, 1e-27m);
        actual[0, 2].Should().Be(0);
    }

    [Theory]
    [InlineData(0.0, 17.0 / 15, 8.0 / 15, 4.0 / 15, 11.0 / 15)]
    [InlineData(0.5, 37.0 / 30, 19.0 / 30, -2.0 / 15, 8.0 / 15)]
    [InlineData(1.0, 17.0 / 15, 8.0 / 15, 4.0 / 15, 11.0 / 15)]
    public void DoublySparse_AnalyticSpectrum_ThresholdsSignalAndRetainsNoiseProjector(
        double threshold, double diagonal, double pair, double cross, double last)
    {
        // R=[[1,.7,.2],[.7,1,.2],[.2,.2,1]] has eigenvalues(1.8,.9,.3),
        // leading v=(2,2,1)/3. At T=64 the MP edge is <1.8 and >.9.
        // Noise mean=.6, so Rclean=1.8*u*u' + .6*(I-v*v').
        // threshold=.5 makes u=(1,1,0)/sqrt(2); threshold=1 invokes the documented fallback.
        var rows = ExactCorrelationPanel();
        var actual = new DoublySparseEstimator<double>(threshold).Estimate(rows);
        double[,] expected = { { diagonal, pair, cross }, { pair, diagonal, cross }, { cross, cross, last } };
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], 1e-12);
            }
        }
    }

    private static double[,] ExactCorrelationPanel()
    {
        const int count = 64;
        var rows = new double[count, 3];
        var scale = Math.Sqrt(63.0 / 64);
        var l11 = Math.Sqrt(.51);
        var l21 = .06 / l11;
        var l22 = Math.Sqrt(.96 - l21 * l21);
        for (var r = 0; r < count; r++)
        {
            // Three balanced, mutually orthogonal Walsh columns: Z'Z=64I.
            var a = (r & 1) == 0 ? 1.0 : -1.0;
            var b = (r & 2) == 0 ? 1.0 : -1.0;
            var c = (r & 4) == 0 ? 1.0 : -1.0;
            rows[r, 0] = scale * a;
            rows[r, 1] = scale * (.7 * a + l11 * b);
            rows[r, 2] = scale * (.2 * a + l21 * b + l22 * c);
        }

        return rows;
    }
}
