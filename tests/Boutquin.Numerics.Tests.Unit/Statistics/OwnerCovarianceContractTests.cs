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

using Boutquin.Numerics.Statistics;
using FluentAssertions;
namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class OwnerCovarianceContractTests
{
    [Fact]
    public void Oas_KnownZeroMean_UsesRawSecondMoment()
    {
        // Constant observations (2,1): S=[[4,2],[2,1]], n=4, rho=1/2.
        var actual = new OracleApproximatingShrinkageEstimator<decimal>().Estimate(new decimal[,] { { 2, 1 }, { 2, 1 }, { 2, 1 }, { 2, 1 } });
        actual[0, 0].Should().BeApproximately(3.25m, 1e-24m);
        actual[0, 1].Should().BeApproximately(1m, 1e-24m);
        actual[1, 1].Should().BeApproximately(1.75m, 1e-24m);
    }
    [Fact]
    public void Nercome_KnownZeroMean_ConstantRowsRetainSecondMoment()
    {
        var actual = new NercomeCovarianceEstimator<double>().Estimate(new double[,] { { 2, 1 }, { 2, 1 }, { 2, 1 }, { 2, 1 } });
        actual[0, 0].Should().BeApproximately(4, 1e-12);
        actual[0, 1].Should().BeApproximately(2, 1e-12);
        actual[1, 1].Should().BeApproximately(1, 1e-12);
    }
    [Fact]
    public void Glasso_AllEntryPenalty_HasAnalyticTwoVariableSolution()
    {
        // KKT: W diagonal=S diagonal+lambda; offdiagonal=soft(1,.5).
        var actual = GraphicalLassoSolver<double>.Solve(new double[,] { { 2, 1 }, { 1, 3 } }, .5);
        actual.Covariance[0, 0].Should().BeApproximately(2.5, 1e-12);
        actual.Covariance[1, 1].Should().BeApproximately(3.5, 1e-12);
        actual.Covariance[0, 1].Should().BeApproximately(.5, 1e-12);
        actual.Precision[0, 0].Should().BeApproximately(7.0 / 17, 1e-12);
    }

    [Theory]
    [InlineData(false, 263.0 / 150, -8.0 / 25, 487.0 / 150)]
    [InlineData(true, 7339.0 / 3480, -1513.0 / 3480, 10061.0 / 3480)]
    public void Nercome_ExhaustiveSubsets_MatchesRationalProjector(bool fixedSplit, double a, double b, double d)
    {
        // Independent spectral projector: for A=[[a,b],[b,d]], H=[[a-d,2b],[2b,d-a]],
        // projection(B)=tr(B)/2 I + tr(H B)/(2*((a-d)^2+4b²))*H.
        // Enumerating all C(4,m) subsets gives scores m=3:43913/13345,
        // m=1:6217/45000, m=2:16909/20880; thus the selector chooses m=1.
        var estimator = fixedSplit ? new NercomeCovarianceEstimator<double>(.5) : new NercomeCovarianceEstimator<double>();
        var actual = estimator.Estimate(new double[,] { { 1, 0 }, { 0, 2 }, { 2, 1 }, { -1, 3 } });
        actual[0, 0].Should().BeApproximately(a, 1e-11);
        actual[0, 1].Should().BeApproximately(b, 1e-11);
        actual[1, 0].Should().BeApproximately(b, 1e-11);
        actual[1, 1].Should().BeApproximately(d, 1e-11);
    }

    [Fact]
    public void Nercome_CandidateGrid_FloorsClampsAndDeduplicatesInPublishedOrder()
    {
        var method = typeof(NercomeCovarianceEstimator<double>).GetMethod("CandidateSizes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        ((int[])method.Invoke(null, [100])!).Should().Equal(20, 40, 60, 80, 75, 85);
        ((int[])method.Invoke(null, [4])!).Should().Equal(3, 1, 2);
        ((int[])method.Invoke(null, [2])!).Should().Equal(1);
        ((int[])method.Invoke(null, [51])!).Should().Equal(14, 10, 20, 30, 40, 33);
    }

    [Fact]
    public void Nercome_SeedAndReplicationPolicy_IsReproducibleAndUsesDifferentSubsets()
    {
        double[,] rows = { { 1, 0 }, { 0, 2 }, { 2, 1 }, { -1, 3 } };
        var estimator = new NercomeCovarianceEstimator<double>(.5, 50, 7);
        var first = estimator.Estimate(rows);
        first.Should().BeEquivalentTo(estimator.Estimate(rows));
        // Six distinct subsets exhaust C(4,2), regardless of seed and requested cap.
        first[0, 1].Should().BeApproximately(-1513.0 / 3480, 1e-11);
        new NercomeCovarianceEstimator<double>(.5, 6, 123).Estimate(rows)[0, 1].Should().BeApproximately(first[0, 1], 1e-11);
        new NercomeCovarianceEstimator<double>(.5, 1, 123).Estimate(rows)[0, 1].Should().NotBeApproximately(first[0, 1], 1e-4);
    }

    [Fact]
    public void Glasso_StopUsesMeanCovarianceChangeAndDefaultRelativeScale()
    {
        // p=2: first sweep changes both offdiagonals by lambda=.0015,
        // mean deltaW=.00075 < .001*meanOffdiagS=.001: one sweep is sufficient.
        var result = GraphicalLassoSolver<double>.Solve(new double[,] { { 2, 1 }, { 1, 3 } }, .0015);
        result.Converged.Should().BeTrue();
        result.Iterations.Should().Be(1);
        // Strict inequality at equality (all values exactly binary): .5/2 == .25*1.
        var strict = GraphicalLassoSolver<double>.Solve(new double[,] { { 2, 1 }, { 1, 3 } }, .5,
            new GraphicalLassoOptions<double>(MaxIterations: 1, Tolerance: .25));
        strict.Converged.Should().BeFalse();
        // Scale S and lambda together: stopping iteration is invariant.
        var scaled = GraphicalLassoSolver<double>.Solve(new double[,] { { 200, 100 }, { 100, 300 } }, .15);
        scaled.Iterations.Should().Be(1);
    }

    [Fact]
    public void Glasso_DiagonalScaleBoundaryAndWeightedDiagonalRespectKkt()
    {
        var result = GraphicalLassoSolver<double>.Solve(new double[,] { { 2, 0 }, { 0, 3 } }, .5,
            new GraphicalLassoOptions<double>(Weights: new double[,] { { 2, 1 }, { 1, 0 } }));
        result.Covariance[0, 0].Should().BeApproximately(3, 1e-12);
        result.Covariance[1, 1].Should().BeApproximately(3, 1e-12);
        result.Converged.Should().BeTrue();
        result.Iterations.Should().Be(1);
        var scalar = GraphicalLassoSolver<double>.Solve(new double[,] { { 2 } }, .5);
        scalar.Covariance[0, 0].Should().BeApproximately(2.5, 1e-12);
        scalar.Converged.Should().BeTrue();
    }

    [Fact]
    public void Nercome_SelectionScore_SquaresTheAveragedResidual()
    {
        // Exact projector enumeration: ||mean residual||F²=6217/45000,
        // whereas mean(||residual||F²)=283/450. They are different objectives.
        var method = typeof(NercomeCovarianceEstimator<double>).GetMethod("AverageSplits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var estimator = new NercomeCovarianceEstimator<double>();
        var result = ((double[,] Estimate, double Score))method.Invoke(estimator,
            [new double[,] { { 1, 0 }, { 0, 2 }, { 2, 1 }, { -1, 3 } }, 1, new Boutquin.Numerics.Random.Xoshiro256StarStarRandomSource(0)])!;
        // Production compares the norm to avoid overflow/underflow; squaring it recovers eq. 4.8.
        (result.Score * result.Score).Should().BeApproximately(6217.0 / 45000, 1e-12);
    }
}
