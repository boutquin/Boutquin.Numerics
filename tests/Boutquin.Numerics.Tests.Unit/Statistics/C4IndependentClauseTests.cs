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

/// <summary>Exact arithmetic controls for individual estimator predicates.</summary>
public sealed class C4IndependentClauseTests
{
    [Fact]
    public void ScaledIdentity_InteriorIntensity_MatchesExactRationalPanel()
    {
        // JMVA eq.14: S_T=[[5,1],[1,1]], m=3, normalized d²=5,
        // b-bar²=3, delta=3/5. Apply the documented 4/3 output scale.
        // Distinguishes LWID-RAWB, LWID-DISTANCE, LWID-INTENSITY and
        // LWID-OUTPUT from an extra rho subtraction or Bessel intensity.
        decimal[,] observations = { { -3, -1 }, { -1, 1 }, { 1, -1 }, { 3, 1 } };
        decimal[,] expected = { { 76m / 15, 8m / 15 }, { 8m / 15, 44m / 15 } };
        AssertMatrix(new LedoitWolfShrinkageEstimator<decimal>().Estimate(observations), expected);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void ConstantCorrelation_ClippedAndInteriorIntensity_MatchesRationalCrossMoments(int repeats)
    {
        // Honey Appendix B: S_T has diagonal 5 and offdiagonals 2,0,-1;
        // r-bar=1/15, pi=156, rho=724/15, gamma=28/3.
        // Unclipped delta=404/(35*T): 101/35 at T=4, 101/140 at T=16.
        // LWCC-RHOOFF, LWCC-KAPPA, LWCC-HIGH, LWCC-INTERIOR: omitting
        // rho or its diagonal cancellation changes the interior result.
        decimal[,] rows = { { -3, -3, -1 }, { -1, 1, 3 }, { 1, 3, -3 }, { 3, -1, 1 } };
        var observations = new decimal[4 * repeats, 3];
        for (var row = 0; row < observations.GetLength(0); row++)
        {
            for (var column = 0; column < 3; column++)
            {
                observations[row, column] = rows[row % 4, column];
            }
        }

        decimal[,] expected = repeats == 1
            ? new decimal[,] { { 20m / 3, 4m / 9, 4m / 9 }, { 4m / 9, 20m / 3, 4m / 9 }, { 4m / 9, 4m / 9, 20m / 3 } }
            : new decimal[,] { { 16m / 3, 268m / 315, 404m / 1575 }, { 268m / 315, 16m / 3, -64m / 1575 }, { 404m / 1575, -64m / 1575, 16m / 3 } };
        AssertMatrix(new LedoitWolfConstantCorrelationEstimator<decimal>().Estimate(observations), expected);
    }

    [Fact]
    public void SingleFactor_InteriorIntensity_MatchesRationalMarketCrossMoments()
    {
        // LW2003 eq.8, Lemmas1–3: equal-weight market, pi=76,
        // rho=107566/1521, gamma=5140/1521, delta=803/3084.
        // LWSF-ROFF, LWSF-RDIAG, LWSF-K, LWSF-OUTPUT: dropping the
        // market-variance derivative or rho entirely changes [1,2].
        decimal[,] observations =
        {
            { 3, 1, 2 }, { -3, -1, -2 }, { 1, -1, 0 },
            { -1, 1, 0 }, { 2, 3, -1 }, { -2, -3, 1 },
        };
        decimal[,] expected =
        {
            { 28m / 5, 5095m / 1542, 8m / 5 },
            { 5095m / 1542, 22m / 5, -149m / 20046 },
            { 8m / 5, -149m / 20046, 2 },
        };
        AssertMatrix(new LedoitWolfSingleFactorEstimator<decimal>().Estimate(observations), expected);
    }

    [Theory]
    [InlineData(.5)]
    [InlineData(.6)]
    public void DoublySparseVariant_EqualLoadingSignal_PreservesAnalyticCovariance(double threshold)
    {
        // Selected library variant only; no published DSCE parity claim.
        // Four orthogonal balanced sign columns imply S=(16/15)(I+4J).
        // Correlation eigenvalues are 13/5,1/5,1/5. The MP edge
        // (1+sqrt(3/16))² is below 13/5 and above 1/5.
        // The signal vector has every magnitude 1/sqrt(3): .5 keeps all;
        // .6 removes all and must retain the original direction. Noise
        // averaging leaves the equal 1/5 pair unchanged. Thus every output
        // diagonal is 16/3 and every offdiagonal is 64/15, independently of
        // eigensolver orientation. Distinguishes dropped signal on empty support.
        var observations = new double[16, 3];
        for (var row = 0; row < 16; row++)
        {
            var common = (row & 1) == 0 ? -1 : 1;
            for (var column = 0; column < 3; column++)
            {
                var individual = (row & (1 << (column + 1))) == 0 ? -1 : 1;
                observations[row, column] = 2 * common + individual;
            }
        }

        var actual = new DoublySparseEstimator<double>(threshold).Estimate(observations);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual[i, j].Should().BeApproximately(i == j ? 16.0 / 3 : 64.0 / 15, 1e-10);
            }
        }
    }

    [Fact]
    public void DoublySparseVariant_UnequalNoiseSpectrum_MatchesClosedFormCovariance()
    {
        // DSCE-012 selected variant. Orthogonal balanced signs and this exact
        // Cholesky factor yield correlation C=[[1,.8,.6],[.8,1,.6],[.6,.6,1]].
        // Its antisymmetric eigenvalue is 1/5. The other two are
        // (14 +/- sqrt(88))/10. Only the larger exceeds the MP edge for T=16.
        // Averaging the two unequal noise eigenvalues gives
        // mu=(16-sqrt(88))/20. With r=(sqrt(88)-4)/6 and
        // v=(1,1,r)/sqrt(2+r*r), the cleaned matrix is
        // mu*I+(lambda-mu)*v*v'. Original standard deviations are one, so
        // this reconstructed matrix is the returned covariance. The selected
        // DSCE variant does not normalize the cleaned diagonal to one.
        // This closed-form 2x2 characteristic equation is independent of the
        // production eigensolver, eigenvalue loop and covariance generator.
        var observations = new double[16, 3];
        var scale = Math.Sqrt(15.0 / 16);
        for (var row = 0; row < 16; row++)
        {
            var a = (row & 1) == 0 ? -1 : 1;
            var b = (row & 2) == 0 ? -1 : 1;
            var c = (row & 4) == 0 ? -1 : 1;
            observations[row, 0] = scale * a;
            observations[row, 1] = scale * (.8 * a + .6 * b);
            observations[row, 2] = scale * (.6 * a + .2 * b + Math.Sqrt(.6) * c);
        }

        var actual = new DoublySparseEstimator<double>(0).Estimate(observations);
        double[,] expected =
        {
            { 1.0466998209110976, .7157413968934405, .6418806447200495 },
            { .7157413968934405, 1.0466998209110976, .6418806447200495 },
            { .6418806447200495, .6418806447200495, .9066003581778053 },
        };
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], 1e-10);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void NeweyWest_NonzeroLags_MatchesExactToeplitzQuadraticForm(int lags)
    {
        // NW1987 eqs4-5, scalar mean specialization. For x=(-3,-1,1,3),
        // x'x=20, lag-one product sum=5, lag-two product sum=-6.
        // x'B*x/16 = (20+5)/16 at L=1 and
        // (20+2*(2/3)*5+2*(1/3)*(-6))/16=17/12 at L=2.
        // The fixed T divisor at every lag, L+1 Bartlett denominator,
        // two symmetric offdiagonal contributions and final mean scaling
        // are all necessary. No automatic bandwidth contract is assumed.
        decimal[] observations = [-3, -1, 1, 3];
        var expected = lags == 1 ? 25m / 16 : 17m / 12;
        NeweyWestVariance<decimal>.MeanVariance(observations, lags)
            .Should().BeApproximately(expected, 0.00000000000000000001m);
    }

    [Fact]
    public void Nercome_FixedSplit_UsesFirstBasisAndSecondQuadraticForms()
    {
        // Lam eq4.6: exhaust the six distinct first subsets at m=2.
        // The independent 2x2 spectral-projector formula gives the rational mean below.
        decimal[,] observations = { { -1, -1 }, { 1, 1 }, { -2, -1 }, { 2, 1 } };
        decimal[,] expected = { { 233m / 100, 36m / 25 }, { 36m / 25, 117m / 100 } };
        AssertMatrix(new NercomeCovarianceEstimator<decimal>(0.5m).Estimate(observations), expected);
    }

    private static void AssertMatrix(decimal[,] actual, decimal[,] expected)
    {
        for (var i = 0; i < expected.GetLength(0); i++)
        {
            for (var j = 0; j < expected.GetLength(1); j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], 0.00000000000000000001m);
            }
        }
    }
}
