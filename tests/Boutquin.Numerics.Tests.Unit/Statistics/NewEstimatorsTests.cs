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

public sealed class NewEstimatorsTests
{
    [Fact]
    public void QisEstimator_ProducesSymmetricMatrix()
    {
        var returns = MakeFactorReturns(150, 6, seed: 1);
        var cov = new QuadraticInverseShrinkageEstimator().Estimate(returns);
        AssertSymmetric(cov);
    }

    [Fact]
    public void QisEstimator_DecimalPathPreservesSubDoubleInputPerturbation()
    {
        var baseline = new decimal[40, 3];
        var perturbed = new decimal[40, 3];
        for (var i = 0; i < 40; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var value = (i % (5 + j) - (2 + j)) / 100m + j / 1000m;
                baseline[i, j] = value;
                perturbed[i, j] = value;
            }
        }

        // This increment is below a double ULP at the 0.01 input scale but is
        // representable by decimal and must survive the estimator pipeline.
        perturbed[17, 1] += 0.000000000000000000000001m;

        var before = new QuadraticInverseShrinkageEstimator<decimal>().Estimate(baseline);
        var after = new QuadraticInverseShrinkageEstimator<decimal>().Estimate(perturbed);

        // Golden from the LW (2022) Appendix A code evaluated in numpy on the same input.
        ((double)after[1, 1]).Should().BeApproximately(0.0003035767131058607, 1e-18);
        after[1, 1].Should().NotBe(before[1, 1]);
        after[0, 1].Should().NotBe(before[0, 1]);
    }

    [Fact]
    public void QisEstimator_ConstantAssetKeepsZeroVarianceAndOthersStayPositive()
    {
        // A constant asset puts an exact zero inside the non-null eigenvalue block
        // (N = 4 <= T - 1). The paper's 1/lambda would divide by it; the estimator keeps
        // that direction at zero and shrinks the remaining spectrum.
        var returns = MakeFactorReturns(60, 4, seed: 23);
        for (var i = 0; i < 60; i++)
        {
            returns[i, 2] = 0.005m;
        }

        var cov = new QuadraticInverseShrinkageEstimator<decimal>().Estimate(returns);

        AssertSymmetric(cov);
        Math.Abs(cov[2, 2]).Should().BeLessThan(1e-20m);
        cov[0, 0].Should().BeGreaterThan(0m);
        cov[1, 1].Should().BeGreaterThan(0m);
        cov[3, 3].Should().BeGreaterThan(0m);
    }

    [Fact]
    public void LedoitWolfConstantCorrelation_PreservesDiagonal()
    {
        var returns = MakeFactorReturns(100, 4, seed: 2);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var lwcc = new LedoitWolfConstantCorrelationEstimator().Estimate(returns);

        for (var i = 0; i < 4; i++)
        {
            ((double)Math.Abs(sample[i, i] - lwcc[i, i])).Should().BeLessThan(1e-12);
        }
    }

    [Fact]
    public void LedoitWolfSingleFactor_PreservesDiagonal()
    {
        var returns = MakeFactorReturns(120, 5, seed: 3);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var lwfm = new LedoitWolfSingleFactorEstimator().Estimate(returns);

        for (var i = 0; i < 5; i++)
        {
            ((double)Math.Abs(sample[i, i] - lwfm[i, i])).Should().BeLessThan(1e-12);
        }
    }

    [Fact]
    public void OracleApproximatingShrinkage_StaysWithinIdentityAndSample()
    {
        var returns = MakeFactorReturns(80, 4, seed: 4);
        var sample = new SampleCovarianceEstimator().Estimate(returns);
        var oas = new OracleApproximatingShrinkageEstimator().Estimate(returns);

        // OAS off-diagonals are bounded between sample (when ρ=0) and 0 (when ρ=1).
        for (var i = 0; i < 4; i++)
        {
            for (var j = i + 1; j < 4; j++)
            {
                var sampleAbs = Math.Abs(sample[i, j]);
                var oasAbs = Math.Abs(oas[i, j]);
                oasAbs.Should().BeLessThanOrEqualTo(sampleAbs + 1e-20m);
            }
        }
    }

    [Fact]
    public void TracyWidomDenoised_ProducesPsd()
    {
        var returns = MakeFactorReturns(200, 5, seed: 5);
        var cov = new TracyWidomDenoisedCovarianceEstimator().Estimate(returns);
        AssertSymmetric(cov);
        var eigenvalues = JacobiEigenDecomposition<decimal>.Decompose(cov).Values;
        eigenvalues.Should().OnlyContain(value => value >= -1e-12m,
            "symmetry alone does not imply positive semidefiniteness");
    }

    // F-L7-01 (arena 2026-06-29-0803-1f9f0c): the Tracy-Widom finite-N shift
    // must depend on BOTH dimensions (observations t and variables n), per
    // Johnstone (2001)'s sigma_np = (sqrt(n-1)+sqrt(p)).(1/sqrt(n-1)+1/sqrt(p))^(1/3).
    // The pre-fix code used a fixed product N^(-2/3).2.02.1.21 whose shift term
    // depends ONLY on the variable count n — so holding n fixed and changing the
    // observation count t left the shift identical. This test pins the bug:
    // with n fixed, two different observation counts must give DIFFERENT shifts.
    [Fact]
    public void TracyWidomThreshold_ShiftDependsOnObservationCount()
    {
        const int n = 20;

        var shiftFew = TracyWidomDenoisedCovarianceEstimator<double>.TracyWidomShift(t: 40, n);
        var shiftMany = TracyWidomDenoisedCovarianceEstimator<double>.TracyWidomShift(t: 200, n);

        // The fixed-product code returns the same shift for both (n is identical);
        // Johnstone's sigma_np makes the q=2 regime fluctuate more than the q=10
        // regime, so the shift must STRICTLY decrease as observations grow.
        shiftMany.Should().BeLessThan(shiftFew,
            "Johnstone (2001) sigma_np shrinks the finite-N shift as the sample grows for fixed variable count");
    }

    // Independent derivation pin: the shift must equal Johnstone (2001)'s
    // sigma_np / t . F1^-1(0.99) on the correlation-eigenvalue scale. Theorem 1.1
    // centres/scales the largest eigenvalue by mu_np and sigma_np (eqs. 1.3-1.4), so
    // dividing by t (the observation count) maps the relation onto the correlation
    // scale. F1^-1(0.99) = 2.0234 (2.02 in Johnstone's Table 1); the TW mean is
    // already carried by the centering mu_np, so no |E[TW1]| = 1.21 factor enters.
    // Values computed in Python, NOT read back from C# output.
    [Theory]
    [InlineData(120, 8, 0.17687077)]   // small_well_conditioned regime, q=15
    [InlineData(60, 30, 0.30121121)]   // moderate regime, q=2
    [InlineData(40, 50, 0.45169925552843937)]   // overconcentrated regime, q=0.8
    public void TracyWidomShift_MatchesJohnstoneSigmaNp(int t, int n, double expectedShift)
    {
        var shift = TracyWidomDenoisedCovarianceEstimator<double>.TracyWidomShift(t, n);

        shift.Should().BeApproximately(expectedShift, 1e-7,
            "the shift must implement Johnstone (2001) sigma_np / t . F1^-1(0.99)");
    }

    [Fact]
    public void ReturnsMatrix_AcceptsBothLayouts()
    {
        decimal[,] tn =
        {
            { 0.01m, 0.02m },
            { -0.01m, 0.005m },
            { 0.015m, -0.005m },
            { -0.005m, 0.01m },
        };

        decimal[][] assetMajor =
        [
            [0.01m, -0.01m, 0.015m, -0.005m],
            [0.02m, 0.005m, -0.005m, 0.01m],
        ];

        ICovarianceEstimator estimator = new SampleCovarianceEstimator();
        var fromTN = estimator.Estimate(new ReturnsMatrix(tn));
        var fromAM = estimator.Estimate(new ReturnsMatrix(assetMajor));

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                ((double)Math.Abs(fromTN[i, j] - fromAM[i, j])).Should().BeLessThan(1e-20);
            }
        }
    }

    [Fact]
    public void DeflatedSharpeRatio_FromReturns_MatchesManualComputation()
    {
        // Exact moments: mean=5/2, population variance=5/4, skew=0,
        // raw kurtosis=41/25. One trial sets the threshold to zero;
        // the sampling error gives z=sqrt(25/3), hence p=1-Phi(z).
        var fromReturns = DeflatedSharpeRatio.ComputeFromReturns([1m, 2m, 3m, 4m], numTrials: 1, tradingDaysPerYear: 1);
        fromReturns.PValue.Should().BeApproximately(0.0019462085613893136m, 1e-12m);
        fromReturns.DeflatedSharpe.Should().BeApproximately(2.236067977499789696m, 1e-15m);
        fromReturns.NumTrials.Should().Be(1);
    }

    private static decimal[,] MakeFactorReturns(int t, int n, int seed)
    {
        var rng = new System.Random(seed);
        var returns = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            var factor = rng.NextDouble() * 0.02 - 0.01;
            for (var j = 0; j < n; j++)
            {
                var idio = rng.NextDouble() * 0.002 - 0.001;
                returns[i, j] = (decimal)(factor + idio);
            }
        }

        return returns;
    }

    private static void AssertSymmetric(decimal[,] m)
    {
        var n = m.GetLength(0);
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                ((double)Math.Abs(m[i, j] - m[j, i])).Should().BeLessThan(1e-10);
            }
        }
    }
}
