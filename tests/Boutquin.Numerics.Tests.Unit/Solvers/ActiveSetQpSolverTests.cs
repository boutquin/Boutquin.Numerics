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

using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Unit tests for <see cref="ActiveSetQpSolver"/> and <see cref="ActiveSetQpSolver{T}"/>.
/// All tests use the decimal facade or direct generic instantiation at T=decimal/double.
/// </summary>
public sealed class ActiveSetQpSolverTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SolveVariance_AsymmetricCovariance_RejectsEitherTriangle(bool meanVariance, bool upperTriangle)
    {
        var covariance = new double[,] { { 1, 0 }, { 0, 4 } };
        covariance[upperTriangle ? 0 : 1, upperTriangle ? 1 : 0] = 0.8;

        Action solve = () =>
        {
            if (meanVariance)
            {
                ActiveSetQpSolver<double>.SolveMeanVariance(covariance, new double[2], 1, 0, 1);
            }
            else
            {
                ActiveSetQpSolver<double>.SolveMinVariance(covariance, 0, 1);
            }
        };

        solve.Should().Throw<ArgumentException>().WithParameterName("covariance")
            .WithMessage("*symmetric*");
    }

    [Fact]
    public void SolveMeanVariance_OverflowingLpBudget_Throws()
    {
        Action solve = () => ActiveSetQpSolver<double>.SolveMeanVariance(
            new double[,] { { 1, 0 }, { 0, 1 } }, new[] { 2.0, 1.0 }, 0, -1e308, 1e308);

        solve.Should().Throw<InvalidOperationException>().WithMessage("*non-finite*");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SolveVariance_SubnormalCovarianceScale_PreservesFiniteOptimum(bool meanVariance)
    {
        var covariance = new double[,] { { 1e-309, 0 }, { 0, 1e-309 } };
        var weights = meanVariance
            ? ActiveSetQpSolver<double>.SolveMeanVariance(covariance, new double[2], 1, 0, 1)
            : ActiveSetQpSolver<double>.SolveMinVariance(covariance, 0, 1);

        // The inverse overflows, but the constrained optimum is representable.
        // Scaling the budget direction removes the need to form that inverse.
        weights.Should().Equal(0.5, 0.5);
    }

    [Theory]
    [InlineData(1e308)]
    [InlineData(-1e308)]
    public void SolveMeanVariance_LargeCommonMeanShift_PreservesOptimum(double commonMean)
    {
        var covariance = new double[,] { { 1, 0 }, { 0, 1 } };
        var means = new[] { commonMean, commonMean };

        Func<double[]> solve = () => ActiveSetQpSolver<double>.SolveMeanVariance(covariance, means, 1, 0, 1);
        var weights = solve.Should().NotThrow().Subject;

        // The common mean contributes a constant under sum(w)=1. Identity covariance
        // has its unique quadratic minimum at equal weights.
        weights[0].Should().BeApproximately(0.5, 1e-12);
        weights[1].Should().BeApproximately(0.5, 1e-12);
    }

    // ─── SolveMinVariance tests ────────────────────────────────────────

    /// <summary>
    /// min w′Σw s.t. 1′w=1, lb≤w≤ub is invariant to rescaling Σ by any positive constant c:
    /// the objective is scaled by c but the argmin is unchanged, since the feasible region
    /// never depends on Σ. An absolute KKT-release threshold breaks that invariance once Σ's
    /// entries fall near or below the threshold's own magnitude (here c = 1e-10 against a
    /// threshold of 1e-10): a fixed variable that should be released is never released, and
    /// the solver returns a feasible-but-suboptimal point at the scaled instance.
    /// </summary>
    [Fact]
    public void SolveMinVariance_ScaleInvariance_HoldsAtTinyCovarianceMagnitude()
    {
        var cov = new double[,]
        {
            { 1.6109243657107088, -0.57105365832069843, -0.41897734705301537, 0.026841543963564723, -0.35217703278626444 },
            { -0.57105365832069843, 2.1748598315997421, -0.51356228810903193, 0.93140947475639169, 0.92487568195784053 },
            { -0.41897734705301537, -0.51356228810903193, 3.3544150141669959, -0.26000934731927594, 0.01697151237489386 },
            { 0.026841543963564723, 0.93140947475639169, -0.26000934731927594, 1.8266763145994238, 1.0390249368736244 },
            { -0.35217703278626444, 0.92487568195784053, 0.01697151237489386, 1.0390249368736244, 1.0700339913150525 },
        };
        const double lo = 0.0;
        const double hi = 0.27803013326042803;
        const int n = 5;

        var reference = ActiveSetQpSolver<double>.SolveMinVariance(cov, lo, hi);

        var scaled = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                scaled[i, j] = cov[i, j] * 1e-10;
            }
        }

        var atScale = ActiveSetQpSolver<double>.SolveMinVariance(scaled, lo, hi);

        for (var i = 0; i < n; i++)
        {
            atScale[i].Should().BeApproximately(
                reference[i], 1e-6,
                $"weight[{i}] must match the c=1 optimum; min-variance weights are invariant " +
                "to a positive rescaling of the covariance matrix");
        }
    }

    /// <summary>
    /// The MeanVar objective max w′μ − (λ/2)w′Σw is invariant to rescaling BOTH μ and Σ by the
    /// same positive constant c, with λ unchanged: the objective itself scales by c (a positive
    /// affine transform of the argmax target), so the argmax is unchanged. This is the
    /// mean-variance scale-invariance counterpart to
    /// <see cref="SolveMinVariance_ScaleInvariance_HoldsAtTinyCovarianceMagnitude"/> — same
    /// covariance and bounds, so the same KKT releases are triggered, but through the
    /// <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/> entry point with its own
    /// scale-relative tolerance fix.
    /// </summary>
    [Fact]
    public void SolveMeanVariance_ScaleInvariance_HoldsAtTinyCovarianceAndMeanMagnitude()
    {
        var cov = new double[,]
        {
            { 1.6109243657107088, -0.57105365832069843, -0.41897734705301537, 0.026841543963564723, -0.35217703278626444 },
            { -0.57105365832069843, 2.1748598315997421, -0.51356228810903193, 0.93140947475639169, 0.92487568195784053 },
            { -0.41897734705301537, -0.51356228810903193, 3.3544150141669959, -0.26000934731927594, 0.01697151237489386 },
            { 0.026841543963564723, 0.93140947475639169, -0.26000934731927594, 1.8266763145994238, 1.0390249368736244 },
            { -0.35217703278626444, 0.92487568195784053, 0.01697151237489386, 1.0390249368736244, 1.0700339913150525 },
        };
        var means = new double[] { 0.001, 0.002, 0.0015, 0.0005, 0.0012 };
        const double lo = 0.0;
        const double hi = 0.27803013326042803;
        const double riskAversion = 1.0;
        const int n = 5;

        var reference = ActiveSetQpSolver<double>.SolveMeanVariance(cov, means, riskAversion, lo, hi);

        var scaled = new double[n, n];
        var scaledMeans = new double[n];
        for (var i = 0; i < n; i++)
        {
            scaledMeans[i] = means[i] * 1e-10;
            for (var j = 0; j < n; j++)
            {
                scaled[i, j] = cov[i, j] * 1e-10;
            }
        }

        var atScale = ActiveSetQpSolver<double>.SolveMeanVariance(scaled, scaledMeans, riskAversion, lo, hi);

        for (var i = 0; i < n; i++)
        {
            atScale[i].Should().BeApproximately(
                reference[i], 1e-6,
                $"weight[{i}] must match the c=1 optimum; mean-variance weights are invariant " +
                "to a positive joint rescaling of the covariance and mean vector");
        }
    }

    [Fact]
    public void SolveMinVariance_AllFixedExit_PreservesFeasibleLowerBound()
    {
        var covariance = new double[,] { { 1, 0, 0 }, { 0, 100, 0 }, { 0, 0, 100 } };

        var weights = ActiveSetQpSolver<double>.SolveMinVariance(covariance, 0.2, 0.7);

        weights.Should().OnlyContain(weight => weight >= 0.2 && weight <= 0.7);
        weights.Sum().Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void SolveMeanVariance_AllFixedExit_PreservesFeasibleLowerBound()
    {
        var covariance = new double[,] { { 1, 0, 0 }, { 0, 100, 0 }, { 0, 0, 100 } };
        var means = new double[] { 0, 0, 0 };

        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(covariance, means, 1.0, 0.2, 0.7);

        weights.Should().OnlyContain(weight => weight >= 0.2 && weight <= 0.7);
        weights.Sum().Should().BeApproximately(1.0, 1e-12);
    }

    /// <summary>
    /// n=7 at the exactly-feasible equal-weight frontier (max = 1/7): every asset is fixed at
    /// the upper bound and n repeated additions of 1/7 round to 0.9999999999999998 in double, a
    /// few ULPs short of exactly one. The pre-fix `fixedSum != T.One` branch reads that rounding
    /// noise as an infeasible budget and releases status -1 (the lower-bound side), which no
    /// asset holds — a no-op that spins until the 2N iteration cap throws.
    /// </summary>
    [Fact]
    public void SolveMinVariance_AllFixedAtEqualWeightFrontier_RoundingNoiseDoesNotThrow()
    {
        const int n = 7;
        var covariance = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            covariance[i, i] = 1.0 + i;
        }

        var weights = ActiveSetQpSolver<double>.SolveMinVariance(covariance, 0.0, 1.0 / n);

        weights.Should().OnlyContain(weight => weight >= 0.0 && weight <= 1.0 / n + 1e-9);
        weights.Sum().Should().BeApproximately(1.0, 1e-9);
    }

    /// <summary>
    /// Same rounding-noise frontier as <see cref="SolveMinVariance_AllFixedAtEqualWeightFrontier_RoundingNoiseDoesNotThrow"/>,
    /// exercised through <see cref="ActiveSetQpSolver{T}.SolveMeanVariance"/>'s separately-maintained
    /// copy of the same all-fixed exact-sum comparison.
    /// </summary>
    [Fact]
    public void SolveMeanVariance_AllFixedAtEqualWeightFrontier_RoundingNoiseDoesNotThrow()
    {
        const int n = 7;
        var covariance = new double[n, n];
        var means = new double[n];
        for (var i = 0; i < n; i++)
        {
            covariance[i, i] = 1.0 + i;
            means[i] = 0.1 / (i + 1);
        }

        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(covariance, means, 2.0, 0.0, 1.0 / n);

        weights.Should().OnlyContain(weight => weight >= 0.0 && weight <= 1.0 / n + 1e-9);
        weights.Sum().Should().BeApproximately(1.0, 1e-9);
    }

    [Fact]
    public void SolveMinVariance_EqualCovariance_ReturnsEqualWeights()
    {
        // Diagonal with equal variances: all assets identical risk, expect equal weights.
        var cov = DiagonalCov(1m, 1m, 1m);
        var w = ActiveSetQpSolver.SolveMinVariance(cov, minWeight: 0m, maxWeight: 1m);

        w.Should().HaveCount(3);
        foreach (var wi in w)
        {
            wi.Should().BeApproximately(1m / 3m, 1e-10m);
        }
    }

    [Fact]
    public void SolveMinVariance_UncorrelatedAssets_WeightsInverseVariance()
    {
        // σ² = [1, 4, 9] → inverse variance = [1, 1/4, 1/9]
        // Normalized: [36, 9, 4] / 49
        var cov = DiagonalCov(1m, 4m, 9m);
        var w = ActiveSetQpSolver.SolveMinVariance(cov, minWeight: 0m, maxWeight: 1m);

        var total = 36m + 9m + 4m; // 49
        w[0].Should().BeApproximately(36m / total, 1e-6m);
        w[1].Should().BeApproximately(9m / total, 1e-6m);
        w[2].Should().BeApproximately(4m / total, 1e-6m);
    }

    [Fact]
    public void SolveMinVariance_BoundConstrained_WeightsClamped()
    {
        // 3 assets; maxWeight=0.5 must keep each weight at or below 0.5.
        var cov = DiagonalCov(1m, 2m, 3m);
        var w = ActiveSetQpSolver.SolveMinVariance(cov, minWeight: 0m, maxWeight: 0.5m);

        w.Should().HaveCount(3);
        foreach (var wi in w)
        {
            wi.Should().BeLessThanOrEqualTo(0.5m + 1e-12m);
        }
    }

    [Fact]
    public void SolveMinVariance_SingleAsset_ReturnsOne()
    {
        var cov = new decimal[1, 1] { { 2m } };
        var w = ActiveSetQpSolver.SolveMinVariance(cov, minWeight: 0m, maxWeight: 1m);

        w.Should().HaveCount(1);
        w[0].Should().Be(1m);
    }

    [Fact]
    public void SolveMinVariance_WeightsSumToOne()
    {
        // 4-asset SPD matrix.
        var cov = SparseSpdCov4();
        var w = ActiveSetQpSolver.SolveMinVariance(cov, minWeight: 0m, maxWeight: 1m);

        w.Should().HaveCount(4);
        w.Sum().Should().BeApproximately(1m, 1e-10m);
    }

    // ─── SolveMeanVariance tests ───────────────────────────────────────

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_MaxReturnAllocation()
    {
        // means = [0.05, 0.10, 0.03]; best asset is index 1.
        var cov = DiagonalCov(1m, 1m, 1m);
        var means = new decimal[] { 0.05m, 0.10m, 0.03m };
        var w = ActiveSetQpSolver.SolveMeanVariance(
            cov, means, riskAversion: 0m, minWeight: 0m, maxWeight: 1m);

        w.Should().HaveCount(3);
        // With λ=0 and unconstrained max, all weight on best asset.
        w[1].Should().BeApproximately(1m, 1e-10m);
        w[0].Should().BeApproximately(0m, 1e-10m);
        w[2].Should().BeApproximately(0m, 1e-10m);
    }

    [Theory]
    [InlineData(1e-16)]
    [InlineData(1e-320)]
    public void SolveMeanVariance_TinyPositiveRiskAversion_KeepsBudgetAndBounds(double riskAversion)
    {
        var covariance = new double[2, 2] { { 1, 0 }, { 0, 1 } };
        var means = new double[] { 1, 2 };

        var weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            covariance, means, riskAversion, minWeight: 0, maxWeight: 1);

        weights.Should().OnlyContain(weight => double.IsFinite(weight));
        weights.Sum().Should().BeApproximately(1, 1e-12);
        weights.Should().OnlyContain(weight => weight >= 0 && weight <= 1);
        weights[1].Should().BeApproximately(1, 1e-12);
    }

    [Fact]
    public void SolveMeanVariance_HighRiskAversion_ConvergesToMinVar()
    {
        // Very high λ makes the return term negligible; result should approach MinVar.
        var cov = DiagonalCov(1m, 4m, 9m);
        var means = new decimal[] { 0.05m, 0.10m, 0.03m };

        var wMv = ActiveSetQpSolver.SolveMeanVariance(
            cov, means, riskAversion: 1_000_000m, minWeight: 0m, maxWeight: 1m);
        var wMin = ActiveSetQpSolver.SolveMinVariance(cov, minWeight: 0m, maxWeight: 1m);

        wMv.Should().HaveCount(3);
        for (var i = 0; i < 3; i++)
        {
            wMv[i].Should().BeApproximately(wMin[i], 1e-4m);
        }
    }

    [Fact]
    public void SolveMeanVariance_WeightsSumToOne()
    {
        var cov = DiagonalCov(1m, 2m, 3m);
        var means = new decimal[] { 0.08m, 0.12m, 0.05m };
        var w = ActiveSetQpSolver.SolveMeanVariance(
            cov, means, riskAversion: 2m, minWeight: 0m, maxWeight: 1m);

        w.Should().HaveCount(3);
        w.Sum().Should().BeApproximately(1m, 1e-10m);
    }

    [Fact]
    public void SolveMeanVariance_BoundConstrained_AllWeightsInBounds()
    {
        // maxWeight=0.4 must keep every weight at or below 0.4.
        var cov = DiagonalCov(1m, 1m, 1m);
        var means = new decimal[] { 0.08m, 0.12m, 0.05m };
        var w = ActiveSetQpSolver.SolveMeanVariance(
            cov, means, riskAversion: 1m, minWeight: 0m, maxWeight: 0.4m);

        w.Should().HaveCount(3);
        foreach (var wi in w)
        {
            wi.Should().BeLessThanOrEqualTo(0.4m + 1e-12m);
        }
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_InfeasibleMinWeight_Throws()
    {
        // n=3, minWeight=0.5 → n*minWeight=1.5 > 1: infeasible regardless of riskAversion.
        var cov = DiagonalCov(1m, 1m, 1m);
        var means = new decimal[] { 0.05m, 0.10m, 0.03m };

        var act = () => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            cov, means, riskAversion: 0m, minWeight: 0.5m, maxWeight: 1m);

        act.Should().Throw<ArgumentException>().WithParameterName("minWeight");
    }

    [Fact]
    public void SolveMeanVariance_ZeroRiskAversion_InfeasibleMaxWeight_Throws()
    {
        // n=3, maxWeight=0.2 → n*maxWeight=0.6 < 1: infeasible regardless of riskAversion.
        var cov = DiagonalCov(1m, 1m, 1m);
        var means = new decimal[] { 0.05m, 0.10m, 0.03m };

        var act = () => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            cov, means, riskAversion: 0m, minWeight: 0m, maxWeight: 0.2m);

        act.Should().Throw<ArgumentException>().WithParameterName("maxWeight");
    }

    // ─── N-5: Infeasible-bounds clamp to 1/n frontier ────────────────────

    /// <summary>
    /// AC-1: n=1, maxWeight=0.5 returns [1.0] — the n=1 early-return path is preserved.
    /// </summary>
    [Fact]
    public void SolveMinVariance_N1_MaxWeightHalf_ReturnsOne()
    {
        var cov = new decimal[1, 1] { { 2m } };
        var w = ActiveSetQpSolver<decimal>.SolveMinVariance(cov, minWeight: 0m, maxWeight: 0.5m);

        w.Should().HaveCount(1);
        w[0].Should().Be(1m);
    }

    /// <summary>
    /// AC-2: n=2, maxWeight=0.4 (n·maxWeight=0.8 &lt; 1) must NOT throw.
    /// The solver clamps effMax to 0.5 (1/n) and returns a feasible solution.
    /// </summary>
    [Fact]
    public void SolveMinVariance_InfeasibleMaxWeight_ClampsAndReturnsFeasible()
    {
        // n=2, maxWeight=0.4 → n*maxWeight=0.8 < 1 → clamp effMax to 0.5
        var cov = new decimal[2, 2] { { 1m, 0m }, { 0m, 2m } };
        var w = ActiveSetQpSolver<decimal>.SolveMinVariance(cov, minWeight: 0m, maxWeight: 0.4m);

        w.Should().HaveCount(2);
        w.Sum().Should().BeApproximately(1m, 1e-10m);
        foreach (var wi in w)
        {
            wi.Should().BeLessThanOrEqualTo(0.5m + 1e-10m); // effMax = 1/2
            wi.Should().BeGreaterThanOrEqualTo(0m - 1e-10m);
        }
    }

    /// <summary>
    /// AC-2 (minWeight variant): n=2, minWeight=0.6 (n·minWeight=1.2 &gt; 1) must NOT throw.
    /// The solver clamps effMin to 0.5 (1/n) and returns a feasible solution.
    /// </summary>
    [Fact]
    public void SolveMinVariance_InfeasibleMinWeight_ClampsAndReturnsFeasible()
    {
        // n=2, minWeight=0.6 → n*minWeight=1.2 > 1 → clamp effMin to 0.5
        var cov = new decimal[2, 2] { { 1m, 0m }, { 0m, 2m } };
        var w = ActiveSetQpSolver<decimal>.SolveMinVariance(cov, minWeight: 0.6m, maxWeight: 1m);

        w.Should().HaveCount(2);
        w.Sum().Should().BeApproximately(1m, 1e-10m);
        foreach (var wi in w)
        {
            wi.Should().BeGreaterThanOrEqualTo(0.5m - 1e-10m); // effMin = 1/2
        }
    }

    /// <summary>
    /// AC-3: SolveMeanVariance with n=2, maxWeight=0.4 must NOT throw.
    /// </summary>
    [Fact]
    public void SolveMeanVariance_InfeasibleMaxWeight_ClampsAndReturnsFeasible()
    {
        var cov = new decimal[2, 2] { { 1m, 0m }, { 0m, 2m } };
        var means = new decimal[] { 0.08m, 0.12m };

        var w = ActiveSetQpSolver<decimal>.SolveMeanVariance(
            cov, means, riskAversion: 1m, minWeight: 0m, maxWeight: 0.4m);

        w.Should().HaveCount(2);
        w.Sum().Should().BeApproximately(1m, 1e-10m);
        foreach (var wi in w)
        {
            wi.Should().BeLessThanOrEqualTo(0.5m + 1e-10m); // effMax = 1/2
            wi.Should().BeGreaterThanOrEqualTo(0m - 1e-10m);
        }
    }

    /// <summary>
    /// AC-4 (characterization): feasible call (n=4, minWeight=0, maxWeight=0.5) on a diagonal
    /// covariance returns byte-identical weights before and after the clamp fix.
    /// The clamp is a no-op here because n·maxWeight=2 ≥ 1 and n·minWeight=0 ≤ 1.
    /// </summary>
    [Fact]
    public void SolveMinVariance_FeasibleBounds_ByteIdenticalAfterFix()
    {
        // Diagonal cov diag(1, 2, 3, 4); bounds (0, 0.5) → feasible, clamp is no-op.
        var cov = new double[4, 4]
        {
            { 1.0, 0.0, 0.0, 0.0 },
            { 0.0, 2.0, 0.0, 0.0 },
            { 0.0, 0.0, 3.0, 0.0 },
            { 0.0, 0.0, 0.0, 4.0 },
        };

        // The exact optimum is (12, 6, 4, 3)/25. The 2026-06-28 characterization pinned the pre-rewrite core's
        // bits, whose w[1] = 0.23999999999999996 is one ulp below the correctly rounded 6/25; the primal active-set
        // core returns every weight correctly rounded, so the byte-identical pin is now the correctly rounded optimum.
        const double Expected0 = 0.48;
        const double Expected1 = 0.24;
        const double Expected2 = 0.16;
        const double Expected3 = 0.12;

        var w = ActiveSetQpSolver<double>.SolveMinVariance(cov, minWeight: 0.0, maxWeight: 0.5);

        w.Should().HaveCount(4);
        w[0].Should().Be(Expected0);
        w[1].Should().Be(Expected1);
        w[2].Should().Be(Expected2);
        w[3].Should().Be(Expected3);
    }

    /// <summary>
    /// AC-5: result satisfies Σw=1 and w_i ∈ [effMin, effMax] for the degenerate cases.
    /// </summary>
    [Fact]
    public void SolveMinVariance_InfeasibleBounds_ResultAlwaysFeasible()
    {
        // n=3, maxWeight=0.2 → effMax=1/3, minWeight=0 → effMin=0
        var cov = DiagonalCov(1m, 1m, 1m);
        var w = ActiveSetQpSolver<decimal>.SolveMinVariance(cov, minWeight: 0m, maxWeight: 0.2m);

        w.Should().HaveCount(3);
        w.Sum().Should().BeApproximately(1m, 1e-10m);
        var effMax = 1m / 3m;
        foreach (var wi in w)
        {
            wi.Should().BeLessThanOrEqualTo(effMax + 1e-10m);
            wi.Should().BeGreaterThanOrEqualTo(-1e-10m);
        }
    }

    // ─── M-1: Zero-dimension + null guards ──────────────────────────────

    /// <summary>
    /// AC-1 (double): a 0×0 covariance throws <see cref="ArgumentException"/> with
    /// ParamName "covariance" — NOT the pre-fix <see cref="InvalidOperationException"/>
    /// ("did not converge") produced when <c>1.0/0.0 = Infinity</c> falls through the loop.
    /// </summary>
    [Fact]
    public void SolveMinVariance_ZeroDimension_Double_ThrowsArgumentException()
    {
        var cov = new double[0, 0];

        var act = () => ActiveSetQpSolver<double>.SolveMinVariance(cov, minWeight: 0.0, maxWeight: 1.0);

        act.Should().Throw<ArgumentException>().WithParameterName("covariance");
    }

    /// <summary>
    /// AC-1 (decimal): a 0×0 covariance throws <see cref="ArgumentException"/> with
    /// ParamName "covariance" — NOT the pre-fix <see cref="DivideByZeroException"/>
    /// produced by <c>1m / 0m</c> in the equal-weight clamp.
    /// </summary>
    [Fact]
    public void SolveMinVariance_ZeroDimension_Decimal_ThrowsArgumentException()
    {
        var cov = new decimal[0, 0];

        var act = () => ActiveSetQpSolver<decimal>.SolveMinVariance(cov, minWeight: 0m, maxWeight: 1m);

        act.Should().Throw<ArgumentException>().WithParameterName("covariance");
    }

    /// <summary>
    /// AC-1 (double): a 0×0 covariance on the MeanVar entry point (riskAversion ≠ 0 so the
    /// LP delegation is bypassed) throws <see cref="ArgumentException"/> with ParamName
    /// "covariance".
    /// </summary>
    [Fact]
    public void SolveMeanVariance_ZeroDimension_Double_ThrowsArgumentException()
    {
        var cov = new double[0, 0];
        var means = System.Array.Empty<double>();

        var act = () => ActiveSetQpSolver<double>.SolveMeanVariance(
            cov, means, riskAversion: 1.0, minWeight: 0.0, maxWeight: 1.0);

        act.Should().Throw<ArgumentException>().WithParameterName("covariance");
    }

    /// <summary>
    /// AC-1 (decimal): a 0×0 covariance on the MeanVar entry point throws
    /// <see cref="ArgumentException"/> with ParamName "covariance" — NOT the pre-fix
    /// <see cref="DivideByZeroException"/>.
    /// </summary>
    [Fact]
    public void SolveMeanVariance_ZeroDimension_Decimal_ThrowsArgumentException()
    {
        var cov = new decimal[0, 0];
        var means = System.Array.Empty<decimal>();

        var act = () => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            cov, means, riskAversion: 1m, minWeight: 0m, maxWeight: 1m);

        act.Should().Throw<ArgumentException>().WithParameterName("covariance");
    }

    /// <summary>
    /// AC-2: a null covariance throws <see cref="ArgumentNullException"/> (ParamName
    /// "covariance") on <c>SolveMinVariance</c> at both T=double and T=decimal — NOT the
    /// pre-fix <see cref="NullReferenceException"/> from <c>covariance.GetLength(0)</c>.
    /// </summary>
    [Fact]
    public void SolveMinVariance_NullCovariance_ThrowsArgumentNullException()
    {
        var actDouble = () => ActiveSetQpSolver<double>.SolveMinVariance(null!, 0.0, 1.0);
        var actDecimal = () => ActiveSetQpSolver<decimal>.SolveMinVariance(null!, 0m, 1m);

        actDouble.Should().Throw<ArgumentNullException>().WithParameterName("covariance");
        actDecimal.Should().Throw<ArgumentNullException>().WithParameterName("covariance");
    }

    /// <summary>
    /// AC-2: a null covariance throws <see cref="ArgumentNullException"/> (ParamName
    /// "covariance") on <c>SolveMeanVariance</c> at both T=double and T=decimal.
    /// (The <c>means</c> parameter is a non-nullable <see cref="ReadOnlySpan{T}"/>, so
    /// there is no null-means path to guard — AC-2's expectedReturns sub-clause is N/A by type.)
    /// </summary>
    [Fact]
    public void SolveMeanVariance_NullCovariance_ThrowsArgumentNullException()
    {
        var meansDouble = new double[] { 0.1, 0.2 };
        var meansDecimal = new decimal[] { 0.1m, 0.2m };

        var actDouble = () => ActiveSetQpSolver<double>.SolveMeanVariance(
            null!, meansDouble, riskAversion: 1.0, minWeight: 0.0, maxWeight: 1.0);
        var actDecimal = () => ActiveSetQpSolver<decimal>.SolveMeanVariance(
            null!, meansDecimal, riskAversion: 1m, minWeight: 0m, maxWeight: 1m);

        actDouble.Should().Throw<ArgumentNullException>().WithParameterName("covariance");
        actDecimal.Should().Throw<ArgumentNullException>().WithParameterName("covariance");
    }

    // ─── M-1: SolveMaxReturnLP residual absorption never breaches bounds ──

    /// <summary>
    /// AC-3: randomized property test over the near-boundary <c>maxWeight ≈ 1/n</c> regime.
    /// Every <c>SolveMaxReturnLP</c> output (reached via <c>SolveMeanVariance</c> with
    /// riskAversion = 0) must satisfy <c>0 ≤ wᵢ ≤ maxWeight</c> EXACTLY (no ULP allowance
    /// above maxWeight) and <c>Σwᵢ == 1</c> within 1e-10. Pre-fix the residual is absorbed
    /// into the LARGEST weight (the LEAST headroom below maxWeight), breaching the upper
    /// bound by 1–2 ULP on the near-boundary lattice.
    /// </summary>
    [Fact]
    public void SolveMaxReturnLP_NearBoundary_AllWeightsWithinBoundsExactly()
    {
        var rng = new System.Random(20260707);
        const int Trials = 10_000;
        var breaches = 0;
        var firstBreach = string.Empty;

        for (var t = 0; t < Trials; t++)
        {
            var n = rng.Next(2, 13);

            // maxWeight in [1/n, 1.5/n): near the equal-weight frontier, n·maxWeight ∈ [1, 1.5).
            var inv = 1.0 / n;
            var maxWeight = inv * (1.0 + rng.NextDouble() * 0.5);

            var means = new double[n];
            for (var i = 0; i < n; i++)
            {
                means[i] = rng.NextDouble();
            }

            var cov = IdentityCov(n);

            var w = ActiveSetQpSolver<double>.SolveMeanVariance(
                cov, means, riskAversion: 0.0, minWeight: 0.0, maxWeight: maxWeight);

            // Full sum (never short-circuit) so the Σw check is on the whole vector, and
            // a separate scan records any bound breach without corrupting the sum.
            var sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                sum += w[i];
            }

            for (var i = 0; i < n; i++)
            {
                if (w[i] > maxWeight || w[i] < 0.0)
                {
                    breaches++;
                    if (firstBreach.Length == 0)
                    {
                        firstBreach =
                            $"trial {t}: n={n}, maxWeight={maxWeight:R}, w[{i}]={w[i]:R}, over={w[i] - maxWeight:R}";
                    }

                    break;
                }
            }

            System.Math.Abs(sum - 1.0).Should().BeLessThanOrEqualTo(
                1e-10, $"Σw must equal 1 (trial {t}, n={n})");
        }

        breaches.Should().Be(0, $"no weight may breach [0, maxWeight]; first breach: {firstBreach}");
    }

    /// <summary>
    /// Exact-frontier coverage that the randomized near-boundary generator above cannot
    /// reach: <c>minWeight == maxWeight == 1/n</c> and <c>maxWeight == 1/n, minWeight == 0</c>
    /// pin every weight against <c>1/n</c>, so the whole post-renormalisation residual must be
    /// absorbed where there is zero headroom. Pre-fix, <c>SolveMaxReturnLP</c> adds the residual
    /// to a single max-headroom index without capping it to that index's headroom, overshooting
    /// <c>maxWeight</c> by O(n)·ULP (7/18 n's at double, 2/18 at float). Every <c>wᵢ ≤ maxWeight</c>
    /// must hold EXACTLY (no ULP allowance) and <c>Σwᵢ == 1</c> within tolerance, at both
    /// <c>T = double</c> and <c>T = float</c>.
    /// </summary>
    [Fact]
    public void SolveMaxReturnLP_ExactFrontier_NoUpperBoundBreach_Double()
    {
        for (var n = 3; n <= 20; n++)
        {
            var inv = 1.0 / n;
            AssertExactFrontierWithinBounds(n, minWeight: inv, maxWeight: inv, sumTolerance: 1e-10);
            AssertExactFrontierWithinBounds(n, minWeight: 0.0, maxWeight: inv, sumTolerance: 1e-10);
        }
    }

    [Fact]
    public void SolveMaxReturnLP_ExactFrontier_NoUpperBoundBreach_Float()
    {
        for (var n = 3; n <= 20; n++)
        {
            var inv = 1.0f / n;
            AssertExactFrontierWithinBounds(n, minWeight: inv, maxWeight: inv, sumTolerance: 1e-4f);
            AssertExactFrontierWithinBounds(n, minWeight: 0.0f, maxWeight: inv, sumTolerance: 1e-4f);
        }
    }

    private static void AssertExactFrontierWithinBounds<T>(int n, T minWeight, T maxWeight, T sumTolerance)
        where T : System.Numerics.IFloatingPoint<T>
    {
        var cov = new T[n, n];
        var means = new T[n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = T.One;
            means[i] = T.CreateChecked(i + 1); // distinct ascending returns → deterministic order
        }

        var w = ActiveSetQpSolver<T>.SolveMeanVariance(
            cov, means, riskAversion: T.Zero, minWeight, maxWeight);

        var sum = T.Zero;
        for (var i = 0; i < n; i++)
        {
            (w[i] <= maxWeight).Should().BeTrue(
                $"w[{i}]={w[i]} must not exceed maxWeight={maxWeight} (no ULP allowance; " +
                $"n={n}, T={typeof(T).Name}, over={w[i] - maxWeight})");
            (w[i] >= minWeight - sumTolerance).Should().BeTrue(
                $"w[{i}]={w[i]} must be at least minWeight={minWeight} (n={n}, T={typeof(T).Name})");
            sum += w[i];
        }

        (T.Abs(sum - T.One) <= sumTolerance).Should().BeTrue(
            $"Σw={sum} must equal 1 within {sumTolerance} (n={n}, T={typeof(T).Name})");
    }

    // ─── Cross-type parity ─────────────────────────────────────────────

    [Fact]
    public void SolveMinVariance_GenericDoubleMatchesDecimal()
    {
        // Same 3-asset diagonal problem at T=double and T=decimal.
        var covDec = DiagonalCov(1m, 4m, 9m);
        var covDbl = new double[3, 3]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 4.0, 0.0 },
            { 0.0, 0.0, 9.0 },
        };

        var wDec = ActiveSetQpSolver<decimal>.SolveMinVariance(covDec, 0m, 1m);
        var wDbl = ActiveSetQpSolver<double>.SolveMinVariance(covDbl, 0.0, 1.0);

        wDec.Should().HaveCount(3);
        wDbl.Should().HaveCount(3);
        for (var i = 0; i < 3; i++)
        {
            ((double)wDec[i]).Should().BeApproximately(wDbl[i], 1e-6);
        }
    }

    // ─── Test helpers ──────────────────────────────────────────────────

    /// <summary>Creates an N×N identity covariance matrix (LP path ignores its contents).</summary>
    private static double[,] IdentityCov(int n)
    {
        var cov = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            cov[i, i] = 1.0;
        }

        return cov;
    }

    // The pre-rewrite core, its verification patch, its release normalisation and every per-site rescue path of it
    // were deleted when the primal-feasible active set became the only core. A rescue path that returns would bring
    // back at least one of these names; the scan covers every source file of the library.
    private static readonly string[] s_removedNames =
    [
        "VerifiedFreeWeights", "NormalizeReleaseGradient", "TryKktResidual", "KktRows", "TryNullSpaceFreeWeights",
        "TryReleaseMinVar", "TryReleaseMeanVar", "SolveMinVarianceLegacyCore", "SolveMeanVarianceLegacyCore", "legacyCore",
        "RepairFixedBudget", "IsFixedSumAtBudget", "SolveScaledForQp", "SolveCholeskyForQp", "FindWorstViolation",
        "BuildFullWeights", "ReleaseTolerance", "FixedCrossTerm", "MagnitudeOrderedProduct", "RangeSafeSum",
        "RangeSafeRatio", "FreeWeight", "IsDegenerateInverseBudget", "BudgetRightHandSideScale", "CollectFreeIndices",
        "RequireFiniteCrossTerm",
    ];

    [Fact]
    public void Source_ContainsNoneOfTheRemovedOldCoreNamesAndNoPerSiteRescuePath()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Boutquin.Numerics.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.True(directory is not null, "the repository root (Boutquin.Numerics.slnx) was not found above " + AppContext.BaseDirectory);
        var sources = Directory.EnumerateFiles(Path.Combine(directory!.FullName, "src", "Boutquin.Numerics"), "*.cs", SearchOption.AllDirectories)
            .Where(path => !path.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar, StringComparison.Ordinal)
                && !path.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            .ToList();
        Assert.Contains(sources, path => path.EndsWith("ActiveSetQpSolver.cs", StringComparison.Ordinal));

        var hits = new List<string>();
        foreach (var path in sources)
        {
            var lines = File.ReadAllLines(path);
            for (var i = 0; i < lines.Length; i++)
            {
                foreach (var name in s_removedNames.Where(name => lines[i].Contains(name, StringComparison.Ordinal)))
                {
                    hits.Add(Path.GetFileName(path) + ":" + (i + 1) + " " + name);
                }
            }
        }

        Assert.True(hits.Count == 0, "removed old-core names found in src: " + string.Join(", ", hits));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(20)]
    [InlineData(100)]
    public void WorkingSetChangeBudgets_AreEightNPlusEightRoundedAndTwelveNPlusTwelveExact(int n)
    {
        ActiveSetQpSolver<double>.IterationBudget(n).Should().Be((8 * n) + 8);
        ActiveSetQpSolver<double>.ExactRouteBudget(n).Should().Be((12 * n) + 12);
    }

    /// <summary>Creates a 3×3 diagonal covariance matrix from three variance values.</summary>
    private static decimal[,] DiagonalCov(decimal v0, decimal v1, decimal v2)
        => new decimal[3, 3]
        {
            { v0, 0m, 0m },
            { 0m, v1, 0m },
            { 0m, 0m, v2 },
        };

    /// <summary>
    /// Returns a symmetric positive-definite 4×4 covariance matrix
    /// constructed as A = D + 0.2·11ᵀ where D = diag(1,2,3,4).
    /// </summary>
    private static decimal[,] SparseSpdCov4()
    {
        var offDiag = 0.2m;
        return new decimal[4, 4]
        {
            { 1m + offDiag, offDiag,       offDiag,       offDiag       },
            { offDiag,       2m + offDiag,  offDiag,       offDiag       },
            { offDiag,       offDiag,       3m + offDiag,  offDiag       },
            { offDiag,       offDiag,       offDiag,       4m + offDiag  },
        };
    }
}
