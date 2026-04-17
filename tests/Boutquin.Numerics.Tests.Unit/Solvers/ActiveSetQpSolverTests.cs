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
    // ─── SolveMinVariance tests ────────────────────────────────────────

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
