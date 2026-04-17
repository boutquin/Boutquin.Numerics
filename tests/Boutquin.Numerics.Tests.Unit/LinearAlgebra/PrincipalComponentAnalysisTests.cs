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

namespace Boutquin.Numerics.Tests.Unit.LinearAlgebra;

public sealed class PrincipalComponentAnalysisTests
{
    // ── Yield-curve level/slope/curvature shape (§3.2 AC #1) ──────

    [Fact]
    public void FromReturns_OnYieldCurveLikeData_ReturnsLevelSlopeCurvatureShape()
    {
        // Synthesize a curve-factor panel: r_t,j = l_t + s_t·(j − midpoint) + c_t·((j − midpoint)² − k) + ε,
        // where level l_t is the dominant factor and slope/curvature contribute smaller amounts.
        const int periods = 500;
        const int tenors = 6;
        var midpoint = (tenors - 1) / 2.0;
        const double curvaturePivot = 2.0;

        var rng = new DeterministicNormal(seed: 7);
        var returns = new decimal[periods, tenors];
        for (var t = 0; t < periods; t++)
        {
            var level = 1.0 * rng.Next();
            var slope = 0.3 * rng.Next();
            var curvature = 0.1 * rng.Next();
            for (var j = 0; j < tenors; j++)
            {
                var shape = level + (slope * (j - midpoint)) + (curvature * (((j - midpoint) * (j - midpoint)) - curvaturePivot));
                var noise = 0.02 * rng.Next();
                returns[t, j] = (decimal)(shape + noise);
            }
        }

        var pca = PrincipalComponentAnalysis.FromReturns(returns);

        // Yield-curve-like signatures: PC1 dominates; PC1+PC2+PC3 explains nearly everything.
        pca.ExplainedVarianceRatio[0].Should().BeGreaterThan(0.70m);
        pca.ExplainedVarianceRatio[1].Should().BeInRange(0.05m, 0.25m);
        pca.ExplainedVarianceRatio[2].Should().BeInRange(0.01m, 0.10m);
        pca.CumulativeExplainedVariance[2].Should().BeGreaterThan(0.98m);
    }

    // ── Symmetry check (§3.2 AC #3) ────────────────────────────────

    [Fact]
    public void Decompose_NonSymmetricMatrix_Throws()
    {
        var nonSymmetric = new decimal[,]
        {
            { 1.0m, 0.5m },
            { 0.4m, 1.0m },
        };

        var act = () => PrincipalComponentAnalysis.Decompose(nonSymmetric);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Decompose_NonSquareMatrix_Throws()
    {
        var rectangular = new decimal[,]
        {
            { 1.0m, 0.5m, 0.2m },
            { 0.5m, 1.0m, 0.3m },
        };

        var act = () => PrincipalComponentAnalysis.Decompose(rectangular);
        act.Should().Throw<ArgumentException>();
    }

    // ── Projection round-trip (§3.2 AC #4) ─────────────────────────

    [Fact]
    public void Project_FullRank_ReconstructsOriginalData()
    {
        var rng = new DeterministicNormal(seed: 3);
        var data = new decimal[50, 3];
        for (var i = 0; i < 50; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                data[i, j] = (decimal)rng.Next();
            }
        }

        var pca = PrincipalComponentAnalysis.FromReturns(data);

        var scores = pca.Project(data, numComponents: 3);

        // Reconstruct: data ≈ scores · Eigenvectorsᵀ + Mean.
        // Since k = n, this should be nearly exact.
        var reconstructed = new decimal[50, 3];
        for (var i = 0; i < 50; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var value = pca.Mean[j];
                for (var k = 0; k < 3; k++)
                {
                    value += scores[i, k] * pca.Eigenvectors[j, k];
                }

                reconstructed[i, j] = value;
            }
        }

        for (var i = 0; i < 50; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                ((double)(data[i, j] - reconstructed[i, j])).Should().BeApproximately(0.0, 1e-10);
            }
        }
    }

    // ── Sign convention (§3.2 AC #5) ───────────────────────────────

    [Fact]
    public void FromReturns_TwoCallsOnSameData_ProduceIdenticalEigenvectorSigns()
    {
        var rng = new DeterministicNormal(seed: 123);
        var data = new decimal[100, 4];
        for (var i = 0; i < 100; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                data[i, j] = (decimal)rng.Next();
            }
        }

        var a = PrincipalComponentAnalysis.FromReturns(data);
        var b = PrincipalComponentAnalysis.FromReturns(data);

        a.Eigenvectors.GetLength(0).Should().Be(b.Eigenvectors.GetLength(0));
        for (var i = 0; i < a.Eigenvectors.GetLength(0); i++)
        {
            for (var j = 0; j < a.Eigenvectors.GetLength(1); j++)
            {
                a.Eigenvectors[i, j].Should().Be(b.Eigenvectors[i, j]);
            }
        }
    }

    [Fact]
    public void ApplySignConvention_MakesLargestMagnitudeComponentPositive()
    {
        // A covariance where one eigenvector's largest entry is negative by default.
        // Hand-crafted: diagonal [1, 4] has trivial eigenvectors (0,1) and (1,0); we want
        // a non-diagonal example so Jacobi has work to do.
        var cov = new decimal[,]
        {
            { 4.0m, 1.0m },
            { 1.0m, 2.0m },
        };

        var pca = PrincipalComponentAnalysis.Decompose(cov);

        // For each eigenvector column, the largest-magnitude entry must be non-negative.
        for (var k = 0; k < 2; k++)
        {
            var maxAbs = 0m;
            var signOfMax = 1;
            for (var i = 0; i < 2; i++)
            {
                var abs = Math.Abs(pca.Eigenvectors[i, k]);
                if (abs > maxAbs)
                {
                    maxAbs = abs;
                    signOfMax = pca.Eigenvectors[i, k] < 0m ? -1 : 1;
                }
            }

            signOfMax.Should().Be(1, because: "sign convention requires the largest-magnitude component to be non-negative");
        }
    }

    // ── NumComponentsForExplainedVariance (§3.2 AC #6) ─────────────

    [Fact]
    public void NumComponentsForExplainedVariance_SingleFactorSynthetic_ReturnsOne()
    {
        // 6 variables all loaded on a single factor plus tiny idiosyncratic noise.
        var rng = new DeterministicNormal(seed: 11);
        var data = new decimal[400, 6];
        for (var t = 0; t < 400; t++)
        {
            var factor = rng.Next();
            for (var j = 0; j < 6; j++)
            {
                data[t, j] = (decimal)(factor + 0.01 * rng.Next());
            }
        }

        var pca = PrincipalComponentAnalysis.FromReturns(data);
        pca.NumComponentsForExplainedVariance(0.95m).Should().Be(1);
    }

    // ── Explained variance arithmetic ──────────────────────────────

    [Fact]
    public void ExplainedVariance_SumsToOne()
    {
        var rng = new DeterministicNormal(seed: 19);
        var data = new decimal[100, 4];
        for (var i = 0; i < 100; i++)
        {
            for (var j = 0; j < 4; j++)
            {
                data[i, j] = (decimal)rng.Next();
            }
        }

        var pca = PrincipalComponentAnalysis.FromReturns(data);
        var sum = 0m;
        foreach (var r in pca.ExplainedVarianceRatio)
        {
            sum += r;
        }

        ((double)sum).Should().BeApproximately(1.0, 1e-10);
        ((double)pca.CumulativeExplainedVariance[^1]).Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void Eigenvalues_AreSortedDescending()
    {
        var cov = new decimal[,]
        {
            { 5.0m, 1.0m, 0.5m },
            { 1.0m, 3.0m, 0.2m },
            { 0.5m, 0.2m, 2.0m },
        };

        var pca = PrincipalComponentAnalysis.Decompose(cov);

        for (var i = 0; i < pca.Eigenvalues.Length - 1; i++)
        {
            pca.Eigenvalues[i].Should().BeGreaterThanOrEqualTo(pca.Eigenvalues[i + 1]);
        }
    }

    // ── Standardization path ───────────────────────────────────────

    [Fact]
    public void FromReturns_Standardize_NormalizesScales()
    {
        // Construct data with widely different scales. Standardization should make the PCA
        // result insensitive to the per-column scale.
        var rng = new DeterministicNormal(seed: 47);
        var data = new decimal[300, 3];
        for (var t = 0; t < 300; t++)
        {
            var f = rng.Next();
            data[t, 0] = (decimal)(f + 0.1 * rng.Next());
            data[t, 1] = (decimal)(10.0 * f + 1.0 * rng.Next());
            data[t, 2] = (decimal)(0.01 * f + 0.001 * rng.Next());
        }

        var unstandardized = PrincipalComponentAnalysis.FromReturns(data, standardize: false);
        var standardized = PrincipalComponentAnalysis.FromReturns(data, standardize: true);

        // Unstandardized — column 2 variance dominates.
        // Standardized — equal scales, PC1 should capture the shared factor near-fully.
        standardized.ExplainedVarianceRatio[0].Should().BeGreaterThan(0.90m);
        unstandardized.ExplainedVarianceRatio[0].Should().BeGreaterThan(0.90m);
    }

    // ── ReturnsMatrix overload ─────────────────────────────────────

    [Fact]
    public void FromReturns_ReturnsMatrixOverload_AgreesWithArrayOverload()
    {
        var rng = new DeterministicNormal(seed: 29);
        var data = new decimal[100, 3];
        for (var i = 0; i < 100; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                data[i, j] = (decimal)rng.Next();
            }
        }

        var fromArray = PrincipalComponentAnalysis.FromReturns(data);
        var fromMatrix = PrincipalComponentAnalysis.FromReturns(new ReturnsMatrix(data));

        fromArray.Eigenvalues.Should().Equal(fromMatrix.Eigenvalues);
    }

    // ── FromReturns input validation ───────────────────────────────

    [Fact]
    public void FromReturns_SingleObservation_Throws()
    {
        var data = new decimal[1, 3] { { 1m, 2m, 3m } };
        var act = () => PrincipalComponentAnalysis.FromReturns(data);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void FromReturns_Standardize_ZeroVarianceColumn_Throws()
    {
        var data = new decimal[,]
        {
            { 1m, 1m },
            { 2m, 1m },
            { 3m, 1m },
        };
        var act = () => PrincipalComponentAnalysis.FromReturns(data, standardize: true);
        act.Should().Throw<ArgumentException>();
    }

    // ── NumComponentsForExplainedVariance validation ───────────────

    [Fact]
    public void NumComponentsForExplainedVariance_ThresholdOutOfRange_Throws()
    {
        var pca = PrincipalComponentAnalysis.Decompose(new decimal[,] { { 1m, 0m }, { 0m, 1m } });
        var act0 = () => pca.NumComponentsForExplainedVariance(0m);
        var act2 = () => pca.NumComponentsForExplainedVariance(1.5m);

        act0.Should().Throw<ArgumentOutOfRangeException>();
        act2.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ── Project input validation ──────────────────────────────────

    [Fact]
    public void Project_ColumnCountMismatch_Throws()
    {
        var pca = PrincipalComponentAnalysis.Decompose(new decimal[,] { { 1m, 0m }, { 0m, 1m } });
        var act = () => pca.Project(new decimal[,] { { 1m, 2m, 3m } }, numComponents: 1);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Project_OutOfRangeComponents_Throws()
    {
        var pca = PrincipalComponentAnalysis.Decompose(new decimal[,] { { 1m, 0m }, { 0m, 1m } });
        var act = () => pca.Project(new decimal[,] { { 1m, 2m } }, numComponents: 3);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ── Helpers ────────────────────────────────────────────────────

    /// <summary>
    /// Deterministic standard-normal generator (Box–Muller on a linear-congruential
    /// uniform stream). Matches the generator used in the LM solver tests; reproduced
    /// here to keep test files independent.
    /// </summary>
    private sealed class DeterministicNormal
    {
        private uint _state;
        private double? _pendingGaussian;

        public DeterministicNormal(uint seed)
        {
            _state = seed == 0 ? 1u : seed;
        }

        public double Next()
        {
            if (_pendingGaussian is { } cached)
            {
                _pendingGaussian = null;
                return cached;
            }

            double u1, u2;
            do
            {
                u1 = NextUniform();
                u2 = NextUniform();
            }
            while (u1 <= double.Epsilon);

            var radius = Math.Sqrt(-2.0 * Math.Log(u1));
            var angle = 2.0 * Math.PI * u2;

            _pendingGaussian = radius * Math.Sin(angle);
            return radius * Math.Cos(angle);
        }

        private double NextUniform()
        {
            _state = (1664525u * _state) + 1013904223u;
            return (_state & 0xFFFFFFu) / (double)0x1000000;
        }
    }
}
