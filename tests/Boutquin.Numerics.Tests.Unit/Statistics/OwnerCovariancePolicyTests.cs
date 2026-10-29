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

using Boutquin.Numerics.Random;
using Boutquin.Numerics.Statistics;
using FluentAssertions;
namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class OwnerCovariancePolicyTests
{
    [Fact]
    public void Oas_IsotropicDecimal_DispersionBoundaryReturnsIdentity()
    {
        decimal[,] rows = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
        decimal[,]? result = null;
        var action = () => result = new OracleApproximatingShrinkageEstimator<decimal>().Estimate(rows);
        action.Should().NotThrow("zero dispersion must bypass division by zero");
        result![0, 0].Should().Be(.5m);
        result[1, 1].Should().Be(.5m);
        result[0, 1].Should().Be(0);
    }

    [Fact]
    public void Nercome_ProductionAverager_RejectsDuplicateSubsetsAndUsesEachBasis()
    {
        // Scripted Fisher-Yates: identity, duplicate identity, then swap0/3.
        // Accepted first subsets {0,1} and {1,3}; exact 2x2 projector mean
        // [[13/5,3/10],[3/10,53/20]], squared averaged residual=1/40.
        var random = new ScriptedRandom();
        var method = typeof(NercomeCovarianceEstimator<double>).GetMethod("AverageSplits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var estimator = new NercomeCovarianceEstimator<double>(.5, 2, 0);
        (double[,] Estimate, double Score) result = default;
        var action = () => result = ((double[,] Estimate, double Score))method.Invoke(estimator,
            [new double[,] { { 1, 0 }, { 0, 2 }, { 2, 1 }, { -1, 3 } }, 2, random])!;
        action.Should().NotThrow();
        random.Calls.Should().Be(9);
        result.Estimate.Should().NotBeNull();
        result.Estimate![0, 0].Should().BeApproximately(13.0 / 5, 1e-12);
        result.Estimate![0, 1].Should().BeApproximately(3.0 / 10, 1e-12);
        result.Estimate![1, 1].Should().BeApproximately(53.0 / 20, 1e-12);
        (result.Score * result.Score).Should().BeApproximately(1.0 / 40, 1e-12);
    }

    private sealed class ScriptedRandom : IRandomSource
    {
        private readonly int[] _choices = [3, 2, 1, 3, 2, 1, 0, 2, 1];
        public int Calls { get; private set; }
        public ulong NextULong() => throw new NotSupportedException();
        public int NextInt(int upperExclusive)
        {
            upperExclusive.Should().Be(4 - Calls % 3);
            return _choices[Calls++];
        }
    }
}
