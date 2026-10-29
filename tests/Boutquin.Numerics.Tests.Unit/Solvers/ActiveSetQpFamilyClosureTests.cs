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

using System.Globalization;
using System.Numerics;

using Boutquin.Numerics.Solvers;

using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>Large accepted solves across both public objectives and guaranteed numeric types.</summary>
[Collection(ActiveSetQpPerformanceCollection.Name)]
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpFamilyClosureTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(100, false)]
    [InlineData(150, false)]
    [InlineData(200, false)]
    [InlineData(100, true)]
    [InlineData(150, true)]
    [InlineData(200, true)]
    public void AcceptedLongShortProblems_HaveBoundedAllocationAndIndependentOptimality(int n, bool meanVariance)
    {
        Check<double>(n, meanVariance, ActiveSetQpLargeProblemTests.DoubleStationarity);
        Check<decimal>(n, meanVariance, ActiveSetQpLargeProblemTests.DecimalStationarity);
    }

    private void Check<T>(int n, bool meanVariance, double tolerance)
        where T : IFloatingPoint<T>
    {
        var covariance = ActiveSetQpLargeProblemTests.FactorModel<T>(n, seed: 20261021, digits: 14);
        var means = new T[n];
        var rng = new System.Random(20261022);
        for (var i = 0; i < n; i++)
        {
            means[i] = T.CreateChecked(0.02 + (rng.NextDouble() * 0.08));
        }

        var lower = -T.One / T.CreateChecked(2);
        var upper = -lower;
        var risk = T.CreateChecked(3);
        var small = ActiveSetQpLargeProblemTests.FactorModel<T>(8, seed: 2, digits: 14);
        _ = ActiveSetQpSolver<T>.SolveMinVariance(small, lower, upper);
        var trace = new ActiveSetQpSolver<T>.RouteTrace();
        var before = GC.GetAllocatedBytesForCurrentThread();
        var weights = meanVariance
            ? ActiveSetQpSolver<T>.SolveMeanVarianceUsingCore(covariance, means, risk, lower, upper, trace: trace)
            : ActiveSetQpSolver<T>.SolveMinVarianceUsingCore(covariance, lower, upper, trace: trace);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // A sequence of n growing working sets does cubic scalar work with prefix reuse. Extend the existing
        // order-100 allocation bar cubically; do not substitute a wall-clock ratio for this work bound.
        var ceiling = (384L << 20) * n * n * n / 1_000_000;
        var label = string.Create(CultureInfo.InvariantCulture,
            $"{typeof(T).Name} n={n} meanVariance={meanVariance}: {allocated / 1048576.0:F1} MiB / {ceiling / 1048576.0:F1} MiB; rounded={trace.RoundedChanges}, exact={trace.ExactChanges}, guard exact={trace.ExactConditioningTests}");
        output.WriteLine(label);
        Assert.Equal(ActiveSetQpSolver<T>.ExactRouteReason.None, trace.ExactRouteReason);
        Assert.Equal(0, trace.ExactChanges);
        Assert.Equal(0, trace.ExactConditioningTests);
        Assert.InRange(trace.RoundedChanges, n / 2, (8 * n) + 8);
        Assert.True(allocated <= ceiling, label);

        var budget = weights.Aggregate(T.Zero, (sum, value) => sum + value);
        Assert.True(T.Abs(budget - T.One) <= T.CreateChecked(tolerance * n));
        Assert.All(weights, w => Assert.InRange(w, lower, upper));
        ActiveSetQpLargeProblemTests.AssertOptimal(covariance, weights, lower, upper, meanVariance ? means : null,
            meanVariance ? risk : T.One, tolerance);
    }
}
