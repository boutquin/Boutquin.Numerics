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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Pins two documented refusals of <see cref="ActiveSetQpSolver{T}"/> through both public objectives at both
/// guaranteed types, each with its exact exception type and message: a single asset whose variance is not positive
/// (the order-one branch of both solves, which returns before the general positive-definiteness decision runs), and the
/// translation of a <c>decimal</c> <see cref="OverflowException"/> into <see cref="InvalidOperationException"/>.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpEntryPointRefusalParityTests
{
    private const string NotPositiveDefinite = "Covariance matrix must be positive definite.";

    private const string ExceededRange = "QP arithmetic exceeded the numeric type's range; rescale the problem.";

    // The largest decimal is about 7.9228e28: with two assets, a bound of 7e28 makes n * bound (1.4e29) unrepresentable.
    private const decimal LargeBound = 7e28m;

    /// <summary>
    /// A single asset whose variance is zero or negative is not positive definite: <c>SolveMinVariance</c> and
    /// <c>SolveMeanVariance</c> (λ &gt; 0) refuse with <see cref="InvalidOperationException"/> instead of returning the
    /// weight vector <c>[1]</c>, at <c>double</c> and at <c>decimal</c>.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(-0.25)]
    public void SingleAssetWithNonPositiveVariance_IsRefusedByBothObjectivesAtBothTypes(double variance)
    {
        AssertSingleAssetRefused<double>(variance);
        AssertSingleAssetRefused<decimal>(variance);
    }

    /// <summary>
    /// The counterpart of the refusal: a single asset with a positive variance, however small, is not refused and
    /// holds the whole budget, so the refusal does not reach past the non-positive variances.
    /// </summary>
    [Fact]
    public void SingleAssetWithTinyPositiveVariance_ReturnsTheWholeBudgetFromBothObjectivesAtBothTypes()
    {
        AssertSingleAssetSolved(1e-20, 1e-20m);
        AssertSingleAssetSolved(4.0, 4m);
    }

    /// <summary>
    /// At <c>λ = 0</c> the mean-variance solve is the max-return LP, whose budget arithmetic overflows <c>decimal</c> for
    /// bounds this large; the documented contract reports that as <see cref="InvalidOperationException"/> (never as the
    /// bare <see cref="OverflowException"/>), with the range message and the overflow as the inner exception. Each case
    /// overflows at a different product of the LP's feasibility checks.
    /// </summary>
    [Theory]
    [InlineData(-7e28, 7e28, 0.1, 0.2)]
    [InlineData(-7e28, 7e28, 0.0, 1.0)]
    [InlineData(0.0, 7e28, 0.2, 0.1)]
    public void SolveMeanVariance_DecimalLpBudgetOverflow_IsReportedAsInvalidOperation(double lower, double upper, double firstMean, double secondMean)
    {
        var identity = new decimal[,] { { 1m, 0m }, { 0m, 1m } };
        var means = new[] { (decimal)firstMean, (decimal)secondMean };

        var error = Assert.Throws<InvalidOperationException>(
            () => ActiveSetQpSolver<decimal>.SolveMeanVariance(identity, means, 0m, (decimal)lower, (decimal)upper));

        Assert.Equal(ExceededRange, error.Message);
        Assert.IsType<OverflowException>(error.InnerException);
    }

    /// <summary>
    /// The same translation through the public min-variance entry point: a seeded sweep over <c>decimal</c> covariances
    /// and boxes at the edge of the type's range, which no input is known to push to an <see cref="OverflowException"/>
    /// (every solve either succeeds or is refused). Whatever such an input does, the only exception that may leave
    /// <c>SolveMinVariance</c> is <see cref="InvalidOperationException"/> (with the range message when it wraps an
    /// overflow), never the bare <see cref="OverflowException"/>.
    /// </summary>
    [Fact]
    public void SolveMinVariance_DecimalCovariancesAtTheEdgeOfTheRange_NeverLeaksAnOverflowException()
    {
        var rng = new System.Random(20261101);
        var solved = 0;
        var refused = 0;
        for (var trial = 0; trial < 150; trial++)
        {
            var n = 2 + rng.Next(0, 4);
            var covariance = new decimal[n, n];
            for (var i = 0; i < n; i++)
            {
                covariance[i, i] = rng.Next(0, 2) == 0
                    ? (decimal)Math.Pow(10.0, rng.NextDouble() * 28.8)
                    : 1m / (decimal)Math.Pow(10.0, rng.Next(0, 28));
            }

            var upper = rng.Next(0, 2) == 0 ? LargeBound : 1m;
            var lower = rng.Next(0, 2) == 0 ? -upper : 0m;
            try
            {
                var weights = ActiveSetQpSolver<decimal>.SolveMinVariance(covariance, lower, upper);
                Assert.True(Math.Abs(weights.Sum() - 1m) <= 1e-20m, weights.Sum().ToString(CultureInfo.InvariantCulture));
                solved++;
            }
            catch (InvalidOperationException error)
            {
                if (error.InnerException is OverflowException)
                {
                    Assert.Equal(ExceededRange, error.Message);
                }

                refused++;
            }
        }

        Assert.Equal(150, solved + refused);
        Assert.True(solved > 0, "no sweep input was solved");
    }

    private static void AssertSingleAssetRefused<T>(double variance)
        where T : IFloatingPoint<T>
    {
        var covariance = new T[1, 1] { { T.CreateChecked(variance) } };
        var label = typeof(T).Name + " variance " + variance.ToString(CultureInfo.InvariantCulture);

        var minVariance = Assert.Throws<InvalidOperationException>(
            () => ActiveSetQpSolver<T>.SolveMinVariance(covariance, T.Zero, T.One));
        Assert.True(NotPositiveDefinite == minVariance.Message, "SolveMinVariance " + label + ": " + minVariance.Message);

        var meanVariance = Assert.Throws<InvalidOperationException>(
            () => ActiveSetQpSolver<T>.SolveMeanVariance(covariance, [T.CreateChecked(0.1)], T.One, T.Zero, T.One));
        Assert.True(NotPositiveDefinite == meanVariance.Message, "SolveMeanVariance " + label + ": " + meanVariance.Message);
    }

    private static void AssertSingleAssetSolved(double doubleVariance, decimal decimalVariance)
    {
        AssertSolved(new double[1, 1] { { doubleVariance } }, 0.1);
        AssertSolved(new decimal[1, 1] { { decimalVariance } }, 0.1m);
    }

    private static void AssertSolved<T>(T[,] covariance, T mean)
        where T : IFloatingPoint<T>
    {
        Assert.Equal([T.One], ActiveSetQpSolver<T>.SolveMinVariance(covariance, T.Zero, T.One));
        Assert.Equal([T.One], ActiveSetQpSolver<T>.SolveMeanVariance(covariance, [mean], T.One, T.Zero, T.One));
    }
}
