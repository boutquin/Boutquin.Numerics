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

using System.Numerics;
using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class Arena20260928QpTests
{
    [Theory]
    [InlineData(false, 12)]
    [InlineData(true, 12)]
    [InlineData(false, 19)]
    [InlineData(true, 19)]
    public void FloatEqualFrontier_RepairsBudgetWithinBounds(bool meanVariance, int count)
    {
        var covariance = new float[count, count];
        for (var i = 0; i < count; i++)
        {
            covariance[i, i] = i + 1;
        }

        var upper = 1f / count;
        float[]? weights = null;
        Action solve = () => weights = meanVariance
            ? ActiveSetQpSolver<float>.SolveMeanVariance(covariance, new float[count], 1, 0, upper)
            : ActiveSetQpSolver<float>.SolveMinVariance(covariance, 0, upper);
        solve.Should().NotThrow("roundoff in an all-fixed frontier must not cycle the active set");
        weights.Should().OnlyContain(w => w >= 0 && w <= upper);
        weights!.Sum(w => (double)w).Should().BeApproximately(1, 1e-8);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void CovarianceUnits_DoNotChangeRepresentableOptimum(bool meanVariance, bool useDecimal)
    {
        if (useDecimal)
        {
            CheckScaled<decimal>(meanVariance);
        }
        else
        {
            CheckScaled<double>(meanVariance);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DecimalCovarianceUnits_PreserveNonTerminatingInverseRatios(bool meanVariance)
    {
        decimal[,] covariance = { { 4e22m, 0 }, { 0, 7e22m } };
        var weights = meanVariance
            ? ActiveSetQpSolver<decimal>.SolveMeanVariance(covariance, [0, 0], 1, 0, 1)
            : ActiveSetQpSolver<decimal>.SolveMinVariance(covariance, 0, 1);

        weights[0].Should().BeApproximately(7m / 11, 1e-25m);
        weights[1].Should().BeApproximately(4m / 11, 1e-25m);
    }

    private static void CheckScaled<T>(bool meanVariance) where T : IFloatingPoint<T>
    {
        var scale = T.CreateChecked(1e22m);
        T[,] covariance = { { scale, T.Zero }, { T.Zero, scale + scale } };
        T[]? weights = null;
        Action solve = () => weights = meanVariance
            ? ActiveSetQpSolver<T>.SolveMeanVariance(covariance, [T.Zero, T.Zero], T.One, T.Zero, T.One)
            : ActiveSetQpSolver<T>.SolveMinVariance(covariance, T.Zero, T.One);

        solve.Should().NotThrow("positive scalar covariance units preserve the constrained quadratic optimum");
        double.CreateChecked(weights![0]).Should().BeApproximately(2.0 / 3, 1e-14);
        double.CreateChecked(weights[1]).Should().BeApproximately(1.0 / 3, 1e-14);
    }
}
