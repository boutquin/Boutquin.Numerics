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
/// The reduced QP solves Σ_FF·x = r for the budget direction (<c>a</c>, <c>z</c>), the mean term
/// (<c>b</c>) and the fixed-variable cross term (<c>zc</c>), but the weights need only linear
/// combinations and ratios of those solutions. Scaling the means and the risk aversion by the same
/// exact power of the radix leaves the optimum unchanged, so the solver must return that optimum for
/// every such scale at which it is representable, even when an inverse-solve component itself
/// leaves the type's range. Expected values are the exact rational KKT optimum (active-set
/// enumeration in <c>fractions.Fraction</c> on the exact inputs), never the implementation's output.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpInverseSolveRangeTests
{
    // Witness (1): the all-free mean solve b = Σ⁻¹μ′ is 0.81·MaxValue at scale 2^0 and beyond the
    // range at every larger scale; the optimum pins assets 1 and 2 at the lower bound.
    private static readonly double[,] s_doubleCovariance =
    {
        { 3.2439554628389534e-150, -3.2439554628260043e-150, 0.0 },
        { -3.2439554628260043e-150, 3.2439554628389534e-150, 0.0 },
        { 0.0, 0.0, 7.06931522728001e-35 },
    };

    private static readonly double[] s_doubleMeans =
        [1.8771217870003988e147, 1.8771217857034463e147, -1.8771217854622924e147];

    private const double DoubleRiskAversion = 3.5591885256394595e-07;
    private const double DoubleLower = -5.097088906927171e140;
    private const double DoubleUpper = 4.8952270514171424e147;

    // Exact optimum, rounded once to double: (1 − 2·lb, lb, lb).
    private static readonly double[] s_doubleOptimum =
        [1.0194177813854342e141, -5.097088906927171e140, -5.097088906927171e140];

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void QpMeanVariance_DoubleMeanSolveBeyondRange_ReturnsTheScaleInvariantOptimum(int exponent)
    {
        var scale = Math.ScaleB(1.0, exponent);
        var means = s_doubleMeans.Select(m => m * scale).ToArray();

        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            s_doubleCovariance, means, DoubleRiskAversion * scale, DoubleLower, DoubleUpper));

        error.Should().BeNull("the optimum is unchanged by an exact power-of-two rescale of μ and λ and is representable");
        for (var i = 0; i < 3; i++)
        {
            weights[i].Should().BeApproximately(
                s_doubleOptimum[i], Math.Abs(s_doubleOptimum[0]) * 1e-12, "weight {0} must be the exact optimum", i);
        }

        // Power-of-two scaling is exact in binary arithmetic, so every scale reproduces the bits of
        // the in-range scale 2^0, whose mean solve never leaves the range.
        var reference = ActiveSetQpSolver<double>.SolveMeanVariance(
            s_doubleCovariance, s_doubleMeans, DoubleRiskAversion, DoubleLower, DoubleUpper);
        weights.Should().Equal(reference);
    }

    // Witness (2): T = decimal, in domain; the exact max|b_i| is 3.16·decimal.MaxValue at scale 1.
    private static readonly decimal[,] s_decimalCovariance =
    {
        { 1e-8m, 0m, 0m },
        { 0m, 2e-12m, -1.9998e-12m },
        { 0m, -1.9998e-12m, 2e-12m },
    };

    private static readonly decimal[] s_decimalMeans = [5e13m, -5.00072e13m, -5.00029e13m];

    // Exact optimum (interior of the box), to 28 significant digits.
    private static readonly decimal[] s_decimalOptimum =
    [
        10000504899.994951010050489899m,
        -5537779325.8412926958847879269m,
        -4462725573.1536583141657019726m,
    ];

    [Theory]
    [InlineData("1")]
    [InlineData("0.5")]
    [InlineData("0.1")]
    [InlineData("0.01")]
    public void QpMeanVariance_DecimalMeanSolveBeyondRange_ReturnsTheScaleInvariantOptimum(string scaleText)
    {
        var scale = decimal.Parse(scaleText, System.Globalization.CultureInfo.InvariantCulture);
        var means = s_decimalMeans.Select(m => m * scale).ToArray();

        decimal[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<decimal>.SolveMeanVariance(
            s_decimalCovariance, means, 1e12m * scale, -8e13m, 4e12m));

        error.Should().BeNull("the optimum is unchanged by an exact power-of-ten rescale of μ and λ and is representable");
        for (var i = 0; i < 3; i++)
        {
            weights[i].Should().BeApproximately(
                s_decimalOptimum[i], 1e-8m, "weight {0} must be the exact optimum to a relative 1e-18", i);
        }
    }

    // Witness (3): the budget direction s·Σ⁻¹·1 (s = the largest variance) is ≈ 5e308 although its
    // scale is arbitrary and cancels in every weight.
    private static readonly double[,] s_budgetCovariance =
    {
        { 1e150, 0.0, 0.0 },
        { 0.0, 2e-150, -1.999999998e-150 },
        { 0.0, -1.999999998e-150, 2e-150 },
    };

    [Fact]
    public void QpMeanVariance_BudgetDirectionSolveBeyondRange_ReturnsTheRepresentableOptimum()
    {
        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMeanVariance(
            s_budgetCovariance, [1.0, -1.0, 0.5], 1e-140, -1e100, 1e100));

        error.Should().BeNull("the budget direction's scale cancels and the optimum is representable");
        weights[0].Should().BeApproximately(2.0000000000000000718e-10, 1e100 * 1e-12);
        weights[1].Should().Be(-1e100);
        weights[2].Should().Be(1e100);
    }

    [Fact]
    public void QpMinVariance_BudgetDirectionSolveBeyondRange_ReturnsTheRepresentableOptimum()
    {
        double[] weights = [];
        var error = Record.Exception(() => weights = ActiveSetQpSolver<double>.SolveMinVariance(
            s_budgetCovariance, -1e100, 1e100));

        error.Should().BeNull("the budget direction's scale cancels and the optimum is representable");
        weights[0].Should().BeApproximately(9.999999510271909e-310, 1e-12);
        weights[1].Should().BeApproximately(0.5, 1e-12);
        weights[2].Should().BeApproximately(0.5, 1e-12);
    }
}
