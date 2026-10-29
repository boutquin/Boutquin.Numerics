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
/// Pins how <see cref="OrdinaryLeastSquares{T}"/> behaves away from unit scale and
/// outside <c>double</c>: the <c>decimal</c> factorisation keeps its digits on
/// small-magnitude data, <c>Half</c> designs whose squares exceed its range still fit,
/// and the augmented-system refinement at <c>T = double</c> recovers the exact solution
/// of a consistent, ill-conditioned, non-integer system.
/// </summary>
public sealed class OrdinaryLeastSquaresScaleAndRangeTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    public void Decimal_LargeConstantResponse_InterceptMeanDoesNotOverflow(int count)
    {
        var y = Enumerable.Repeat(1e28m, count).ToArray();
        OlsResult<decimal>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<decimal>.Fit(new decimal[count, 0], y))
            .Should().NotThrow();
        result!.Coefficients[0].Should().Be(1e28m);
        result.ResidualSumOfSquares.Should().Be(0m);
        result.RSquared.Should().Be(0m);
    }

    [Fact]
    public void Decimal_LargeRepresentableExactFit_DoesNotOverflowConvergenceDiagnostic()
    {
        decimal[,] x = { { 1m }, { 1m }, { 1m } };
        decimal[] y = [1e15m, 1e15m, 1e15m];
        OlsResult<decimal>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<decimal>.Fit(x, y, false))
            .Should().NotThrow();
        result!.Coefficients[0].Should().Be(1e15m);
        result.ResidualSumOfSquares.Should().Be(0m);
    }

    [Theory]
    [InlineData(double.NaN, false)]
    [InlineData(double.PositiveInfinity, false)]
    [InlineData(double.NegativeInfinity, false)]
    [InlineData(double.NaN, true)]
    [InlineData(double.PositiveInfinity, true)]
    [InlineData(double.NegativeInfinity, true)]
    public void Fit_NonFiniteInput_RejectsAtBoundary(double value, bool inDesign)
    {
        double[,] x = { { 1 }, { 2 }, { 3 } };
        double[] y = [1, 2, 3];
        if (inDesign)
        {
            x[1, 0] = value;
        }
        else
        {
            y[1] = value;
        }

        FluentActions.Invoking(() => OrdinaryLeastSquares<double>.Fit(x, y, false))
            .Should().Throw<ArgumentException>().WithParameterName(inDesign ? "x" : "y");
    }

    [Fact]
    public void Decimal_LargeResponse_UncenteredRSquaredRemainsRepresentable()
    {
        decimal[,] x = { { 1m }, { 1m }, { 1m } };
        decimal[] y = [1e15m - 1m, 1e15m, 1e15m + 1m];
        OlsResult<decimal>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<decimal>.Fit(x, y, false))
            .Should().NotThrow();
        result!.ResidualSumOfSquares.Should().Be(2m);
        result.RSquared.Should().Be(1m);
    }

    // Noisy two-column no-intercept design with six-decimal entries, so every scaled copy
    // below is exactly representable in decimal.
    private static readonly decimal[,] s_unitX =
    {
        { 0.183746m, 0.927364m }, { 0.562910m, 0.103847m }, { 0.938271m, 0.475610m },
        { 0.271834m, 0.662019m }, { 0.749201m, 0.318273m }, { 0.415926m, 0.835214m },
        { 0.068153m, 0.249371m }, { 0.827364m, 0.590283m }, { 0.354710m, 0.019284m },
        { 0.690817m, 0.781920m }, { 0.129384m, 0.456102m }, { 0.502938m, 0.940173m },
    };

    private static readonly decimal[] s_unitY =
    {
        -1.271034m, 1.483129m, 1.869402m, -0.492110m, 1.620981m, -0.411573m,
        -0.301270m, 1.305116m, 1.029181m, 0.528390m, -0.513291m, -0.378214m,
    };

    /// <summary>
    /// OLS coefficients are invariant when <c>X</c> and <c>y</c> are scaled together, so a
    /// fit of the scaled data must reproduce the unit-scale fit. <c>decimal</c> resolves
    /// 1e-28 absolutely; unless every Householder reflector is normalised, products of
    /// order <c>s²</c> keep only <c>28 + 2·log10(s)</c> digits.
    /// </summary>
    [Theory]
    [InlineData(-8)]
    [InlineData(-10)]
    [InlineData(-12)]
    public void Decimal_JointlyScaledData_ReproducesUnitScaleCoefficients(int log10Scale)
    {
        var s = 1m;
        for (var k = 0; k < -log10Scale; k++)
        {
            s /= 10m;
        }

        var m = s_unitY.Length;
        var scaledX = new decimal[m, 2];
        var scaledY = new decimal[m];
        for (var i = 0; i < m; i++)
        {
            scaledX[i, 0] = s_unitX[i, 0] * s;
            scaledX[i, 1] = s_unitX[i, 1] * s;
            scaledY[i] = s_unitY[i] * s;
        }

        var unit = OrdinaryLeastSquares<decimal>.Fit(s_unitX, s_unitY, includeIntercept: false).Coefficients;
        var scaled = OrdinaryLeastSquares<decimal>.Fit(scaledX, scaledY, includeIntercept: false).Coefficients;

        for (var k = 0; k < 2; k++)
        {
            var relative = Math.Abs((scaled[k] - unit[k]) / unit[k]);
            relative.Should().BeLessThan(1e-15m,
                because: $"b{k} at scale 1e{log10Scale}: {scaled[k]} vs unit-scale {unit[k]}");
        }
    }

    /// <summary>
    /// <c>y = 2·x1 − x2</c> exactly on a well-conditioned design (κ ≈ 3) whose column norms
    /// (≈ 300) square past <c>Half.MaxValue</c> = 65504. The reflector must be normalised
    /// against <c>Half</c>'s own range; a <c>decimal.MaxValue</c> sentinel converts to
    /// <c>+∞</c> in <c>Half</c>, the reflector is never normalised, and <c>vtv</c> overflows.
    /// </summary>
    [Fact]
    public void Half_WellConditionedColumnsBeyondSquareRange_RecoverExactCoefficients()
    {
        double[,] predictors = { { 120, 40 }, { 80, 150 }, { 200, 90 }, { 60, 180 }, { 160, 130 }, { 100, 70 } };
        var x = new Half[6, 2];
        var y = new Half[6];
        for (var i = 0; i < 6; i++)
        {
            x[i, 0] = (Half)predictors[i, 0];
            x[i, 1] = (Half)predictors[i, 1];
            y[i] = (Half)((2 * predictors[i, 0]) - predictors[i, 1]);
        }

        OlsResult<Half>? result = null;
        FluentActions.Invoking(() => result = OrdinaryLeastSquares<Half>.Fit(x, y, includeIntercept: false))
            .Should().NotThrow(because: "the design has full rank with κ ≈ 3");

        var coefficients = result!.Coefficients;
        ((double)coefficients[0]).Should().BeApproximately(2.0, 0.01, because: $"b0 = {coefficients[0]}");
        ((double)coefficients[1]).Should().BeApproximately(-1.0, 0.01, because: $"b1 = {coefficients[1]}");
    }

    /// <summary>
    /// A consistent system (<c>y = X·β</c> exactly, verified in integer arithmetic) has the
    /// least-squares solution <c>β</c> itself. Columns 2 and 3 differ from columns 0 and 1 by
    /// <c>2^-d</c>-sized dyadic perturbations, so κ(X) grows as <c>2^d</c> and every entry is
    /// non-integer. The augmented refinement must evaluate <c>g = −Xᵀr</c> at the residual's
    /// own scale: at the response's scale its fixed <c>decimal</c> resolution, amplified by
    /// <c>1/σ_min²</c>, leaves an error that grows as κ² instead of vanishing with <c>r</c>.
    /// </summary>
    [Theory]
    [InlineData(27)]
    [InlineData(33)]
    [InlineData(38)]
    public void Double_ConsistentIllConditionedDyadicSystem_RecoversExactSolution(int d)
    {
        const int m = 30;
        double[] beta = [1, -2, 3, 1];
        var rng = new System.Random(11);
        var delta = Math.ScaleB(1.0, -d);
        var x = new double[m, 4];
        var y = new double[m];
        for (var i = 0; i < m; i++)
        {
            x[i, 0] = 1.0 + (rng.Next(1024) / 1024.0);
            x[i, 1] = 1.0 + (rng.Next(1024) / 1024.0);
            x[i, 2] = x[i, 0] + (delta * (rng.Next(2048) - 1024) / 1024.0);
            x[i, 3] = x[i, 1] + (delta * (rng.Next(2048) - 1024) / 1024.0);
            for (var k = 0; k < 4; k++)
            {
                y[i] += beta[k] * x[i, k];
            }

            // Every value is a multiple of 2^-60, so scaling by 2^60 gives exact integers.
            var exact = System.Numerics.BigInteger.Zero;
            for (var k = 0; k < 4; k++)
            {
                exact += new System.Numerics.BigInteger(beta[k]) * new System.Numerics.BigInteger(Math.ScaleB(x[i, k], 60));
            }

            new System.Numerics.BigInteger(Math.ScaleB(y[i], 60)).Should().Be(exact, because: $"row {i} must be consistent exactly");
        }

        var coefficients = OrdinaryLeastSquares<double>.Fit(x, y, includeIntercept: false).Coefficients;

        for (var k = 0; k < 4; k++)
        {
            Math.Abs((coefficients[k] - beta[k]) / beta[k]).Should().BeLessThan(1e-15,
                because: $"b{k} at κ ~ 2^{d}: {coefficients[k]:R} vs exact {beta[k]}");
        }
    }
}
