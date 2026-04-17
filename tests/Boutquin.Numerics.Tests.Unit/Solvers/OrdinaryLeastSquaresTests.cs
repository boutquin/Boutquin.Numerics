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
/// API contract tests for <see cref="OrdinaryLeastSquares"/>: intercept toggle, residual
/// arithmetic identities, R² invariants, DoF bookkeeping, and input validation. The
/// NIST-certified accuracy checks live in
/// <see cref="Boutquin.Numerics.Tests.Unit.Solvers.NistStRD.NistLinearRegressionTests"/>.
/// </summary>
public sealed class OrdinaryLeastSquaresTests
{
    [Fact]
    public void Fit_SimpleLinear_RecoversExactSlopeAndIntercept()
    {
        // y = 2 + 3·x over a perfect fit. Coefficients must be exact to round-off.
        var x = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
        };
        var y = new[] { 5.0, 8.0, 11.0, 14.0, 17.0 };

        var result = OrdinaryLeastSquares.Fit(x, y);

        result.Coefficients[0].Should().BeApproximately(2.0, 1e-12);
        result.Coefficients[1].Should().BeApproximately(3.0, 1e-12);
        result.ResidualSumOfSquares.Should().BeLessThan(1e-24);
        result.RSquared.Should().BeApproximately(1.0, 1e-12);
        result.DegreesOfFreedom.Should().Be(3);
    }

    [Fact]
    public void Fit_NoIntercept_ReturnsSingleCoefficient_ForSlopeThroughOrigin()
    {
        // Exact fit through origin: y = 2·x.
        var x = new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } };
        var y = new[] { 2.0, 4.0, 6.0, 8.0 };

        var result = OrdinaryLeastSquares.Fit(x, y, includeIntercept: false);

        result.Coefficients.Should().HaveCount(1);
        result.Coefficients[0].Should().BeApproximately(2.0, 1e-12);
        result.DegreesOfFreedom.Should().Be(3);
        // Uncentred R² definition when no intercept: 1 − RSS/Σy².
        result.RSquared.Should().BeApproximately(1.0, 1e-12);
    }

    [Fact]
    public void Fit_Residuals_SumOfSquares_MatchesRssProperty()
    {
        // Noise-free but with imperfect fit because y has a quadratic curvature the
        // straight-line model cannot absorb. Residual arithmetic must be internally
        // consistent: Σ rᵢ² == ResidualSumOfSquares.
        var rng = new System.Random(42);
        var x = new double[20, 1];
        var y = new double[20];
        for (var i = 0; i < 20; i++)
        {
            x[i, 0] = i;
            y[i] = 1.5 + 0.8 * i + 0.03 * i * i + (rng.NextDouble() - 0.5) * 0.1;
        }

        var result = OrdinaryLeastSquares.Fit(x, y);

        var recomputed = 0.0;
        foreach (var r in result.Residuals)
        {
            recomputed += r * r;
        }

        recomputed.Should().BeApproximately(result.ResidualSumOfSquares, 1e-10);
    }

    [Fact]
    public void Fit_RSquared_EqualsOneMinusRssOverTss_WithIntercept()
    {
        var x = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 },
        };
        var y = new[] { 2.1, 3.9, 6.1, 7.8, 10.2, 12.0 };

        var result = OrdinaryLeastSquares.Fit(x, y);

        // Recompute TSS against the response mean.
        var yMean = 0.0;
        foreach (var yi in y)
        {
            yMean += yi;
        }

        yMean /= y.Length;
        var tss = 0.0;
        foreach (var yi in y)
        {
            var d = yi - yMean;
            tss += d * d;
        }

        var expected = 1.0 - result.ResidualSumOfSquares / tss;
        result.RSquared.Should().BeApproximately(expected, 1e-12);
    }

    [Fact]
    public void Fit_StandardErrors_MatchSqrtDiagOfCovariance()
    {
        var x = new double[,]
        {
            { 1.0, 2.0 }, { 2.0, 3.0 }, { 3.0, 5.0 }, { 4.0, 7.0 },
            { 5.0, 11.0 }, { 6.0, 13.0 }, { 7.0, 17.0 }, { 8.0, 19.0 },
        };
        var y = new[] { 3.1, 5.2, 7.9, 10.3, 14.9, 17.1, 21.8, 25.1 };

        var result = OrdinaryLeastSquares.Fit(x, y);

        for (var i = 0; i < result.StandardErrors.Length; i++)
        {
            result.StandardErrors[i].Should().BeApproximately(
                Math.Sqrt(result.CovarianceMatrix[i, i]), 1e-12);
        }
    }

    [Fact]
    public void Fit_CovarianceMatrix_IsSymmetric()
    {
        // Use non-collinear columns (primes on the third column) so the design matrix has
        // full column rank and the covariance is well-defined.
        var x = new double[,]
        {
            { 1.0, 2.0, 0.7 },
            { 2.0, 3.0, 1.3 },
            { 3.0, 5.0, 1.9 },
            { 4.0, 7.0, 2.3 },
            { 5.0, 11.0, 2.9 },
            { 6.0, 13.0, 3.1 },
        };
        var y = new[] { 4.1, 6.8, 10.2, 13.5, 18.1, 21.7 };

        var result = OrdinaryLeastSquares.Fit(x, y);

        var p = result.CovarianceMatrix.GetLength(0);
        for (var i = 0; i < p; i++)
        {
            for (var j = i + 1; j < p; j++)
            {
                Math.Abs(result.CovarianceMatrix[i, j] - result.CovarianceMatrix[j, i])
                    .Should().BeLessThan(1e-12);
            }
        }
    }

    [Fact]
    public void Fit_RejectsNullDesignMatrix()
    {
        FluentActions.Invoking(() => OrdinaryLeastSquares.Fit(null!, new[] { 1.0 }))
            .Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Fit_RejectsNullResponse()
    {
        FluentActions.Invoking(() => OrdinaryLeastSquares.Fit(new double[1, 1], null!))
            .Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Fit_RejectsMismatchedDimensions()
    {
        var x = new double[,] { { 1.0 }, { 2.0 }, { 3.0 } };
        var y = new[] { 1.0, 2.0 };

        FluentActions.Invoking(() => OrdinaryLeastSquares.Fit(x, y))
            .Should().Throw<ArgumentException>()
            .WithMessage("*does not match*");
    }

    [Fact]
    public void Fit_RejectsEmptyDesignMatrix()
    {
        FluentActions.Invoking(() => OrdinaryLeastSquares.Fit(new double[0, 1], Array.Empty<double>()))
            .Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Fit_RejectsUnderDeterminedSystem()
    {
        // 2 observations but model has intercept + 2 predictors = 3 coefficients.
        var x = new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } };
        var y = new[] { 1.0, 2.0 };

        FluentActions.Invoking(() => OrdinaryLeastSquares.Fit(x, y))
            .Should().Throw<ArgumentException>()
            .WithMessage("*under-determined*");
    }

    [Fact]
    public void Fit_RankDeficientDesign_ThrowsInvalidOperation()
    {
        // Two identical predictor columns → collinear → QR produces a zero reflector
        // in the second column.
        var x = new double[,]
        {
            { 1.0, 1.0 }, { 2.0, 2.0 }, { 3.0, 3.0 }, { 4.0, 4.0 }, { 5.0, 5.0 },
        };
        var y = new[] { 2.0, 4.0, 6.0, 8.0, 10.0 };

        FluentActions.Invoking(() => OrdinaryLeastSquares.Fit(x, y))
            .Should().Throw<InvalidOperationException>()
            .WithMessage("*rank-deficient*");
    }

    [Fact]
    public void Fit_DoesNotMutateInputs()
    {
        var x = new double[,] { { 1.0 }, { 2.0 }, { 3.0 } };
        var y = new[] { 2.0, 4.0, 6.0 };

        var xSnapshot = new double[3, 1];
        Array.Copy(x, xSnapshot, x.Length);
        var ySnapshot = (double[])y.Clone();

        _ = OrdinaryLeastSquares.Fit(x, y);

        for (var i = 0; i < 3; i++)
        {
            x[i, 0].Should().Be(xSnapshot[i, 0]);
            y[i].Should().Be(ySnapshot[i]);
        }
    }
}
