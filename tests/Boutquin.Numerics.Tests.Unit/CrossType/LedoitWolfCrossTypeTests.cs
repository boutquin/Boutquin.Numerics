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

using Boutquin.Numerics.Statistics;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.CrossType;

/// <summary>
/// Cross-type regression test: <see cref="LedoitWolfShrinkageEstimator{T}"/> at
/// <c>T = double</c> and <c>T = decimal</c> on a returns matrix. The two shrinkage
/// intensities must agree to 10 digits, proving the arithmetic-only shrinkage
/// formulas are type-invariant.
/// </summary>
public sealed class LedoitWolfCrossTypeTests
{
    [Fact]
    public void LedoitWolf_DoubleAndDecimal_ShrinkageAgreeToTenDigits()
    {
        // Generate a small returns matrix: 20 observations, 3 assets.
        // Using simple deterministic data that exercises the shrinkage formula.
        var rng = new System.Random(42);
        var t = 20;
        var n = 3;

        var returnsDouble = new double[t, n];
        var returnsDecimal = new decimal[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var v = (rng.NextDouble() - 0.5) * 0.1;
                returnsDouble[i, j] = v;
                returnsDecimal[i, j] = (decimal)v;
            }
        }

        // Run at double.
        var covDouble = new LedoitWolfShrinkageEstimator<double>().Estimate(returnsDouble);

        // Run at decimal.
        var covDecimal = new LedoitWolfShrinkageEstimator<decimal>().Estimate(returnsDecimal);

        // Compare covariance matrices to 10 significant digits.
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var d = covDouble[i, j];
                var dec = (double)covDecimal[i, j];

                if (d == 0.0)
                {
                    Math.Abs(dec).Should().BeLessThan(1e-15,
                        because: $"cov[{i},{j}] should be near-zero in both types");
                    continue;
                }

                var relError = Math.Abs((dec - d) / d);
                relError.Should().BeLessThan(1e-10,
                    because: $"cov[{i},{j}]: double={d:E6}, decimal={dec:E6} must agree to 10 digits");
            }
        }
    }
}
