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

namespace Boutquin.Numerics.Tests.Unit.Statistics;

/// <summary>
/// Pins <see cref="DistanceCorrelation{T}.Compute"/> on samples whose empirical joint
/// distribution is the product of its marginals (every x-level crossed with every
/// y-level). Székely, Rizzo &amp; Bakirov (2007) eq. 2.8 gives the V-statistic
/// <c>dCov²_n = (1/n²)·Σ A_ij·B_ij</c>, which is exactly zero for such a design, so the
/// sample distance correlation is exactly zero. Rounding in the doubly-centred sums can
/// leave a tiny negative <c>dCov²</c>; the documented <c>[0, 1]</c> range requires that it
/// map to zero rather than to the NaN of a square root of a negative number.
/// </summary>
public sealed class DistanceCorrelationProductDesignTests
{
    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    public void Compute_HalfProductDesign_ReturnsZero(int count)
    {
        var x = new Half[count];
        var y = new Half[count];
        for (var i = 0; i < count; i++)
        {
            x[i] = (Half)(i % 2 == 0 ? -1 : 1);
            y[i] = (Half)(i % 4 < 2 ? -1 : 1);
        }

        DistanceCorrelation<Half>.Compute(x, y).Should().Be(Half.Zero,
            "the empirical joint law factorizes exactly");
    }

    [Fact]
    public void Compute_ThreeByThreeProductDesign_ReturnsZeroNotNaN()
    {
        // 3 x-levels crossed with 3 y-levels: the empirical joint law is a product measure.
        double[] xs = [0.13, 0.57, 0.91];
        double[] ys = [0.21, 0.44, 0.83];
        var x = new double[9];
        var y = new double[9];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                x[(3 * i) + j] = xs[i];
                y[(3 * i) + j] = ys[j];
            }
        }

        var dcor = DistanceCorrelation<double>.Compute(x, y);

        double.IsNaN(dcor).Should().BeFalse("the documented range of the sample distance correlation is [0, 1]");
        dcor.Should().BeInRange(0.0, 1e-6, "dCov² is exactly zero for a product design (SRB 2007 eq. 2.8)");
    }

    [Fact]
    public void Compute_RandomProductDesigns_AlwaysInUnitIntervalAndNearZero()
    {
        var rng = new System.Random(20260927);
        for (var trial = 0; trial < 500; trial++)
        {
            var k = rng.Next(2, 6);
            var m = rng.Next(2, 6);
            var xs = new double[k];
            var ys = new double[m];
            for (var i = 0; i < k; i++)
            {
                xs[i] = rng.NextDouble() * 10.0 - 5.0;
            }

            for (var j = 0; j < m; j++)
            {
                ys[j] = rng.NextDouble() * 10.0 - 5.0;
            }

            var x = new double[k * m];
            var y = new double[k * m];
            for (var i = 0; i < k; i++)
            {
                for (var j = 0; j < m; j++)
                {
                    x[(m * i) + j] = xs[i];
                    y[(m * i) + j] = ys[j];
                }
            }

            var dcor = DistanceCorrelation<double>.Compute(x, y);

            dcor.Should().BeInRange(0.0, 1e-6, $"trial {trial} (k = {k}, m = {m}) is a product design");
        }
    }
}
