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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

/// <summary>
/// Premature scalar narrowing: a fractional rank, Beta weight, or parabolic
/// coefficient must not be rounded to the working type before the magnitude that
/// makes the completed result representable has been applied. Expected values come
/// from independent oracles: exact rational products of the binary double rank,
/// SciPy's complementary regularized incomplete beta, and an exact-rational
/// Jain-Chlamtac P² trajectory.
/// </summary>
public sealed class Arena414074R1PercentileTests
{
    private const string Ten28 = "10000000000000000000000000000";

    // Exact rational value of low + p·(high − low) for the binary double p.
    [Theory]
    [InlineData(1e-30, "0", Ten28, "0.0100000000000000008333642061")]
    [InlineData(1e-29, "0", Ten28, "0.0999999999999999943206574175")]
    [InlineData(1e-30, "-1", Ten28, "-0.9899999999999999991666357939")]
    public void DecimalPercentile_TinyRank_PreservesAmplifiedFraction(
        double p, string low, string high, string exact)
    {
        decimal[] sorted = [Parse(low), Parse(high)];
        var expected = Parse(exact);

        var actual = Percentile.Compute<decimal>(sorted, p);

        actual.Should().BeApproximately(expected, 1e-16m);
    }

    [Fact]
    public void DecimalPSquareStartup_TinyRank_PreservesAmplifiedFraction()
    {
        var estimator = new PSquareEstimator<decimal>(1e-30);
        estimator.Add(0m);
        estimator.Add(Parse(Ten28));

        estimator.Estimate.Should().BeApproximately(Parse("0.0100000000000000008333642061"), 1e-16m);
    }

    // SciPy betaincc(3p, 3(1 − p), 1/2) · 60000 is the weighted order statistic.
    // The tolerance is half a binary16 spacing plus 1e-6, so only a correctly
    // rounded completed estimate passes.
    [Theory]
    [InlineData(1e-7, 0.0012266499789011106)]
    [InlineData(1e-6, 0.01226656538310273)]
    [InlineData(1e-4, 1.2273781865809503)]
    public void HalfHarrellDavis_SmallBetaMass_RoundsCompletedEstimateOnce(double p, double oracle)
    {
        var actual = HarrellDavisPercentile.Compute<Half>([(Half)0, (Half)60000], p);

        ((double)actual).Should().BeApproximately(oracle, (HalfSpacing(oracle) / 2) + 1e-6);
    }

    // Stream A reaches the first marker-2 move at count 30001 with
    // cPost = 2 / (15000 · 14999) ≈ 8.9e-9, below Half's least subnormal.
    [Fact]
    public void HalfPSquare_TinyPostCoefficient_KeepsParabolicMove()
    {
        var estimator = new PSquareEstimator<Half>(1e-4);
        foreach (var value in new[] { 0, 0, 0, 60000, 60000 })
        {
            estimator.Add((Half)value);
        }

        for (var i = 0; i < 14998; i++)
        {
            estimator.Add((Half)0);
            estimator.Add((Half)60000);
        }

        estimator.Count.Should().Be(30001);
        var actual = (double)estimator.Estimate;
        actual.Should().BeApproximately(3.554821014404297e-4, HalfSpacing(3.554821014404297e-4));
        actual.Should().BeApproximately(3.5560593017338937e-4, 3.5560593017338937e-4 * 0.005);
    }

    // Stream B reaches the first marker-2 move at count 30002 for the exact
    // binary-double p=0.9999 (the ideal decimal boundary is at 30001), with
    // cPre = −2 / (15000 · 14999) ≈ −8.9e-9; the exact move is −4/14999.
    [Fact]
    public void HalfPSquare_TinyPreCoefficient_KeepsParabolicMove()
    {
        var estimator = new PSquareEstimator<Half>(0.9999);
        foreach (var value in new[] { -60000, -30000, 0, 0, 0 })
        {
            estimator.Add((Half)value);
        }

        for (var i = 0; i < 14998; i++)
        {
            estimator.Add((Half)(-60000));
            estimator.Add((Half)(-30000));
        }

        estimator.Add((Half)(-60000));
        estimator.Count.Should().Be(30002);
        var actual = (double)estimator.Estimate;
        actual.Should().BeApproximately(-2.6679039001464844e-4, HalfSpacing(2.6679039001464844e-4));
        actual.Should().BeApproximately(-4.0 / 14999, 4.0 / 14999 * 0.005);
    }

    private static decimal Parse(string value) => decimal.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

    // Distance between adjacent binary16 values at |x|.
    private static double HalfSpacing(double x)
    {
        var exponent = Math.Max(Math.ILogB(Math.Abs(x)), -14);
        return Math.ScaleB(1.0, exponent - 10);
    }
}
