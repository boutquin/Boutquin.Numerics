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

using System.Reflection;
using Boutquin.Numerics.Distributions;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Distributions;

public sealed class Arena20261001DistributionTests
{
    [Theory]
    [InlineData("A1", -3.969683028665376e+01)]
    [InlineData("A2", 2.209460984245205e+02)]
    [InlineData("A3", -2.759285104469687e+02)]
    [InlineData("A4", 1.383577518672690e+02)]
    [InlineData("A5", -3.066479806614716e+01)]
    [InlineData("A6", 2.506628277459239e+00)]
    [InlineData("B1", -5.447609879822406e+01)]
    [InlineData("B2", 1.615858368580409e+02)]
    [InlineData("B3", -1.556989798598866e+02)]
    [InlineData("B4", 6.680131188771972e+01)]
    [InlineData("B5", -1.328068155288572e+01)]
    [InlineData("C1", -7.784894002430293e-03)]
    [InlineData("C2", -3.223964580411365e-01)]
    [InlineData("C3", -2.400758277161838e+00)]
    [InlineData("C4", -2.549732539343734e+00)]
    [InlineData("C5", 4.374664141464968e+00)]
    [InlineData("C6", 2.938163982698783e+00)]
    [InlineData("D1", 7.784695709041462e-03)]
    [InlineData("D2", 3.224671290700398e-01)]
    [InlineData("D3", 2.445134137142996e+00)]
    [InlineData("D4", 3.754408661907416e+00)]
    public void Acklam_PinsOriginalCoefficient(string name, double expected)
    {
        Coefficient(name).Should().Be(expected);
    }

    [Theory]
    [InlineData(0.024249999999999997, -1.972961051311885)]
    [InlineData(0.02425, -1.972961051311885)]
    [InlineData(0.024250000000000004, -1.9729610513118847)]
    [InlineData(0.1, -1.2815515655446004)]
    [InlineData(0.3, -0.5244005127080408)]
    [InlineData(0.9757499999999999, 1.972961051311883)]
    [InlineData(0.97575, 1.972961051311885)]
    [InlineData(0.9757500000000001, 1.972961051311887)]
    [InlineData(1e-20, -9.262340089798407)]
    [InlineData(1e-08, -5.612001244174789)]
    public void Acklam_RationalBranchesAndOneSidedBoundaries(double p, double expected)
    {
        // Evaluate the original published rational form from pinned production
        // constants, independently of its two Newton polishing steps.
        Coefficient("PLow").Should().Be(.02425);
        var lower = Math.Min(p, 1 - p);
        double raw;
        if (lower < .02425)
        {
            var q = Math.Sqrt(-2 * Math.Log(lower));
            raw = Polynomial("C", 6, q) / (Polynomial("D", 4, q) * q + 1);
        }
        else
        {
            var q = lower - .5;
            var r = q * q;
            raw = q * Polynomial("A", 6, r) / (Polynomial("B", 5, r) * r + 1);
        }

        if (p > .5) { raw = -raw; }
        Math.Abs((raw - expected) / expected).Should().BeLessThan(1.15e-9);
        InverseNormal<double>.Evaluate(p).Should().BeApproximately(expected, 3e-12);
    }

    [Fact]
    public void SubnormalCdf_UsesRoundedHighPrecisionOracle()
    {
        const double Expected = 2.88542836006878430835e-316;
        var actual = CumulativeNormal<double>.Evaluate(-38);
        Math.Abs(BitConverter.DoubleToInt64Bits(actual) - BitConverter.DoubleToInt64Bits(Expected)).Should().BeLessThanOrEqualTo(2);
    }

    [Fact]
    public void SmallestProbability_UsesIndependentQuantile()
    {
        InverseNormal<double>.Evaluate(double.Epsilon).Should().BeApproximately(-38.46740561714434625078, 1e-6);
    }

    private static double Coefficient(string name) =>
        (double)typeof(InverseNormal<double>).GetField(name, BindingFlags.NonPublic | BindingFlags.Static)!.GetRawConstantValue()!;

    private static double Polynomial(string prefix, int count, double x)
    {
        var value = Coefficient(prefix + "1");
        for (var i = 2; i <= count; i++) { value = value * x + Coefficient(prefix + i); }
        return value;
    }
}
