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
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class C3DistanceRationalControls
{
    private delegate double[,] Center(ReadOnlySpan<double> x, int n);

    [Fact]
    public void ActualCentering_HasIndependentRationalEntries()
    {
        // Literal primary distances on x=(0,1,5), y=(0,2,1), using exact
        // Fraction arithmetic. Public normalization cancels global distance
        // factors, so inspect the actual production primitive as well.
        var center = typeof(DistanceCorrelation<double>).GetMethod("DoublyCenter", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Center>();
        AssertMatrix(center([0, 1, 5], 3), A());
        AssertMatrix(center([0, 2, 1], 3), B());
    }

    [Fact]
    public void ActualVStatistics_HaveIndependentRationalNormalization()
    {
        // Direct production primitive makes n^-2 discriminating: a common
        // multiplicative error would cancel out of public correlation.
        var method = typeof(DistanceCorrelation<double>).GetMethod("MeanProduct", BindingFlags.Static | BindingFlags.NonPublic)!;
        var product = method.CreateDelegate<Func<double[,], double[,], int, double>>();
        product(A(), B(), 3).Should().BeApproximately(52.0 / 81, 1e-14);
        product(A(), A(), 3).Should().BeApproximately(304.0 / 81, 1e-14);
        product(B(), B(), 3).Should().BeApproximately(40.0 / 81, 1e-14);
    }

    [Fact]
    public void SampleCorrelation_UsesPrimarySquaredRatioAndFinalRoot()
    {
        var expected = Math.Sqrt((52.0 / 81) / Math.Sqrt((304.0 / 81) * (40.0 / 81)));
        DistanceCorrelation<double>.Compute([0, 1, 5], [0, 2, 1]).Should().BeApproximately(expected, 1e-14);
        DistanceCorrelation<double>.Compute([1000000000000, 1000000000001, 1000000000005], [0, 2, 1]).Should().BeApproximately(expected, 1e-14);
    }

    [Fact]
    public void ConstantSample_HasZeroDistanceVarianceAndCorrelation()
    {
        DistanceCorrelation<double>.Compute([3, 3, 3], [0, 1, 5]).Should().Be(0);
        DistanceCorrelation<double>.Compute([0, 1, 5], [3, 3, 3]).Should().Be(0);
    }

    private static double[,] A() => new double[,]
    {
        { -16.0 / 9, -4.0 / 9, 20.0 / 9 },
        { -4.0 / 9, -10.0 / 9, 14.0 / 9 },
        { 20.0 / 9, 14.0 / 9, -34.0 / 9 },
    };

    private static double[,] B() => new double[,]
    {
        { -10.0 / 9, 8.0 / 9, 2.0 / 9 },
        { 8.0 / 9, -10.0 / 9, 2.0 / 9 },
        { 2.0 / 9, 2.0 / 9, -4.0 / 9 },
    };

    private static void AssertMatrix(double[,] actual, double[,] expected)
    {
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual[i, j].Should().BeApproximately(expected[i, j], 1e-14);
            }
        }
    }
}
