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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3IndependentPSquareTests
{
    // Fractions independently evaluate printed Box1, including its k+1 erratum.
    // Printed TableI rounded heights are retained separately, never equality goldens.
    [Theory]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(18)]
    [InlineData(19)]
    [InlineData(20)]
    public void PaperRow_HasExactRationalMarkerState(int observationCount)
    {
        double[] stream = [0.02, 0.15, 0.74, 3.39, 0.83, 22.37, 10.15, 15.43, 38.62, 15.92, 34.60, 10.28, 1.47, 0.40, 0.05, 11.39, 0.27, 0.42, 0.09, 11.37];
        var estimator = new PSquareEstimator<double>(0.5);
        foreach (var value in stream[..observationCount])
        {
            estimator.Add(value);
        }

        var expected = observationCount switch
        {
            6 => (Positions: new long[] { 1L, 2L, 3L, 4L, 6L }, Heights: new double[] { 1.0 / 50.0, 3.0 / 20.0, 37.0 / 50.0, 83.0 / 100.0, 2237.0 / 100.0 }),
            7 => (Positions: new long[] { 1L, 2L, 3L, 5L, 7L }, Heights: new double[] { 1.0 / 50.0, 3.0 / 20.0, 37.0 / 50.0, 893.0 / 200.0, 2237.0 / 100.0 }),
            8 => (Positions: new long[] { 1L, 2L, 4L, 6L, 8L }, Heights: new double[] { 1.0 / 50.0, 3.0 / 20.0, 1307.0 / 600.0, 3437.0 / 400.0, 2237.0 / 100.0 }),
            9 => (Positions: new long[] { 1L, 3L, 5L, 7L, 9L }, Heights: new double[] { 1.0 / 50.0, 313.0 / 360.0, 51329.0 / 10800.0, 335167.0 / 21600.0, 1931.0 / 50.0 }),
            10 => (Positions: new long[] { 1L, 3L, 5L, 7L, 10L }, Heights: new double[] { 1.0 / 50.0, 313.0 / 360.0, 51329.0 / 10800.0, 335167.0 / 21600.0, 1931.0 / 50.0 }),
            11 => (Positions: new long[] { 1L, 3L, 6L, 8L, 11L }, Heights: new double[] { 1.0 / 50.0, 313.0 / 360.0, 534223.0 / 57600.0, 6212927.0 / 288000.0, 1931.0 / 50.0 }),
            12 => (Positions: new long[] { 1L, 3L, 6L, 9L, 12L }, Heights: new double[] { 1.0 / 50.0, 313.0 / 360.0, 534223.0 / 57600.0, 6212927.0 / 288000.0, 1931.0 / 50.0 }),
            13 => (Positions: new long[] { 1L, 4L, 7L, 10L, 13L }, Heights: new double[] { 1.0 / 50.0, 982639.0 / 460800.0, 534223.0 / 57600.0, 6212927.0 / 288000.0, 1931.0 / 50.0 }),
            14 => (Positions: new long[] { 1L, 5L, 8L, 11L, 14L }, Heights: new double[] { 1.0 / 50.0, 982639.0 / 460800.0, 534223.0 / 57600.0, 6212927.0 / 288000.0, 1931.0 / 50.0 }),
            15 => (Positions: new long[] { 1L, 5L, 8L, 12L, 15L }, Heights: new double[] { 1.0 / 50.0, 1262897.0 / 1728000.0, 15234433.0 / 2419200.0, 6212927.0 / 288000.0, 1931.0 / 50.0 }),
            16 => (Positions: new long[] { 1L, 5L, 8L, 13L, 16L }, Heights: new double[] { 1.0 / 50.0, 1262897.0 / 1728000.0, 15234433.0 / 2419200.0, 6212927.0 / 288000.0, 1931.0 / 50.0 }),
            17 => (Positions: new long[] { 1L, 5L, 9L, 13L, 17L }, Heights: new double[] { 1.0 / 50.0, 1271537.0 / 2160000.0, 15234433.0 / 2419200.0, 2080984261.0 / 120960000.0, 1931.0 / 50.0 }),
            18 => (Positions: new long[] { 1L, 6L, 10L, 14L, 18L }, Heights: new double[] { 1.0 / 50.0, 1271537.0 / 2160000.0, 15234433.0 / 2419200.0, 2080984261.0 / 120960000.0, 1931.0 / 50.0 }),
            19 => (Positions: new long[] { 1L, 6L, 10L, 15L, 19L }, Heights: new double[] { 1.0 / 50.0, 1280177.0 / 2592000.0, 14502756547.0 / 3265920000.0, 2080984261.0 / 120960000.0, 1931.0 / 50.0 }),
            20 => (Positions: new long[] { 1L, 6L, 10L, 16L, 20L }, Heights: new double[] { 1.0 / 50.0, 1280177.0 / 2592000.0, 14502756547.0 / 3265920000.0, 2080984261.0 / 120960000.0, 1931.0 / 50.0 }),
            _ => throw new ArgumentOutOfRangeException(nameof(observationCount)),
        };
        Read<long>(estimator, "_n").Should().Equal(expected.Positions);
        var heights = Read<double>(estimator, "_q");
        for (var i = 0; i < 5; i++)
        {
            heights[i].Should().BeApproximately(expected.Heights[i], 1e-12);
        }

        estimator.Estimate.Should().Be(heights[2]);
        estimator.Count.Should().Be(observationCount);
    }

    [Fact]
    public void Initialization_HasFiveSortedMarkersAndRetainsRequestedPercentile()
    {
        var estimator = new PSquareEstimator<double>(0.25);
        foreach (var value in new double[] { 4, 0, 3, 1, 2 })
        {
            estimator.Add(value);
        }

        Read<double>(estimator, "_q").Should().Equal(0, 1, 2, 3, 4);
        Read<long>(estimator, "_n").Should().Equal(1L, 2L, 3L, 4L, 5L);
        estimator.Estimate.Should().Be(2);
        estimator.Percentile.Should().Be(0.25);
    }

    [Theory]
    [InlineData(0, -1.0, 3L, 4L, 5L)]
    [InlineData(1, 0.0, 3L, 4L, 5L)]
    [InlineData(2, 1.0, 2L, 4L, 5L)]
    [InlineData(3, 2.0, 2L, 3L, 5L)]
    [InlineData(4, 3.0, 2L, 3L, 4L)]
    [InlineData(5, 4.0, 2L, 3L, 4L)]
    [InlineData(6, 5.0, 2L, 3L, 4L)]
    public void SixthObservation_UsesPrintedCellAndNoInteriorMove(int code, double observation, long second, long third, long fourth)
    {
        code.Should().BeInRange(0, 6);
        var estimator = new PSquareEstimator<double>(0.5);
        foreach (var value in new double[] { 0, 1, 2, 3, 4, observation })
        {
            estimator.Add(value);
        }

        Read<long>(estimator, "_n").Should().Equal(1L, second, third, fourth, 6L);
        Read<double>(estimator, "_q").Should().Equal(Math.Min(0, observation), 1, 2, 3, Math.Max(4, observation));
    }

    private static TValue[] Read<TValue>(PSquareEstimator<double> estimator, string field)
        => (TValue[])typeof(PSquareEstimator<double>).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(estimator)!;
}
