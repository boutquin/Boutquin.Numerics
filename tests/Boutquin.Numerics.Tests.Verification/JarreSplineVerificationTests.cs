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

using System.Text.Json;
using Boutquin.Numerics.Interpolation;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Verification;

public sealed class JarreSplineVerificationTests : CrossLanguageVerificationBase
{
    // Jarre, arXiv:2507.05083v1 (July 2025), §5, Tables 1–5. The independent
    // historical generator retains all 100 printed cells. NAT/NAK remain live
    // production oracles. July Q/RNAK values and their 11 historical disagreements
    // stay archived; the authenticated October cases below govern their successor.
    // Random meshes are independent reconstructions, never the author's lost knots.
    [Fact]
    public void Table1_SineZeroEndpointCurvature_UniformMesh()
    {
        VerifyTable(1, Math.Sin, reconstructed: false, expectedDisagreements: 2);
    }

    [Fact]
    public void Table2_ShiftedSine_UniformMesh()
    {
        VerifyTable(2, Math.Sin, reconstructed: false, expectedDisagreements: 7);
    }

    [Fact]
    public void Table3_ShiftedSine_ReconstructedRandomMesh()
    {
        VerifyTable(3, Math.Sin, reconstructed: true, expectedDisagreements: 0);
    }

    [Fact]
    public void Table4_RungeFunction_ReconstructedRandomMesh()
    {
        VerifyTable(4, x => 1 / (1 + x * x), reconstructed: true, expectedDisagreements: 0);
    }

    [Fact]
    public void Table5_Logistic_UniformMesh()
    {
        VerifyTable(5, x => 1 / (1 + Math.Exp(-x)), reconstructed: false, expectedDisagreements: 2);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
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
    [InlineData(21)]
    [InlineData(22)]
    [InlineData(23)]
    [InlineData(24)]
    [InlineData(25)]
    [InlineData(26)]
    [InlineData(27)]
    [InlineData(28)]
    [InlineData(29)]
    public void AuthenticatedOctoberAuthorCoefficients_MatchBetweenKnots(int caseIndex)
    {
        using var document = LoadVector("jarre_author_20251029");
        var row = document.RootElement.GetProperty("cases")[caseIndex];
        var xs = GetDoubleArray(row.GetProperty("x"));
        var ys = GetDoubleArray(row.GetProperty("f"));
        var method = row.GetProperty("method").GetString();
        var boundary = method == "qspline" ? SplineBoundary<double>.QSpline : SplineBoundary<double>.RevisedNotAKnot;
        var spline = new CubicSplineInterpolator<double>(xs, ys, boundary);
        // Preserve both resolutions: the 96-knot shifted-sine Q maximum rounds
        // differently on 20001 versus the author's 1001-point plotting grid.
        double Function(double x) => row.GetProperty("function").GetString() == "logistic" ? 1 / (1 + Math.Exp(-x)) : Math.Sin(x);
        var denseMaximum = 0d;
        var plottingMaximum = 0d;
        for (var j = 0; j <= 20000; j++)
        {
            var x = xs[0] + (xs[^1] - xs[0]) * j / 20000;
            var error = Math.Abs(spline.Interpolate(x) - Function(x));
            denseMaximum = Math.Max(denseMaximum, error);
            if (j % 20 == 0) { plottingMaximum = Math.Max(plottingMaximum, error); }
        }
        denseMaximum.Should().BeApproximately(row.GetProperty("maxError").GetDouble(), 2e-12);
        plottingMaximum.Should().BeApproximately(row.GetProperty("nativeGrid1001Max").GetDouble(), 2e-12);
        var printed = row.GetProperty("journalPrinted").GetDouble();
        var halfUnit = .5 * Math.Pow(10, Math.Floor(Math.Log10(printed)) - 1);
        var printedComparison = row.GetProperty("printedComparisonGrid").GetInt32() == 1001 ? plottingMaximum : denseMaximum;
        printedComparison.Should().BeApproximately(printed, halfUnit + 2e-12);
        var coefficients = row.GetProperty("S");
        for (var interval = 0; interval < xs.Length - 1; interval++)
        {
            var c = GetDoubleArray(coefficients[interval]);
            foreach (var fraction in new[] { .137, .413, .829 })
            {
                var dx = fraction * (xs[interval + 1] - xs[interval]);
                var expected = c[0] + dx * (c[1] + dx * (c[2] + dx * c[3]));
                spline.Interpolate(xs[interval] + dx).Should().BeApproximately(expected, 2e-12,
                    $"authenticated October author {method}, case {caseIndex}, interval {interval}");
            }
        }
    }

    private static void VerifyTable(int tableNumber, Func<double, double> function, bool reconstructed, int expectedDisagreements)
    {
        using var document = LoadVector("jarre_splines");
        var table = document.RootElement.GetProperty("tables")[tableNumber - 1];
        table.GetProperty("table").GetInt32().Should().Be(tableNumber);
        table.GetProperty("mesh").GetString().Should().Be(reconstructed ? "independent_reconstruction" : "published_uniform");
        int[] counts = [6, 12, 24, 48, 96];
        var disagreements = 0;
        for (var index = 0; index < counts.Length; index++)
        {
            var testCase = table.GetProperty("cases")[index];
            var xs = GetDoubleArray(testCase.GetProperty("knots"));
            var ys = GetDoubleArray(testCase.GetProperty("values"));
            xs.Length.Should().Be(counts[index]);
            for (var i = 0; i < xs.Length; i++)
            {
                ys[i].Should().BeApproximately(function(xs[i]), 3e-16);
                if (!reconstructed)
                {
                    xs[i].Should().BeApproximately(xs[0] + (xs[^1] - xs[0]) * i / (xs.Length - 1), 2e-15);
                }
            }

            var rows = testCase.GetProperty("results").EnumerateArray().ToArray();
            rows.Count(row => row.GetProperty("boundary").GetString() == "NAT").Should().Be(1,
                "every table case must execute its natural-boundary comparison");
            rows.Count(row => row.GetProperty("boundary").GetString() == "NAK").Should().Be(1,
                "every table case must execute its not-a-knot comparison");
            foreach (var row in rows)
            {
                var name = row.GetProperty("boundary").GetString();
                if (name is "Q" or "RNAK")
                {
                    // Preserve the July historical census without asserting that
                    // the successor method must return a superseded algorithm's values.
                    if (!reconstructed && !row.GetProperty("matches_printed").GetBoolean())
                    {
                        disagreements++;
                    }
                    continue;
                }
                var boundary = name switch
                {
                    "NAT" => SplineBoundary<double>.Natural,
                    "NAK" => SplineBoundary<double>.NotAKnot,
                    "Q" => SplineBoundary<double>.QSpline,
                    "RNAK" => SplineBoundary<double>.RevisedNotAKnot,
                    _ => throw new InvalidOperationException($"Unknown fixture boundary {name}"),
                };
                var spline = new CubicSplineInterpolator<double>(xs, ys, boundary);
                var expected = row.GetProperty("max_error").GetDouble();
                var actual = MaximumError(xs, function, spline);
                var context = $"Table {tableNumber}, {name}, {xs.Length} knots";
                // SciPy locates zeros of s'-f'; this search maximizes |s-f| directly
                // within every interval. Neither is a midpoint-only estimate.
                actual.Should().BeApproximately(expected, 5e-13, context);
                var atOracleExtremum = Math.Abs(spline.Interpolate(row.GetProperty("arg_max").GetDouble())
                    - function(row.GetProperty("arg_max").GetDouble()));
                atOracleExtremum.Should().BeApproximately(expected, 5e-13, context);

                if (reconstructed)
                {
                    row.GetProperty("matches_printed").ValueKind.Should().Be(JsonValueKind.Null,
                        "the author's random knots and seed are not supplied");
                }
                else
                {
                    var printed = row.GetProperty("printed_error").GetDouble();
                    var halfUnit = row.GetProperty("printed_half_unit").GetDouble();
                    if (row.GetProperty("matches_printed").GetBoolean())
                    {
                        actual.Should().BeApproximately(printed, halfUnit + 5e-15, context);
                    }
                    else
                    {
                        // An explicit source inconsistency, not a successful replay
                        // of this printed number. Keep this outside its rounding bin.
                        Math.Abs(actual - printed).Should().BeGreaterThan(halfUnit, context);
                        disagreements++;
                    }
                }
            }
        }

        disagreements.Should().Be(expectedDisagreements);
    }

    private static double MaximumError(double[] knots, Func<double, double> function, CubicSplineInterpolator<double> spline)
    {
        const int Panels = 16;
        const double GoldenFraction = 0.6180339887498948482;
        var maximum = 0.0;
        double Error(double x) => Math.Abs(spline.Interpolate(x) - function(x));
        for (var i = 0; i < knots.Length - 1; i++)
        {
            for (var panel = 0; panel < Panels; panel++)
            {
                var lo = knots[i] + (knots[i + 1] - knots[i]) * panel / Panels;
                var hi = knots[i] + (knots[i + 1] - knots[i]) * (panel + 1) / Panels;
                maximum = Math.Max(maximum, Math.Max(Error(lo), Error(hi)));
                var left = hi - GoldenFraction * (hi - lo);
                var right = lo + GoldenFraction * (hi - lo);
                var leftError = Error(left);
                var rightError = Error(right);
                for (var iteration = 0; iteration < 64; iteration++)
                {
                    if (leftError < rightError)
                    {
                        lo = left;
                        left = right;
                        leftError = rightError;
                        right = lo + GoldenFraction * (hi - lo);
                        rightError = Error(right);
                    }
                    else
                    {
                        hi = right;
                        right = left;
                        rightError = leftError;
                        left = hi - GoldenFraction * (hi - lo);
                        leftError = Error(left);
                    }
                }

                maximum = Math.Max(maximum, Math.Max(leftError, rightError));
            }
        }

        return maximum;
    }
}
