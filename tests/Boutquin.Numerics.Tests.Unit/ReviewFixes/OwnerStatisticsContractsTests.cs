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

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

public sealed class OwnerStatisticsContractsTests
{
    // The owner generator (replay-v2.py) always emits exactly this many nwCases in a supplied fixture.
    internal const int GeneratorNwCaseCount = 12;

    private delegate (double[,] Covariance, int Lags) Nw(double[,] instruments, ReadOnlySpan<double> residuals, ReadOnlySpan<double> weights, int? pilotLags);
    private delegate (decimal[] Bonferroni, decimal[] Holm, decimal[] Bhy) Panel(ReadOnlySpan<decimal> pValues);
    private delegate DsrResult<double> Dsr(double observedSharpe, int numTrials, double trialSharpeMean, double trialSharpeVariance, double backTestYears, double skewness, double kurtosis, int tradingDaysPerYear);

    [Fact]
    public void NeweyWest_CompleteVarAndRecolour_RationalOracle()
    {
        var method = typeof(NeweyWestVariance<double>).GetMethod("LongRunCovariance");
        Assert.NotNull(method);
        var run = method.CreateDelegate<Nw>();
        var z = new double[24, 2];
        for (var i = 0; i < 24; i++)
        {
            z[i, 0] = (i * 7 % 11) - 5;
            z[i, 1] = ((i * i * 3 + 2 * i) % 13) - 6;
        }
        var result = run(z, Enumerable.Repeat(1.0, 24).ToArray(), [0, 1], null);
        Assert.Equal(1, result.Lags);
        Assert.Equal(3.3582664754829827, result.Covariance[0, 0], 11);
        Assert.Equal(0.6491244480639407, result.Covariance[0, 1], 11);
        Assert.Equal(0.6491244480639407, result.Covariance[1, 0], 11);
        Assert.Equal(18.934860544671157, result.Covariance[1, 1], 11);
    }

    [Fact]
    public void Haircut_OrderedPanel_PublishedExampleInOriginalOrder()
    {
        var method = typeof(HaircutSharpe<decimal>).GetMethod("AdjustPValues");
        Assert.NotNull(method);
        var result = method.CreateDelegate<Panel>()([.06m, .009m, .0135m, .005m, .045m, .0128m]);
        Assert.Equal([.36m, .054m, .081m, .03m, .27m, .0768m], result.Bonferroni);
        Assert.Equal([.09m, .045m, .0512m, .03m, .09m, .0512m], result.Holm);
        Assert.Equal([.06m, .0496125m, .0496125m, .0496125m, .06m, .0496125m], result.Bhy);
    }

    [Fact]
    public void Dsr_TrialMean_IndependentNormalOracle()
    {
        var method = typeof(DeflatedSharpeRatio<double>).GetMethod("ComputeWithTrialMoments");
        Assert.NotNull(method);
        var run = method.CreateDelegate<Dsr>();
        var actual = run(2.2, 37, .7, .36, 4, -.8, 5, 12);
        Assert.Equal(1.9951011182425171, actual.ExpectedMaxSharpe, 8);
        Assert.Equal(.20489888175748305, actual.DeflatedSharpe, 8);
        Assert.Equal(.3846438816005121, actual.PValue, 8);
        var single = run(2.2, 1, .7, .36, 4, -.8, 5, 12);
        Assert.Equal(.7, single.ExpectedMaxSharpe, 12);
    }
    [Fact]
    public void HeldOut_IndependentFractionAndNormalOracles()
    {
        var path = Environment.GetEnvironmentVariable("OWNER_STATISTICS_FIXTURE");
        if (path is null)
        {
            NeweyWest_CompleteVarAndRecolour_RationalOracle();
            Haircut_OrderedPanel_PublishedExampleInOriginalOrder();
            Dsr_TrialMean_IndependentNormalOracle();
            return;
        }

        VerifyFixture(File.ReadAllText(path));
    }

    internal static void VerifyFixture(string json)
    {
        using var document = System.Text.Json.JsonDocument.Parse(json);
        var root = document.RootElement;
        static double[] Vector(System.Text.Json.JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var rows = root.GetProperty("z").EnumerateArray().Select(Vector).ToArray();
        var z = new double[rows.Length, 2];
        for (var i = 0; i < rows.Length; i++)
        {
            z[i, 0] = rows[i][0];
            z[i, 1] = rows[i][1];
        }

        var actual = NeweyWestVariance<double>.LongRunCovariance(z, Vector(root.GetProperty("u")), Vector(root.GetProperty("w")));
        Assert.Equal(root.GetProperty("m").GetInt32(), actual.Lags);
        var expected = root.GetProperty("c").EnumerateArray().Select(Vector).ToArray();
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                Assert.Equal(expected[i][j], actual.Covariance[i, j], 9);
            }
        }

        // An emptied or truncated family would otherwise loop zero or fewer times and pass vacuously.
        var cases = root.GetProperty("nwCases").EnumerateArray().ToArray();
        RequireCount("nwCases", GeneratorNwCaseCount, cases.Length);
        foreach (var item in cases)
        {
            var hs = item.GetProperty("h").EnumerateArray().Select(Vector).ToArray();
            var matrix = new double[hs.Length, 2];
            for (var i = 0; i < hs.Length; i++)
            {
                matrix[i, 0] = hs[i][0];
                matrix[i, 1] = hs[i][1];
            }

            var result = NeweyWestVariance<double>.LongRunCovariance(matrix, Enumerable.Repeat(1.0, hs.Length).ToArray(), Vector(item.GetProperty("w")), item.GetProperty("pilot").GetInt32());
            Assert.Equal(item.GetProperty("m").GetInt32(), result.Lags);
            var want = item.GetProperty("c").EnumerateArray().Select(Vector).ToArray();
            for (var i = 0; i < 2; i++)
            {
                for (var j = 0; j < 2; j++)
                {
                    Assert.Equal(want[i][j], result.Covariance[i, j], 9);
                }
            }
        }

        Assert.Equal(root.GetProperty("meanVariance").GetDouble(), NeweyWestVariance<double>.MeanVarianceAutomatic(Vector(root.GetProperty("series"))), 9);
        var pValues = Vector(root.GetProperty("p"));
        RequireCount("p", 1, pValues.Length, atLeast: true);
        var panel = HaircutSharpe<double>.AdjustPValues(pValues);
        foreach (var (key, values) in new[] { ("bonf", panel.Bonferroni), ("holm", panel.Holm), ("bhy", panel.Bhy) })
        {
            var want = Vector(root.GetProperty(key));
            RequireCount(key, pValues.Length, want.Length);
            for (var i = 0; i < want.Length; i++)
            {
                Assert.Equal(want[i], values[i], 12);
            }
        }

        var input = Vector(root.GetProperty("dsrInput"));
        var dsr = DeflatedSharpeRatio<double>.ComputeWithTrialMoments(input[0], (int)input[1], input[2], input[3], input[4], input[5], input[6], (int)input[7]);
        var wantedDsr = Vector(root.GetProperty("dsrExpected"));
        Assert.Equal(wantedDsr[0], dsr.ExpectedMaxSharpe, 8);
        Assert.Equal(wantedDsr[1], dsr.PValue, 8);
    }

    private static void RequireCount(string family, int required, int actual, bool atLeast = false) =>
        Assert.True(
            atLeast ? actual >= required : actual == required,
            $"held-out family '{family}' has {actual} entries but {(atLeast ? "at least " : string.Empty)}{required} are required; refusing a vacuous comparison");

    [Fact]
    public void NewSurfaces_RejectInvalidInputs_AndPreserveCompatibility()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => HaircutSharpe<double>.AdjustPValues([double.NaN]));
        Assert.Throws<ArgumentOutOfRangeException>(() => HaircutSharpe<double>.AdjustPValues([-.1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => DeflatedSharpeRatio<double>.ComputeWithTrialMoments(1, 5, double.NaN, 1, 3, 0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => DeflatedSharpeRatio<double>.ComputeWithTrialMoments(1, 5, 0, -1, 3, 0, 3));
        Assert.Throws<ArgumentException>(() => NeweyWestVariance<double>.LongRunCovariance(new double[1, 1], [1], [1]));
        Assert.Throws<InvalidOperationException>(() => NeweyWestVariance<double>.LongRunCovariance(new double[3, 1], [1, 1, 1], [1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => NeweyWestVariance<double>.MeanVarianceAutomatic([1, double.NaN]));
        Assert.Equal(0, NeweyWestVariance<double>.MeanVarianceAutomatic([4, 4, 4]));
        var zeroMean = DeflatedSharpeRatio<double>.ComputeWithTrialMoments(2, 40, 0, .5, 5, -1, 4, 12);
        Assert.Equal(DeflatedSharpeRatio<double>.ComputeWithTrialVariance(2, 40, .5, 5, -1, 4, 12), zeroMean);
        var panel = HaircutSharpe<double>.ComputePanel([.6, -.8, 1.1], 5, 12);
        var p = new[] { .6, -.8, 1.1 }.Select(x => HaircutSharpe<double>.Compute(x, 1, 5, 12).BonferroniPValue).ToArray();
        var adjusted = HaircutSharpe<double>.AdjustPValues(p);
        for (var i = 0; i < panel.Length; i++)
        {
            Assert.Equal(adjusted.Holm[i], panel[i].HolmPValue);
            Assert.Equal(adjusted.Bhy[i], panel[i].BhyPValue);
            Assert.Equal(HaircutSharpe<double>.Compute(panel[i].ObservedSharpe, 3, 5, 12).HaircutSharpe, panel[i].HaircutSharpe, 12);
        }
    }
}
