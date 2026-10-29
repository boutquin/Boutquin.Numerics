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

using System.Text.Json.Nodes;
using Boutquin.Numerics.Tests.Unit.Statistics;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Pins that the externally supplied held-out fixtures cannot pass vacuously: a fixture whose
/// expected families are emptied or truncated must make the verifier fail, rather than loop zero times.
/// </summary>
public sealed class HeldoutOracleVacuityTests
{
    private const string RefusalMarker = "refusing a vacuous comparison";

    public static TheoryData<string> CovarianceFamilies => new()
    {
        "nercomeExpected",
        "nercomeFixedExpected",
        "oasExpected",
        "glassoCovariance",
        "glassoPrecision",
    };

    public static TheoryData<string> StatisticsAdjustedFamilies => new() { "bonf", "holm", "bhy" };

    [Fact]
    public void Covariance_UnmutatedFixture_Passes()
    {
        OwnerCovarianceHeldoutTests.VerifyFixture(OwnerCovarianceHeldoutTests.DefaultJson);
    }

    [Theory]
    [MemberData(nameof(CovarianceFamilies))]
    public void Covariance_EmptiedExpectedFamily_IsRefused(string family)
    {
        var node = JsonNode.Parse(OwnerCovarianceHeldoutTests.DefaultJson)!;
        node[family] = new JsonArray();

        var ex = Assert.ThrowsAny<Exception>(() => OwnerCovarianceHeldoutTests.VerifyFixture(node.ToJsonString()));
        Assert.Contains(RefusalMarker, ex.Message);
        Assert.Contains(family, ex.Message);
    }

    [Theory]
    [MemberData(nameof(CovarianceFamilies))]
    public void Covariance_RowTruncatedExpectedFamily_IsRefused(string family)
    {
        var node = JsonNode.Parse(OwnerCovarianceHeldoutTests.DefaultJson)!;
        var values = node[family]!.AsArray();
        var width = family.StartsWith("oas", StringComparison.Ordinal) ? 3 : 2;
        for (var i = 0; i < width; i++)
        {
            values.RemoveAt(values.Count - 1);
        }

        var ex = Assert.ThrowsAny<Exception>(() => OwnerCovarianceHeldoutTests.VerifyFixture(node.ToJsonString()));
        Assert.Contains(RefusalMarker, ex.Message);
        Assert.Contains(family, ex.Message);
    }

    [Fact]
    public void Covariance_WrongExpectedValue_StillFails_ProvingTheFixtureIsConsumed()
    {
        var node = JsonNode.Parse(OwnerCovarianceHeldoutTests.DefaultJson)!;
        node["oasExpected"]![0] = 22.40740740740741 + 0.5;

        var ex = Assert.ThrowsAny<Exception>(() => OwnerCovarianceHeldoutTests.VerifyFixture(node.ToJsonString()));
        Assert.DoesNotContain(RefusalMarker, ex.Message);
    }

    [Fact]
    public void Statistics_UnmutatedFixture_Passes()
    {
        OwnerStatisticsContractsTests.VerifyFixture(StatisticsBaseline().ToJsonString());
    }

    [Fact]
    public void Statistics_EmptiedNwCases_IsRefused()
    {
        var node = StatisticsBaseline();
        node["nwCases"] = new JsonArray();

        var ex = Assert.ThrowsAny<Exception>(() => OwnerStatisticsContractsTests.VerifyFixture(node.ToJsonString()));
        Assert.Contains(RefusalMarker, ex.Message);
        Assert.Contains("nwCases", ex.Message);
    }

    [Fact]
    public void Statistics_DroppedLastNwCase_IsRefused()
    {
        var node = StatisticsBaseline();
        var cases = node["nwCases"]!.AsArray();
        cases.RemoveAt(cases.Count - 1);

        var ex = Assert.ThrowsAny<Exception>(() => OwnerStatisticsContractsTests.VerifyFixture(node.ToJsonString()));
        Assert.Contains(RefusalMarker, ex.Message);
        Assert.Contains("nwCases", ex.Message);
    }

    [Theory]
    [MemberData(nameof(StatisticsAdjustedFamilies))]
    public void Statistics_EmptiedAdjustedFamily_IsRefused(string family)
    {
        var node = StatisticsBaseline();
        node[family] = new JsonArray();

        var ex = Assert.ThrowsAny<Exception>(() => OwnerStatisticsContractsTests.VerifyFixture(node.ToJsonString()));
        Assert.Contains(RefusalMarker, ex.Message);
        Assert.Contains(family, ex.Message);
    }

    [Theory]
    [MemberData(nameof(StatisticsAdjustedFamilies))]
    public void Statistics_DroppedLastAdjustedPValue_IsRefused(string family)
    {
        var node = StatisticsBaseline();
        var values = node[family]!.AsArray();
        values.RemoveAt(values.Count - 1);

        var ex = Assert.ThrowsAny<Exception>(() => OwnerStatisticsContractsTests.VerifyFixture(node.ToJsonString()));
        Assert.Contains(RefusalMarker, ex.Message);
        Assert.Contains(family, ex.Message);
    }

    [Fact]
    public void Statistics_WrongAdjustedValue_StillFails_ProvingTheFixtureIsConsumed()
    {
        var node = StatisticsBaseline();
        node["bonf"]![0] = 0.37;

        var ex = Assert.ThrowsAny<Exception>(() => OwnerStatisticsContractsTests.VerifyFixture(node.ToJsonString()));
        Assert.DoesNotContain(RefusalMarker, ex.Message);
    }

    /// <summary>
    /// A complete, valid fixture in the owner generator's shape, assembled from the in-code oracles of
    /// <see cref="OwnerStatisticsContractsTests"/> (rational-oracle Newey-West values, the published
    /// ordered-panel p-value example, and the independent normal-oracle DSR values).
    /// </summary>
    private static JsonNode StatisticsBaseline()
    {
        var z = new JsonArray();
        for (var i = 0; i < 24; i++)
        {
            z.Add(new JsonArray((i * 7 % 11) - 5, ((i * i * 3 + 2 * i) % 13) - 6));
        }

        JsonArray Ones(int n)
        {
            var a = new JsonArray();
            for (var i = 0; i < n; i++)
            {
                a.Add(1.0);
            }

            return a;
        }

        JsonArray Cov() => new(new JsonArray(3.3582664754829827, 0.6491244480639407), new JsonArray(0.6491244480639407, 18.934860544671157));

        var nwCases = new JsonArray();
        for (var k = 0; k < OwnerStatisticsContractsTests.GeneratorNwCaseCount; k++)
        {
            nwCases.Add(new JsonObject
            {
                ["h"] = z.DeepClone(),
                ["w"] = new JsonArray(0.0, 1.0),
                ["pilot"] = 2,
                ["m"] = 1,
                ["c"] = Cov(),
            });
        }

        return new JsonObject
        {
            ["nwCases"] = nwCases,
            ["z"] = z,
            ["u"] = Ones(24),
            ["w"] = new JsonArray(0.0, 1.0),
            ["m"] = 1,
            ["c"] = Cov(),
            ["series"] = new JsonArray(4.0, 4.0, 4.0),
            ["meanVariance"] = 0.0,
            ["p"] = new JsonArray(.06, .009, .0135, .005, .045, .0128),
            ["bonf"] = new JsonArray(.36, .054, .081, .03, .27, .0768),
            ["holm"] = new JsonArray(.09, .045, .0512, .03, .09, .0512),
            ["bhy"] = new JsonArray(.06, .0496125, .0496125, .0496125, .06, .0496125),
            ["dsrInput"] = new JsonArray(2.2, 37.0, .7, .36, 4.0, -.8, 5.0, 12.0),
            ["dsrExpected"] = new JsonArray(1.9951011182425171, .3846438816005121),
        };
    }
}
