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
using System.Text.RegularExpressions;

using Boutquin.Numerics.MonteCarlo;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.ReviewFixes;

/// <summary>
/// Executes the documented pairing of <see cref="PolitisWhiteBlockLength"/> with
/// <see cref="StationaryBootstrapResampler"/>. The estimator has no lower floor
/// (estimates below one occur on serially independent data) while the resampler requires a
/// mean block length of at least one, so the README snippet must floor the estimate before
/// constructing the resampler. The snippet text is read from the shipped README and
/// evaluated against production on seeded white-noise returns.
/// </summary>
public sealed class Arena414074R2DocumentationTests
{
    private const int ObservationCount = 250;
    private const int SeedCount = 60;

    [Fact]
    public void ReadmeStationarySnippet_Pairing_RunsOnEverySeriallyIndependentSeries()
    {
        var block = Arena414074R1DocumentationClaimTests.ReadmeCodeBlock("Monte Carlo Bootstrap");
        var toMeanBlockLength = ReadmeMeanBlockLength(block);

        var belowOne = 0;
        for (var seed = 1; seed <= SeedCount; seed++)
        {
            var returns = WhiteNoiseReturns(seed);
            var estimate = PolitisWhiteBlockLength.EstimateDouble(Array.ConvertAll(returns, v => (double)v));
            belowOne += estimate < 1.0 ? 1 : 0;
            var meanBlockLength = toMeanBlockLength(estimate);

            var pairing = () => StationaryBootstrapResampler.FromSeed(meanBlockLength, seed: 42).Resample(returns);

            pairing.Should().NotThrow(
                $"the README passes the Politis-White estimate {estimate} (seed {seed}) to the stationary resampler as {meanBlockLength}");
        }

        belowOne.Should().BeGreaterThan(0,
            "the draws must reach the regime where the estimator returns less than one, or the pairing is never exercised");
    }

    [Fact]
    public void PolitisWhiteBlockLength_HasNoLowerFloor_AndStationaryResamplerRequiresAMeanOfOne()
    {
        var below = Enumerable.Range(1, SeedCount)
            .Select(seed => PolitisWhiteBlockLength.EstimateDouble(Array.ConvertAll(WhiteNoiseReturns(seed), v => (double)v)))
            .Where(estimate => estimate < 1.0)
            .ToArray();

        below.Should().NotBeEmpty("estimates below one occur on serially independent returns (documented: no lower floor)");
        below.Should().OnlyContain(estimate => estimate > 0.0 && double.IsFinite(estimate));

        var rejected = () => StationaryBootstrapResampler.FromSeed(below[0], seed: 42);
        rejected.Should().Throw<ArgumentOutOfRangeException>().WithParameterName("meanBlockLength");

        StationaryBootstrapResampler.FromSeed(Math.Max(1.0, below[0]), seed: 42).MeanBlockLength
            .Should().Be(1.0);
    }

    [Fact]
    public void PolitisWhiteBlockLengthRemarks_StateTheFloorTheStationaryResamplerNeeds()
    {
        var source = Arena414074R1DocumentationClaimTests.ReadRepositoryText("src/Boutquin.Numerics/MonteCarlo/PolitisWhiteBlockLength.cs");
        var documentation = string.Join(
            ' ',
            source.Split('\n')
                .Select(line => line.Trim())
                .Where(line => line.StartsWith("///", StringComparison.Ordinal))
                .Select(line => line[3..].Trim()));
        documentation = Regex.Replace(documentation, @"\s+", " ");

        documentation.Should().NotContain("consume this double-valued block length directly",
            because: "the stationary resampler rejects the estimates below one that the selector documents");
        documentation.Should().Contain("Math.Max(1.0, estimate)",
            because: "the remarks must tell the caller how to satisfy the resampler's minimum mean block length");
    }

    // Reads how the README turns the estimator's value into the resampler's first argument:
    // either the raw estimate, or Math.Max(<literal>, <estimate variable>).
    private static Func<double, double> ReadmeMeanBlockLength(string block)
    {
        var argument = Regex.Match(block, @"StationaryBootstrapResampler\.FromSeed\(\s*(?<argument>[A-Za-z_]\w*)\s*,");
        argument.Success.Should().BeTrue("the README constructs the stationary resampler from a named block length");
        var name = argument.Groups["argument"].Value;

        var declaration = Regex.Match(block, $@"double\s+{Regex.Escape(name)}\s*=\s*(?<expression>[^;]+);");
        declaration.Success.Should().BeTrue($"the README declares the double '{name}' it passes to the resampler");
        var expression = declaration.Groups["expression"].Value.Trim();

        var floor = Regex.Match(expression, @"^Math\.Max\(\s*(?<floor>\d+(?:\.\d+)?)\s*,\s*(?<estimate>[A-Za-z_]\w*)\s*\)$");
        if (floor.Success)
        {
            var estimate = floor.Groups["estimate"].Value;
            Regex.IsMatch(block, $@"double\s+{Regex.Escape(estimate)}\s*=\s*PolitisWhiteBlockLength\.EstimateDouble\(")
                .Should().BeTrue($"'{estimate}' must be the raw Politis-White estimate that the README floors");
            var minimum = double.Parse(floor.Groups["floor"].Value, CultureInfo.InvariantCulture);
            return value => Math.Max(minimum, value);
        }

        expression.Should().StartWith("PolitisWhiteBlockLength.EstimateDouble(",
            because: "an unfloored pairing passes the raw estimate straight to the resampler");
        return value => value;
    }

    // Serially independent returns with the scale of daily equity returns (seeded, so the
    // sub-one estimates the test relies on are reproducible).
    private static decimal[] WhiteNoiseReturns(int seed)
    {
        var random = new System.Random(seed);
        var returns = new decimal[ObservationCount];
        for (var i = 0; i < returns.Length; i++)
        {
            var gaussian = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble()))
                * Math.Cos(2.0 * Math.PI * random.NextDouble());
            returns[i] = decimal.Round((decimal)(0.0004 + (0.02 * gaussian)), 6);
        }

        return returns;
    }
}
