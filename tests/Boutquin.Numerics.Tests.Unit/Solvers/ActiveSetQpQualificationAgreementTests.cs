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
using System.Numerics;
using System.Text;

using HeldOutOutcome = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutOutcome;
using HeldOutProblem = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutProblem;
using HeldOutResult = Boutquin.Numerics.Tests.Unit.Solvers.ActiveSetQpCorpusTests.HeldOutResult;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// AC-3.1 and AC-4.1 (amendment #11): the corpus-membership <c>κ_eq</c> is measured after the covariance-only
/// equilibration <c>QpScaling&lt;T&gt;.EquilibrateCovariance</c>, the block the solver's conditioning guard measures, so
/// qualification and refusal cannot disagree. Seeded in-test censuses of decimal problems whose bounds or means carry
/// 28 places (the corpus is at its 2,300-case cap), through the production <c>SolveMinVariance</c> and
/// <c>SolveMeanVariance</c>, each classified by the shared oracle of <see cref="ActiveSetQpCorpusTests"/>: no qualified
/// problem is refused, no result is a wrong vector, and every member whose interior deciding block crosses the margin
/// is refused naming stationarity.
/// Falsifiability: each census holds at least 8 members whose bound- or mean-limited exponents (the definition before
/// the amendment) give <c>48·n·u·κ_eq &lt; 1</c> while the covariance-only exponents give <c>≥ 1</c>, with the optimum
/// strictly inside the box (<c>B = F*</c>); an oracle that restored the earlier definition qualifies those members and
/// the guard refuses them, so the census fails.
/// </summary>
[Trait("Tier", "Unit")]
public sealed class ActiveSetQpQualificationAgreementTests
{
    // Members per census. Achieved at the committed seeds: min variance (seed 2026102011) 40 of 64 qualified and 22
    // members between the two definitions; mean variance (seed 2026102012) 44 of 64 qualified and 18 between them.
    // The members whose deciding block crosses the margin (24 and 20) are all refused naming stationarity.
    private const int Members = 64;

    // At least this many members per entry point lie between the two definitions (spec AC-3.1 note, amendment #11).
    private const int MinimumDisagreeing = 8;

    [Fact]
    public void ActiveSetQpMinVariance_TwentyEightPlaceBoundCensus_QualifiedOnTheCovarianceOnlyScaling_NoQualifiedRefusalNoWrongVector()
        => AssertAgreementCensus(meanVariance: false, seed: 2026102011, minimumQualified: 32, expectedCrossers: 24);

    [Fact]
    public void ActiveSetQpMeanVariance_TwentyEightPlaceBoundAndMeanCensus_QualifiedOnTheCovarianceOnlyScaling_NoQualifiedRefusalNoWrongVector()
        => AssertAgreementCensus(meanVariance: true, seed: 2026102012, minimumQualified: 32, expectedCrossers: 20);

    private static void AssertAgreementCensus(bool meanVariance, int seed, int minimumQualified, int expectedCrossers)
    {
        var rng = new System.Random(seed);
        var results = new List<HeldOutResult>();
        var disagreeing = 0;
        var problems = new StringBuilder();
        for (var trial = 0; trial < Members; trial++)
        {
            var problem = AgreementProblem(rng, trial, meanVariance);
            var margins = ActiveSetQpCorpusTests.MarginsUnderBothDefinitions(problem);
            var result = ActiveSetQpCorpusTests.SolveHeldOut(problem);
            results.Add(result);
            if (margins.Interior && margins.EarlierQualified && !margins.CovarianceOnlyQualified)
            {
                disagreeing++;
            }

            if (result.Qualified != margins.CovarianceOnlyQualified)
            {
                problems.Append(problem.Name).Append(": the shared oracle qualified=").Append(result.Qualified)
                    .Append(", the covariance-only definition (48·n·u·κ_eq = ").Append(margins.CovarianceOnlyMargin.ToString("G4", CultureInfo.InvariantCulture))
                    .Append(") qualified=").Append(margins.CovarianceOnlyQualified).Append(" | ");
            }

            if (!result.Qualified && margins.Interior && result.Outcome == HeldOutOutcome.Refused && !result.Detail.Contains("stationarity", StringComparison.Ordinal))
            {
                problems.Append(problem.Name).Append(": a refusal-permitted interior problem was refused without naming stationarity: ").Append(result.Detail).Append(" | ");
            }
        }

        ActiveSetQpCorpusTests.AssertCensus(results, minimumQualified);
        Assert.True(problems.Length == 0, problems.ToString());
        ActiveSetQpCorpusTests.AssertMarginCrossersAreRefusedNamingStationarity(results, expectedCrossers);
        Assert.True(
            disagreeing >= MinimumDisagreeing,
            "the census must hold at least " + MinimumDisagreeing + " interior members that the earlier definition qualifies and the covariance-only definition does not, found "
            + disagreeing + " (" + results.Count(r => r.Qualified) + " of " + results.Count + " qualified)");
    }

    /// <summary>
    /// A two-asset decimal problem built around the radix-power boundary of the diagonal: <c>Σ₀₀</c> just below and
    /// <c>Σ₁₁</c> at or just above <c>10⁻²</c> (or <c>1</c>), so the covariance-only exponents differ by one, with the
    /// assets randomly swapped. The off-diagonal is <c>−⌊√(Σ₀₀Σ₁₁(1 − t))⌋</c> on the 28-place grid (27 places for the
    /// mean-limited shape, so that the covariance itself scales exactly), with <c>t = m·10⁻ᵏ</c>, <c>k ∈ [21, 26]</c>,
    /// putting <c>48·n·u·κ_eq</c> below, between and above the two definitions' margins. Bound-limited shape (both entry
    /// points): at least one bound carries 28 places, so under the earlier definition it holds the negative exponent at
    /// 0. Mean-limited shape (mean-variance only): both means carry 28 places, so under the earlier definition they hold
    /// the positive exponent at 0.
    /// </summary>
    private static HeldOutProblem AgreementProblem(System.Random rng, int trial, bool meanVariance)
    {
        var meanLimited = meanVariance && rng.Next(2) == 1;
        var places = meanLimited ? 6 : 8;
        var a = new BigInteger(900_000 + rng.Next(99_000));
        var d = new BigInteger(1_000_000 + rng.Next(100_000));
        var k = rng.Next(21, 27);
        var m = rng.Next(1, 10);
        var offPlaces = meanLimited ? 27 : 28;
        var radicand = a * d * (BigInteger.Pow(10, k) - m) * BigInteger.Pow(10, 2 * offPlaces) / (BigInteger.Pow(10, 2 * places) * BigInteger.Pow(10, k));
        var off = "-" + FixedPoint(IntegerSquareRoot(radicand), offPlaces);
        var (first, second) = rng.Next(2) == 0 ? (a, d) : (d, a);
        string[][] cov = [[FixedPoint(first, places), off], [off, FixedPoint(second, places)]];
        var name = (meanLimited ? "mean" : "bound") + "-limited-" + trial + "-t" + m + "e-" + k;
        if (!meanVariance)
        {
            var (lb, ub) = rng.Next(3) switch
            {
                0 => (Signed28(rng, negativeFirstDigits: 5, positiveFirstDigits: 4), "1"),
                1 => ("0", Places28(rng, 6, 9)),
                _ => (Signed28(rng, negativeFirstDigits: 5, positiveFirstDigits: 4), Places28(rng, 6, 9)),
            };
            return new HeldOutProblem(name, "decimal", cov, lb, ub);
        }

        var lambda = new[] { "0.5", "1", "2" }[rng.Next(3)];
        if (meanLimited)
        {
            var lbMean = new[] { "0", "-1", "-" + Places28(rng, 0, 4) }[rng.Next(3)];
            var ubMean = new[] { "1", "2" }[rng.Next(2)];
            return new HeldOutProblem(name, "decimal", cov, lbMean, ubMean, [Signed(rng, Places28(rng, 0, 1)), Signed(rng, Places28(rng, 0, 1))], lambda);
        }

        var (lower, upper) = rng.Next(3) switch
        {
            0 => (Signed28(rng, negativeFirstDigits: 5, positiveFirstDigits: 3), "1"),
            1 => ("0", Places28(rng, 7, 9)),
            _ => (Signed28(rng, negativeFirstDigits: 5, positiveFirstDigits: 3), Places28(rng, 7, 9)),
        };
        string[] means = [Signed(rng, SmallPlaces28(rng)), Signed(rng, SmallPlaces28(rng))];
        return new HeldOutProblem(name, "decimal", cov, lower, upper, means, lambda);
    }

    // A value with exactly 28 decimal places in [first/10, (last + 1)/10): the first digit drawn from [first, last],
    // 26 random digits and a non-zero last digit.
    private static string Places28(System.Random rng, int first, int last)
    {
        var sb = new StringBuilder("0.");
        sb.Append((char)('0' + rng.Next(first, last + 1)));
        for (var i = 0; i < 26; i++)
        {
            sb.Append((char)('0' + rng.Next(10)));
        }

        sb.Append((char)('1' + rng.Next(9)));
        return sb.ToString();
    }

    // A value with exactly 28 decimal places below 0.002 in magnitude.
    private static string SmallPlaces28(System.Random rng)
    {
        var sb = new StringBuilder("0.00");
        sb.Append((char)('0' + rng.Next(2)));
        for (var i = 0; i < 24; i++)
        {
            sb.Append((char)('0' + rng.Next(10)));
        }

        sb.Append((char)('1' + rng.Next(9)));
        return sb.ToString();
    }

    // A 28-place value in (−negativeFirstDigits/10, positiveFirstDigits/10).
    private static string Signed28(System.Random rng, int negativeFirstDigits, int positiveFirstDigits) =>
        rng.Next(negativeFirstDigits + positiveFirstDigits) < negativeFirstDigits
            ? "-" + Places28(rng, 0, negativeFirstDigits - 1)
            : Places28(rng, 0, positiveFirstDigits - 1);

    private static string Signed(System.Random rng, string magnitude) => rng.Next(2) == 0 ? magnitude : "-" + magnitude;

    private static string FixedPoint(BigInteger value, int places)
    {
        var digits = value.ToString(CultureInfo.InvariantCulture).PadLeft(places + 1, '0');
        return digits[..^places] + "." + digits[^places..];
    }

    private static BigInteger IntegerSquareRoot(BigInteger n)
    {
        if (n.IsZero)
        {
            return n;
        }

        var x = BigInteger.One << (int)((n.GetBitLength() + 1) / 2);
        while (true)
        {
            var y = (x + (n / x)) / 2;
            if (y >= x)
            {
                return x;
            }

            x = y;
        }
    }
}
