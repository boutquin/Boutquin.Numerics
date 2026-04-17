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

namespace Boutquin.Numerics.Tests.Unit.Statistics.NistStRD;

/// <summary>
/// NIST Statistical Reference Datasets — univariate-summary benchmark suite for
/// <see cref="WelfordMoments"/>. Each test feeds one of nine canonical datasets
/// from <c>https://www.itl.nist.gov/div898/strd/univ/univ.shtml</c> through the
/// single-pass accumulator and compares mean, variance, and standard deviation
/// against NIST-certified values; each dataset is then bisected and merged via
/// Chan-Golub-LeVeque to verify the <see cref="WelfordMoments.Merge"/> round-trip.
/// </summary>
/// <remarks>
/// <para>
/// The suite deliberately includes catastrophic-cancellation stressors — <c>NumAcc4</c>
/// at mean 10<sup>7</sup> with variance 0.1 and <c>Mavro</c> at variance ~6·10<sup>−9</sup>
/// — because naive one-pass implementations fail precisely on these shapes while hand-crafted
/// unit tests don't notice. A regression in the <see cref="WelfordMoments.Merge"/> update
/// step would degrade every downstream estimator; pairing each dataset with a bisect-and-merge
/// check keeps that regression class visible.
/// </para>
/// <para>
/// Tolerance: 14 significant digits of relative agreement against NIST, applied to both the
/// full-stream accumulation and the merged bisection result. Mean and standard deviation come
/// directly from NIST; variance is derived as <c>s²</c> because the univariate suite publishes
/// <c>s</c> but not <c>s²</c>.
/// </para>
/// </remarks>
public sealed class NistWelfordTests
{
    private const double RelativeTolerance = 1e-14;

    public static TheoryData<string> DatasetNames => new()
    {
        "NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4",
        "Lew", "Lottery", "Mavro", "Michelso", "PiDigits",
    };

    [Theory]
    [MemberData(nameof(DatasetNames))]
    public void WelfordMoments_MatchesCertifiedValues(string name)
    {
        var data = Resolve(name);

        var accumulator = new WelfordMoments();
        foreach (var v in data.Values)
        {
            accumulator.Add(v);
        }

        AssertMatches(accumulator, data);
    }

    [Theory]
    [MemberData(nameof(DatasetNames))]
    public void WelfordMoments_MergeRoundTripMatchesSerial(string name)
    {
        var data = Resolve(name);
        var split = data.Values.Length / 2;

        // Bisect, accumulate each half independently, merge, and assert the combined moments
        // match the whole-stream accumulator to the full NIST precision bar.
        var left = new WelfordMoments();
        for (var i = 0; i < split; i++)
        {
            left.Add(data.Values[i]);
        }

        var right = new WelfordMoments();
        for (var i = split; i < data.Values.Length; i++)
        {
            right.Add(data.Values[i]);
        }

        left.Merge(right);

        AssertMatches(left, data);
    }

    private static NistUnivariateData.Dataset Resolve(string name) => name switch
    {
        "NumAcc1" => NistUnivariateData.NumAcc1(),
        "NumAcc2" => NistUnivariateData.NumAcc2(),
        "NumAcc3" => NistUnivariateData.NumAcc3(),
        "NumAcc4" => NistUnivariateData.NumAcc4(),
        "Lew" => NistUnivariateData.Lew(),
        "Lottery" => NistUnivariateData.Lottery(),
        "Mavro" => NistUnivariateData.Mavro(),
        "Michelso" => NistUnivariateData.Michelso(),
        "PiDigits" => NistUnivariateData.PiDigits(),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown NIST univariate dataset."),
    };

    private static void AssertMatches(WelfordMoments accumulator, NistUnivariateData.Dataset data)
    {
        var mean = (double)accumulator.Mean;
        var variance = (double)accumulator.Variance;
        var stdDev = (double)accumulator.StdDev;

        RelativeAgreement(mean, data.CertifiedMean)
            .Should().BeLessThanOrEqualTo(RelativeTolerance,
                because: $"{data.Name}: mean {mean:G17} must match NIST certified {data.CertifiedMean:G17} to 14 digits");

        RelativeAgreement(variance, data.CertifiedVariance)
            .Should().BeLessThanOrEqualTo(RelativeTolerance,
                because: $"{data.Name}: variance {variance:G17} must match NIST certified {data.CertifiedVariance:G17} to 14 digits");

        RelativeAgreement(stdDev, data.CertifiedStdDev)
            .Should().BeLessThanOrEqualTo(RelativeTolerance,
                because: $"{data.Name}: standard deviation {stdDev:G17} must match NIST certified {data.CertifiedStdDev:G17} to 14 digits");
    }

    /// <summary>
    /// Relative agreement. Pure relative error — all NIST-certified targets in this
    /// suite are strictly positive (variances, standard deviations) or away from zero
    /// (means), so there is no denominator-underflow case to guard.
    /// </summary>
    private static double RelativeAgreement(double actual, double expected)
    {
        return Math.Abs(actual - expected) / Math.Abs(expected);
    }
}
