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

namespace Boutquin.Numerics.Tests.Verification;

public sealed class DsrVerificationTests : CrossLanguageVerificationBase
{
    [Fact]
    public void DeflatedSharpe_Scalar_MatchesReferenceImplementation()
    {
        using var doc = LoadVector("dsr");
        var returns = GetDecimalArray(doc.RootElement.GetProperty("returns"));
        var numTrials = doc.RootElement.GetProperty("num_trials").GetInt32();
        var tradingDays = doc.RootElement.GetProperty("trading_days_per_year").GetInt32();
        var expectedScalar = doc.RootElement.GetProperty("deflated_sharpe_scalar_100_trials").GetDouble();
        var expectedMaxSharpe = doc.RootElement.GetProperty("expected_max_sharpe_100_trials").GetDouble();

        var result = DeflatedSharpeRatio.ComputeFromReturns(returns, numTrials, tradingDays);

        // The extreme-value expected-max is deterministic; scalar deflation is tight.
        AssertScalarWithin((double)result.ExpectedMaxSharpe, expectedMaxSharpe, PrecisionNumeric, "E[max SR]");
        AssertScalarWithin((double)result.DeflatedSharpe, expectedScalar, PrecisionStatistical, "DSR scalar");
        result.PValue.Should().BeInRange(0m, 1m);
    }

    [Fact]
    public void MinimumTrackRecordLength_MatchesReferenceImplementation()
    {
        // Bailey-López de Prado 2012 MinTRL formula, cross-checked against the
        // scipy.stats.norm.ppf-based reference in tests/Verification/generate_dsr_vectors.py
        // (function `minimum_track_record_length`). Inputs come from dsr.json rather than
        // being recomputed from `returns` so the moment-convention mismatch between Python
        // (scipy.stats.skew/kurtosis, Pearson raw) and the C# moment path is factored out
        // of this test — we compare the MinTRL arithmetic only. Moment parity is already
        // covered by DeflatedSharpe_Scalar_MatchesReferenceImplementation above.
        using var doc = LoadVector("dsr");
        var observedSharpe = doc.RootElement.GetProperty("sharpe").GetDouble();
        var skew = doc.RootElement.GetProperty("skew").GetDouble();
        var kurtosis = doc.RootElement.GetProperty("kurtosis_pearson").GetDouble();
        var tradingDays = doc.RootElement.GetProperty("trading_days_per_year").GetInt32();

        // Python reference reports T* in observation-time units (days). C# reports T* in
        // years (divides by tradingDaysPerYear internally). Multiply C# by tradingDays
        // to compare on the same axis.
        var expectedDays = doc.RootElement.GetProperty("min_trl_sr0_conf95").GetDouble();

        var resultYears = MinimumTrackRecordLength.Compute(
            observedSharpe: observedSharpe,
            benchmarkSharpe: 0.0,
            skewness: skew,
            kurtosis: kurtosis,
            significanceLevel: 0.05,
            tradingDaysPerYear: tradingDays);
        var resultDays = resultYears * tradingDays;

        AssertScalarWithin(resultDays, expectedDays, PrecisionExact, "MinTRL (days) vs reference");

        // Boundary: MinTRL is +infinity when the observed Sharpe does not exceed the benchmark.
        var unreachable = MinimumTrackRecordLength.Compute(
            observedSharpe: 0.5,
            benchmarkSharpe: 1.0,
            skewness: skew,
            kurtosis: kurtosis,
            significanceLevel: 0.05,
            tradingDaysPerYear: tradingDays);
        unreachable.Should().Be(double.PositiveInfinity);
    }
}
