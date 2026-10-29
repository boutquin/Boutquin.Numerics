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

namespace Boutquin.Numerics.Tests.Unit;

public sealed class OwnerGarchVarianceFloorContractTests
{
    [Fact]
    public void DecayingObservationsRetainTheFittedRecurrenceAndAverageLikelihood()
    {
        var data = Enumerable.Range(0, 32).Select(i => Math.ScaleB(i % 2 == 0 ? 1.0 : -1.0, -i)).ToArray();
        var fit = GarchOneOneEstimator<double>.Fit(data, new() { FixedZeroMean = true });
        var previousVariance = data.Sum(x => x * x) / data.Length;
        var previousShockSquared = previousVariance;
        var objective = 0.0;
        for (var i = 0; i < data.Length; i++)
        {
            var expected = fit.Omega + fit.Alpha * previousShockSquared + fit.Beta * previousVariance;
            Assert.InRange(Math.Abs(fit.ConditionalVariances[i] / expected - 1), 0, 1e-12);
            objective += -0.5 * (Math.Log(expected) + data[i] * data[i] / expected);
            previousVariance = expected;
            previousShockSquared = data[i] * data[i];
        }

        Assert.Equal(objective / data.Length, fit.LogLikelihood, 10);
    }
}
