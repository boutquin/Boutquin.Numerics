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

namespace Boutquin.Numerics.Tests.Unit.Statistics;

[Trait("Tier", "Unit")]
public sealed class C4CorrelationClippingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(-1)]
    public void ExactRankOneCovariance_ClipsCorrelationRoundingToUnitInterval(int sign)
    {
        // DenoisedCovarianceEstimator-002, MLAM snippet2.3 cov2corr.
        // This exact rational matrix is 3*v*v', v=(1,2*sign), hence PSD.
        // Its mathematical correlations are exactly sign. IEEE evaluation
        // 6/(sqrt(3)*sqrt(12)) exceeds1 by one representable step, so the
        // source clamp is observable before any spectral cleaning.
        double[,] covariance = { { 3, sign * 6 }, { sign * 6, 12 } };
        var result = CovarianceHelpers<double>.CovarianceToCorrelation(covariance).Correlation;
        result[0, 1].Should().Be(sign);
        result[1, 0].Should().Be(sign);
    }
}
