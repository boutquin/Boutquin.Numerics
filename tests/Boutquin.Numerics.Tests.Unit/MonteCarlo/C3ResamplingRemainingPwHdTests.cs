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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class C3ResamplingRemainingPwHdTests
{
    [Fact]
    public void StationaryOptimum_MaTwoLaw_MinimizesExactLeadingMse()
    {
        // X_t=e_t-(1/2)e_(t-1)+(1/4)e_(t-2), Var(e)=1.
        // Independent finite convolution gives R0=21/16,R1=-5/8,R2=1/4;
        // all other covariances vanish. Both-sided G=-1/4,g(0)=9/16.
        const double covarianceMoment = -1.0 / 4;
        const double spectralSum = 9.0 / 16;
        var block = PolitisWhiteBlockLength.StationaryOptimum(covarianceMoment, spectralSum, 324);
        block.Should().BeApproximately(4, 1e-13);
        // Corrected D=81/128; D/N=1/512. Leading MSE is
        // Q(b)=1/(16*b²)+b/512. Q'(b)=-1/(8*b³)+1/512.
        (-1 / (8 * block * block * block) + 1.0 / 512).Should().BeApproximately(0, 1e-15);
        (1 / (16 * block * block) + block / 512).Should().BeApproximately(3.0 / 256, 1e-15);
        // Q''=3/(8*b^4)>0 for b>0 proves the unique minimizer algebraically.
        // This checks the leading coefficient and optimum, not asymptotic remainder bounds.
    }

    [Fact]
    public void HarrellDavis_InteriorDomain_DoesNotCollapseToEndpoints()
    {
        // Exact Beta(1,3) and Beta(3,1) bin integrals on three order statistics.
        HarrellDavisPercentile.Compute<double>([1, 4, 13], .25)
            .Should().BeApproximately(20.0 / 9, 1e-10);
        HarrellDavisPercentile.Compute<double>([1, 4, 13], .75)
            .Should().BeApproximately(92.0 / 9, 1e-10);
        // Endpoints are explicit limiting-value extensions, not the paper's open domain.
        HarrellDavisPercentile.Compute<double>([1, 4, 13], 0).Should().Be(1);
        HarrellDavisPercentile.Compute<double>([1, 4, 13], 1).Should().Be(13);
    }

    [Fact]
    public void HarrellDavis_RejectsOutsideItsDocumentedDomain()
    {
        foreach (var p in new[] { -.25, 1.25, double.NaN, double.PositiveInfinity, double.NegativeInfinity })
        {
            Action invalid = () => HarrellDavisPercentile.Compute<double>([1, 4, 13], p);
            invalid.Should().ThrowExactly<ArgumentOutOfRangeException>().WithParameterName("p");
        }
    }
}
