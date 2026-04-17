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

using Boutquin.Numerics.Distributions;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.GenericParity;

public sealed class Distributions_ParityTests
{
    [Fact]
    public void CumulativeNormal_GenericMatchesLegacy_AtDouble()
    {
        var testValues = new[] { -3.0, -1.5, 0.0, 1.5, 3.0 };

        foreach (var x in testValues)
        {
            var legacy = CumulativeNormal.Evaluate(x);
            var generic = CumulativeNormal<double>.Evaluate(x);

            generic.Should().BeApproximately(legacy, 1e-12);
        }
    }

    [Fact]
    public void InverseNormal_GenericMatchesLegacy_AtDouble()
    {
        var testValues = new[] { 0.01, 0.25, 0.5, 0.75, 0.99 };

        foreach (var p in testValues)
        {
            var legacy = InverseNormal.Evaluate(p);
            var generic = InverseNormal<double>.Evaluate(p);

            generic.Should().BeApproximately(legacy, 1e-12);
        }
    }

    [Fact]
    public void NormalDistribution_Pdf_GenericMatchesLegacy_AtDouble()
    {
        var testValues = new[] { -2.0, -1.0, 0.0, 1.0, 2.0 };

        foreach (var x in testValues)
        {
            var legacy = NormalDistribution.Pdf(x);
            var generic = NormalDistribution<double>.Pdf(x);

            generic.Should().BeApproximately(legacy, 1e-12);
        }
    }

    [Fact]
    public void NormalDistribution_Cdf_GenericMatchesLegacy_AtDouble()
    {
        var testValues = new[] { -2.0, -1.0, 0.0, 1.0, 2.0 };

        foreach (var x in testValues)
        {
            var legacy = NormalDistribution.Cdf(x);
            var generic = NormalDistribution<double>.Cdf(x);

            generic.Should().BeApproximately(legacy, 1e-12);
        }
    }

    [Fact]
    public void NormalDistribution_InverseCdf_GenericMatchesLegacy_AtDouble()
    {
        var testValues = new[] { 0.01, 0.25, 0.5, 0.75, 0.99 };

        foreach (var p in testValues)
        {
            var legacy = NormalDistribution.InverseCdf(p);
            var generic = NormalDistribution<double>.InverseCdf(p);

            generic.Should().BeApproximately(legacy, 1e-12);
        }
    }
}
