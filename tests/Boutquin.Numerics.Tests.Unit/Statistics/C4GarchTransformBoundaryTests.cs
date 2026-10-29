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

using System.Numerics;
using System.Reflection;
using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class C4GarchTransformBoundaryTests
{
    [Fact]
    public void SaturatedDoublePersistence_UsesDeclaredStrictCeiling()
    {
        // Local transform contract, not optimizer reachability: at candidate1,
        // the declared1e-8 margin dominates16 double ULPs.
        ApplyProductionClamp(1.0).Should().Be(0.99999999);
        ApplyProductionClamp(0.5).Should().Be(0.5);
    }

    [Fact]
    public void SaturatedFloatPersistence_UsesRepresentableStrictCeiling()
    {
        // At1, float ULP=2^-23;16 ULPs=2^-19.
        ApplyProductionClamp(1.0f).Should().Be(0.9999980926513671875f);
        ApplyProductionClamp(0.5f).Should().Be(0.5f);
    }

    [Fact]
    public void SaturatedHalfPersistence_UsesRepresentableStrictCeiling()
    {
        // At1, Half ULP=2^-10;16 ULPs=1/64, ceiling63/64.
        ApplyProductionClamp((Half)1).Should().Be((Half)0.984375);
        ApplyProductionClamp((Half)0.5).Should().Be((Half)0.5);
    }

    private static T ApplyProductionClamp<T>(T candidate)
        where T : struct, IFloatingPointIeee754<T>
    {
        // Exercise the actual private production boundary without changing its
        // API or reproducing its formula in a test implementation.
        var method = typeof(GarchOneOneEstimator<T>).GetMethod(
            "StrictlyBelowOne", BindingFlags.Static | BindingFlags.NonPublic)!;
        return (T)method.Invoke(null, new object[] { candidate })!;
    }
}
