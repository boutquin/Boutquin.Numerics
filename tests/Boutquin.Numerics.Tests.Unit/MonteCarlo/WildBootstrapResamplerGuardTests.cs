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

using System.Reflection;

using Boutquin.Numerics.MonteCarlo;
using Boutquin.Numerics.Random;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

/// <summary>
/// Routing-invariant guard tests for <see cref="WildBootstrapResampler{T}"/>.
/// Pins the constructor's <c>Enum.IsDefined</c> validation and the
/// <c>NextWeight</c> dispatch default against drift between the
/// <see cref="WildBootstrapWeights"/> enum and the dispatch arms.
/// </summary>
public sealed class WildBootstrapResamplerGuardTests
{
    [Fact]
    public void Constructor_UndefinedWeightsValue_ThrowsArgumentOutOfRangeExceptionNamingTheSupportedFamilies()
    {
        var rng = new Pcg64RandomSource(seed: 42UL);
        var bogus = (WildBootstrapWeights)42;

        var act = () => new WildBootstrapResampler<double>(bogus, rng);

        act.Should()
            .Throw<ArgumentOutOfRangeException>()
            .Where(ex => ex.ParamName == "weights")
            .Where(ex => ex.Message.Contains("Mammen"))
            .Where(ex => ex.Message.Contains("Rademacher"))
            .Where(ex => ex.Message.Contains("Gaussian"));
    }

    [Fact]
    public void Resample_WhenWeightsFieldIsCorruptedBypassingConstructor_DispatchDefaultThrowsRoutingInvariantViolation()
    {
        // Construct a legitimate instance via the validating constructor.
        var rng = new Pcg64RandomSource(seed: 42UL);
        var resampler = new WildBootstrapResampler<double>(WildBootstrapWeights.Rademacher, rng);

        // Bypass the constructor's Enum.IsDefined guard by writing an
        // out-of-range value directly to the private _weights field. This
        // simulates the drift the dispatch default is designed to catch:
        // a future WildBootstrapWeights enum value added without a matching
        // arm in NextWeight().
        var weightsField = typeof(WildBootstrapResampler<double>)
            .GetField("_weights", BindingFlags.Instance | BindingFlags.NonPublic);
        weightsField.Should().NotBeNull("the dispatch's routing field must be reachable for drift simulation");
        weightsField!.SetValue(resampler, (WildBootstrapWeights)0xFE);

        var act = () => resampler.Resample(new double[] { 1.0, 2.0, 3.0 });

        act.Should()
            .Throw<InvalidOperationException>()
            .Where(ex => ex.Message.Contains("WildBootstrapResampler"))
            .Where(ex => ex.Message.Contains("WildBootstrapWeights"))
            .Where(ex => ex.Message.Contains("Enum.IsDefined"))
            .Where(ex => ex.Message.Contains("drift"));
    }
}
