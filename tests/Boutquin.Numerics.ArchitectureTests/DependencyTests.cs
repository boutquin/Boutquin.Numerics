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
using FluentAssertions;
using NetArchTest.Rules;

namespace Boutquin.Numerics.ArchitectureTests;

/// <summary>
/// Verifies that Boutquin.Numerics has zero domain dependencies.
/// It must not reference MarketData, Analytics, Trading, OptionPricing, or Domain.
/// </summary>
public sealed class DependencyTests
{
    private static readonly Assembly s_numericsAssembly =
        Assembly.Load("Boutquin.Numerics");

    [Theory]
    [InlineData("Boutquin.MarketData")]
    [InlineData("Boutquin.Analytics")]
    [InlineData("Boutquin.Trading")]
    [InlineData("Boutquin.OptionPricing")]
    [InlineData("Boutquin.Domain")]
    public void Numerics_ShouldNotDependOn(string forbidden)
    {
        var result = Types.InAssembly(s_numericsAssembly)
            .Should()
            .NotHaveDependencyOnAny(forbidden)
            .GetResult();

        result.IsSuccessful.Should().BeTrue(
            because: $"Numerics must not depend on {forbidden}: [{string.Join(", ", result.FailingTypes?.Select(t => t.Name) ?? [])}]");
    }
}
