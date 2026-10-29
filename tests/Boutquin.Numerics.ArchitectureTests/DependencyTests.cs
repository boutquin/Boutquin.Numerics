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
using System.Xml.Linq;
using FluentAssertions;
using NetArchTest.Rules;

namespace Boutquin.Numerics.ArchitectureTests;

/// <summary>
/// Verifies that Boutquin.Numerics has zero dependencies on any other
/// Boutquin.* library in the ecosystem. Numerics sits at the deepest
/// foundation tier of the dependency graph (see
/// <c>~/Developer/data-architecture.md</c>) — every other Boutquin.* library
/// either consumes Numerics directly or sits above it transitively, so a
/// dependency in the reverse direction would be a layering violation.
/// </summary>
public sealed class DependencyTests
{
    private static readonly Assembly s_numericsAssembly =
        Assembly.Load("Boutquin.Numerics");

    /// <summary>
    /// Every Boutquin.* library/package currently live in the ecosystem,
    /// EXCLUDING Boutquin.Numerics itself (the subject under test). Sourced
    /// from the dependency-edges table in <c>~/Developer/data-architecture.md</c>
    /// (verified 2026-07-07): <c>Domain</c>, <c>Storage</c> (foundation
    /// siblings); <c>MarketData</c>, <c>MarketData.Adapter</c> (data kernel);
    /// <c>Curves</c>, <c>Trading</c> (domain libraries); <c>OptionPricing</c>
    /// (pricing); <c>Trading.TaxEngine</c> (commercial/app); <c>Mediator</c>
    /// (standalone).
    /// </summary>
    /// <remarks>
    /// This is the staleness guard for <see cref="s_forbiddenDependencies"/>:
    /// every forbidden entry must resolve to a name in this collection, or it
    /// names a dead/renamed assembly (e.g. the retired "Boutquin.Analytics" —
    /// renamed to <c>Boutquin.Curves</c> in the ecosystem's architecture doc)
    /// that no longer corresponds to anything and has stopped guarding
    /// anything, silently. See
    /// <see cref="ForbiddenDependencies_ShouldAllBeKnownLiveEcosystemAssemblies"/>.
    /// </remarks>
    private static readonly IReadOnlyCollection<string> s_liveEcosystemAssemblies =
    [
        "Boutquin.Domain",
        "Boutquin.Storage",
        "Boutquin.MarketData",
        "Boutquin.MarketData.Adapter",
        "Boutquin.Curves",
        "Boutquin.Trading",
        "Boutquin.OptionPricing",
        "Boutquin.Trading.TaxEngine",
        "Boutquin.Mediator",
    ];

    /// <summary>
    /// Assemblies Numerics must never depend on. Deliberately maintained as a
    /// literal <em>separate</em> from <see cref="s_liveEcosystemAssemblies"/>
    /// (not derived from it) so a hand-edit that adds an entry here without
    /// also confirming it is still live is caught by
    /// <see cref="ForbiddenDependencies_ShouldAllBeKnownLiveEcosystemAssemblies"/>
    /// rather than silently guarding a dead name — which is exactly how the
    /// previous "Boutquin.Analytics" staleness went undetected.
    /// </summary>
    private static readonly string[] s_forbiddenDependencies =
    [
        "Boutquin.MarketData",
        "Boutquin.MarketData.Adapter",
        "Boutquin.Curves",
        "Boutquin.Trading",
        "Boutquin.OptionPricing",
        "Boutquin.Trading.TaxEngine",
        "Boutquin.Domain",
        "Boutquin.Mediator",
        "Boutquin.Storage",
    ];

    public static IEnumerable<object[]> ForbiddenDependencyTheoryData =>
        s_forbiddenDependencies.Select(name => new object[] { name });

    [Theory]
    [MemberData(nameof(ForbiddenDependencyTheoryData))]
    public void Numerics_ShouldNotDependOn(string forbidden)
    {
        var result = Types.InAssembly(s_numericsAssembly)
            .Should()
            .NotHaveDependencyOnAny(forbidden)
            .GetResult();

        result.IsSuccessful.Should().BeTrue(
            because: $"Numerics must not depend on {forbidden}: [{string.Join(", ", result.FailingTypes?.Select(t => t.Name) ?? [])}]");
    }

    [Fact]
    public void ForbiddenDependencies_ShouldAllBeKnownLiveEcosystemAssemblies()
    {
        // Staleness self-check: every forbidden-dependency entry must resolve to a
        // currently-live Boutquin.* library. A forbidden name that no longer exists
        // in the ecosystem (a dead rename, a retired repo) stops guarding anything
        // in practice while the theory above keeps reporting green — the firewall
        // goes inert while still reading as "enforced". This assertion fails
        // loudly instead of letting that drift pass unnoticed (recurrence of the
        // 2026-07 "Boutquin.Analytics" finding).
        var staleEntries = s_forbiddenDependencies.Except(s_liveEcosystemAssemblies).ToArray();

        staleEntries.Should().BeEmpty(
            because: "every forbidden dependency must name a live Boutquin.* assembly per " +
                     $"~/Developer/data-architecture.md; found stale/unknown entries: [{string.Join(", ", staleEntries)}]");
    }

    [Fact]
    public void PublicApiAnalyzer_ShouldBeScopedToTheSourceProjectWithoutAPathCondition()
    {
        DirectoryInfo? repository = new(AppContext.BaseDirectory);
        while (repository is not null && !File.Exists(Path.Combine(repository.FullName, "Directory.Build.props")))
        {
            repository = repository.Parent;
        }

        repository.Should().NotBeNull();
        var sourceProject = XDocument.Load(Path.Combine(repository!.FullName, "src", "Boutquin.Numerics", "Boutquin.Numerics.csproj"));
        var analyzerReferences = sourceProject.Descendants("PackageReference")
            .Where(reference => (string?)reference.Attribute("Include") == "Microsoft.CodeAnalysis.PublicApiAnalyzers")
            .ToArray();

        analyzerReferences.Should().ContainSingle("the source project must always load the public API analyzer");
        analyzerReferences[0].Attribute("Condition").Should().BeNull();
        analyzerReferences[0].Parent?.Attribute("Condition").Should().BeNull();

        var sharedProperties = XDocument.Load(Path.Combine(repository.FullName, "Directory.Build.props"));
        sharedProperties.Descendants("PackageReference")
            .Should().NotContain(reference => (string?)reference.Attribute("Include") == "Microsoft.CodeAnalysis.PublicApiAnalyzers",
                "other projects must not inherit the source project's analyzer");
    }
}
