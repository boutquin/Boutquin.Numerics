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

namespace Boutquin.Numerics.Tests.Unit.Solvers.NistStRD;

/// <summary>
/// Asserts that the deterministic test-parallelism bound for the heavy nonlinear
/// NIST Levenberg&#8211;Marquardt solves (<see cref="NistStRDTests"/>) is genuinely
/// present and applied &#8212; not merely documented. These are reflection assertions
/// against the live xunit attributes, so deleting or misconfiguring either the
/// <c>[Collection]</c> placement on <see cref="NistStRDTests"/> or the
/// <c>[CollectionDefinition]</c> on <see cref="NistNonlinearSolveCollection"/> fails
/// this test.
/// </summary>
/// <remarks>
/// xunit v2's <see cref="CollectionAttribute"/> exposes its collection name only as a
/// constructor argument (no public property), so the name assertions here read the
/// applied attributes through <see cref="CustomAttributeData"/> rather than a
/// materialized attribute instance.
/// </remarks>
public sealed class NistStRDParallelismConfigTests
{
    [Fact]
    public void NistStRDTests_IsPlacedInTheBoundedNonParallelCollection()
    {
        var collectionData = typeof(NistStRDTests)
            .GetCustomAttributesData()
            .SingleOrDefault(d => d.AttributeType == typeof(CollectionAttribute));

        collectionData.Should().NotBeNull(
            "NistStRDTests hosts the highest-iteration, most convergence-sensitive NIST/LM " +
            "solves (Lanczos1-3, Eckerle4, Bennett5, MGH17) and must be pinned to a bounded, " +
            "non-parallel xunit collection so it never competes for CPU against the rest of " +
            "the suite — the deterministic parallelism bound for the heavy nonlinear solves");
        collectionData!.ConstructorArguments.Should().HaveCount(1);
        collectionData.ConstructorArguments[0].Value.Should().Be(
            NistNonlinearSolveCollection.Name,
            "the [Collection] name must match the bounded [CollectionDefinition] exactly, " +
            "otherwise NistStRDTests silently falls back to an unbounded implicit collection");
    }

    [Fact]
    public void NistNonlinearSolveCollection_DefinitionNameMatchesTheSharedConstant()
    {
        var definitionData = typeof(NistNonlinearSolveCollection)
            .GetCustomAttributesData()
            .SingleOrDefault(d => d.AttributeType == typeof(CollectionDefinitionAttribute));

        definitionData.Should().NotBeNull(
            "the bounded collection must be declared via [CollectionDefinition] for xunit " +
            "to recognize it as a distinct, non-parallel collection");
        definitionData!.ConstructorArguments.Should().HaveCount(1);
        definitionData.ConstructorArguments[0].Value.Should().Be(
            NistNonlinearSolveCollection.Name,
            "member classes join the collection by this exact name; a drifted definition " +
            "name would orphan every [Collection(Name)] member into an implicit collection");
    }

    [Fact]
    public void NistNonlinearSolveCollection_DisablesCrossCollectionParallelization()
    {
        var definitionAttribute = typeof(NistNonlinearSolveCollection)
            .GetCustomAttribute<CollectionDefinitionAttribute>();

        definitionAttribute.Should().NotBeNull(
            "the bounded collection must be declared via [CollectionDefinition] for xunit " +
            "to recognize it as a distinct, non-parallel collection");
        definitionAttribute!.DisableParallelization.Should().BeTrue(
            "the bound is only real if this collection is excluded from xunit's parallel " +
            "bucket — DisableParallelization=false would leave the [Collection] pairing " +
            "present but inert (no actual cap on cross-collection parallelism)");
    }
}
