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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1NamedSimplexControls
{
    [Fact]
    public void Reflection() => new Arena20260928NelderMeadTests().StandardCoefficientsAndComparisons_FollowPrimaryBranch("reflection");

    [Fact]
    public void Expansion() => new Arena20260928NelderMeadTests().StandardCoefficientsAndComparisons_FollowPrimaryBranch("expansion");

    [Fact]
    public void RejectedExpansion() => new Arena20260928NelderMeadTests().StandardCoefficientsAndComparisons_FollowPrimaryBranch("rejected-expansion");

    [Fact]
    public void OutsideEquality() => new Arena20260928NelderMeadTests().StandardCoefficientsAndComparisons_FollowPrimaryBranch("outside-equality");

    [Fact]
    public void InsideAccepted() => new Arena20260928NelderMeadTests().StandardCoefficientsAndComparisons_FollowPrimaryBranch("inside");

    [Fact]
    public void InsideEqualityShrink() => new Arena20260928NelderMeadTests().StandardCoefficientsAndComparisons_FollowPrimaryBranch("inside-equality-shrink");

    [Fact]
    public void ReflectionZero() => new Arena20260928NelderMeadTests().InvalidFiniteOption_IsRefusedBeforeObjective("Reflection", 0);

    [Fact]
    public void ExpansionOne() => new Arena20260928NelderMeadTests().InvalidFiniteOption_IsRefusedBeforeObjective("Expansion", 1);

    [Fact]
    public void ContractionZero() => new Arena20260928NelderMeadTests().InvalidFiniteOption_IsRefusedBeforeObjective("Contraction", 0);

    [Fact]
    public void ContractionOne() => new Arena20260928NelderMeadTests().InvalidFiniteOption_IsRefusedBeforeObjective("Contraction", 1);

    [Fact]
    public void ShrinkZero() => new Arena20260928NelderMeadTests().InvalidFiniteOption_IsRefusedBeforeObjective("Shrink", 0);

    [Fact]
    public void ShrinkOne() => new Arena20260928NelderMeadTests().InvalidFiniteOption_IsRefusedBeforeObjective("Shrink", 1);

    [Fact]
    public void ExpansionBelowReflection() => new Arena20260928NelderMeadTests().ExpansionNotGreaterThanReflection_IsRefused();

    [Fact]
    public void NonshrinkStableTie() => new Arena20260928NelderMeadTests().EqualNewVertices_PreserveExistingBest(false);

    [Fact]
    public void ShrinkStableTie() => new Arena20260928NelderMeadTests().EqualNewVertices_PreserveExistingBest(true);

    [Fact]
    public void AdaptiveExpansion() => new Arena20260928NelderMeadVariantTests().CallerSuppliedGaoHanCoefficients_ProducePublishedGeometricSteps(3, "expansion");

    [Fact]
    public void AdaptiveContraction() => new Arena20260928NelderMeadVariantTests().CallerSuppliedGaoHanCoefficients_ProducePublishedGeometricSteps(3, "contraction");

    [Fact]
    public void AdaptiveShrink() => new Arena20260928NelderMeadVariantTests().CallerSuppliedGaoHanCoefficients_ProducePublishedGeometricSteps(3, "shrink");

}
