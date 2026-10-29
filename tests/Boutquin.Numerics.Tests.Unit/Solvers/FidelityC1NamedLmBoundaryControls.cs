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
public sealed class FidelityC1NamedLmBoundaryControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GradientEquality(bool writer) => new Arena20260928LmClauseTests().GradientBoundary_IsInclusiveAndIndependentOfOtherTolerances(writer, true);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ParameterEquality(bool writer) => new Arena20260928LmClauseTests().ParameterBoundary_UsesNormPlusAbsoluteFloor(writer, true);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FunctionEquality(bool writer) => new Arena20260928LmClauseTests().FunctionBoundary_RequiresActualAndModelRelativeReductions(writer, true);
}
