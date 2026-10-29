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

using Boutquin.Numerics.Solvers;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

[Trait("Tier", "Unit")]
public sealed class FidelityC1VariantDiscriminatorTests
{
    [Fact]
    public void InsideContraction_IsIndependentOfReflectionCoefficient()
    {
        // Lagarias eq2.7 uses the old worst vertex. With reflection2 and
        // expansion3, substituting Gao-Han's reflected-point expression is different.
        var visits = new List<double[]>();
        double Objective(double[] x)
        {
            visits.Add((double[])x.Clone());
            return (x[0], x[1]) switch
            {
                (0, 0) => 0,
                (1, 0) => 1,
                (0, 1) => 2,
                (1.5, -2) => 3,
                (0.25, 0.5) => 1,
                _ => 10,
            };
        }
        NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1,
                Reflection: 2, Expansion: 3, XTolerance: 0, FTolerance: 1));
        visits.Should().HaveCount(5);
        visits[3].Should().Equal(1.5, -2);
        visits[4].Should().Equal(0.25, 0.5);
    }

    [Fact]
    public void DefaultSimplex_UsesAbsoluteFloorAndCoordinateMagnitude()
    {
        // Repository contract: .05*max(abs(x_i),1). These coordinates distinguish
        // both branches of the printed Gao-Han eq3.1 (.00025 at zero; .05 otherwise).
        var visits = new List<double[]>();
        double Objective(double[] x) { visits.Add((double[])x.Clone()); return 1; }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 2]);
        visits.Should().HaveCount(3);
        visits[0].Should().Equal(0, 2);
        visits[1].Should().Equal(0.05, 2);
        visits[2].Should().Equal(0, 2.1);
        result.Converged.Should().BeTrue("the declared OR stop permits a flat objective despite large coordinate spread");
    }
}
