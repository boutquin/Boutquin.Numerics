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

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class FidelityC1BrentStrictAcceptanceControl
{
    [Fact]
    public void InterpolationHistoryEqualityRejectsAndPreservesNextBisection()
    {
        // This dyadic is the rounded root sqrt(2)-1 selected for the exact
        // floating-point equality of the NR history bound (derivation retained).
        var ratio = BitConverter.Int64BitsToDouble(0x3FDA827999FCEF32);
        double[] nodes = [0, 0.5, 1, 2];
        double[] values = [1, -.4 * ratio, -ratio, -1];
        var visits = new List<double>();
        double Objective(double x)
        {
            visits.Add(x);
            for (var i = 1; i < nodes.Length; i++)
            {
                if (x <= nodes[i]) { return values[i - 1] + (values[i] - values[i - 1]) * ((x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])); }
            }
            return values[^1];
        }
        new BrentSolver<double>(1e-12, 3).Solve(Objective, 0, 2);
        // Equal endpoint magnitudes force x=1. The next inverse interpolation
        // equals the history bound: strict rejection takes midpoint0.5 and e=-.5.
        // At that point the candidate0.24348... exceeds the allowed .25 step,
        // so the next visit is0.25. Accepting equality incorrectly retains e=-1
        // and admits that different candidate on the following iteration.
        Assert.Equal(new[] { 0d, 2d, 1d, 0.5, 0.25 }, visits);
    }
}
