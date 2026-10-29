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

public sealed class FidelityC1BrentGeometricBoundaryControl
{
    [Fact]
    public void GeometricAcceptanceBoundRejectsEquality()
    {
        const double shift = 1048576;
        var ratio = BitConverter.Int64BitsToDouble(0x3FDA827999FCEF31);
        var tolerance = BitConverter.Int64BitsToDouble(0x3FD00E75C5AAA365);
        double[] nodes = [0, .5, 1, 2];
        double[] values = [1, -.45 * ratio, -ratio, -1];
        var visits = new List<double>();
        double Objective(double coordinate)
        {
            visits.Add(coordinate);
            var x = coordinate - shift;
            for (var i = 1; i < nodes.Length; i++)
            {
                if (x <= nodes[i]) { return values[i - 1] + (values[i] - values[i - 1]) * ((x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])); }
            }
            return values[^1];
        }
        new BrentSolver<double>(tolerance, 3).Solve(Objective, shift, shift + 2);
        // Independent NR-bound construction: the first accepted inverse step
        // rounds to shift+.5, retaining e=-1. At that point the printed geometric
        // bound3*m*q-abs(tol*q) equals2*p exactly (bits0x3FE27165E551BA73).
        // It is smaller than abs(e*q), so this exercises the other active bound.
        // Strict rejection takes shift+.25; accepting equality takes the inverse
        // interpolant shift+.1877206430..., whose displacement is .3122793570....
        Assert.Equal(new[] { shift, shift + 2, shift + 1, shift + .5, shift + .25 }, visits);
    }
}
