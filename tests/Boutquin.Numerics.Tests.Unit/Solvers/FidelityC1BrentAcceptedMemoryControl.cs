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

public sealed class FidelityC1BrentAcceptedMemoryControl
{
    [Fact]
    public void RoundedAcceptedHalfStepRetainsPreviousDisplacement()
    {
        const double shift = 1048576;
        var ratio = BitConverter.Int64BitsToDouble(0x3FDA827999FCEF31);
        double[] nodes = [0, 0.5, 1, 2];
        double[] values = [1, -.4 * ratio, -ratio, -1];
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
        new BrentSolver<double>(1e-12, 3).Solve(Objective, shift, shift + 2);
        // The first inverse step is strictly smaller than1/2, but at this
        // abscissa scale its updated coordinate rounds to shift+1/2. Hence the
        // half-width safeguard does not overwrite e on the following iteration.
        Assert.Equal(5, visits.Count);
        Assert.Equal(shift + .5, visits[3]);
        var a = 1d;
        var b = .5;
        var c = 0d;
        var fa = -ratio;
        var fb = -.4 * ratio;
        var fc = 1d;
        var inverse = a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb));
        // Independent Lagrange inverse is .2434824594..., a .2565... step.
        // Correct e=-1 admits it; replacing e with the current m=-.5 rejects
        // it and takes shift+.25. Two abscissa ULPs cover final rounding.
        Assert.InRange(Math.Abs(visits[4] - (shift + inverse)), 0, Math.ScaleB(1, -31));
    }
}
