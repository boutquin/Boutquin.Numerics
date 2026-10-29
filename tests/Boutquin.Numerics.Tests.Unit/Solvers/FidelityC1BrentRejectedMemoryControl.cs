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

public sealed class FidelityC1BrentRejectedMemoryControl
{
    [Fact]
    public void RejectedInterpolationKeepsHistoryForNextAcceptedStep()
    {
        double[] nodes = [0, 0.4, 8.0 / 15, 2.0 / 3, 11.0 / 15, 0.8, 1, 4];
        double[] values = [-1, -1.0 / 16, -1.0 / 32, -1.0 / 64, -1.0 / 256, 1.0 / 8, 1.0 / 4, 3];
        var visited = new List<double>();
        double Objective(double x)
        {
            visited.Add(x);
            for (var i = 1; i < nodes.Length; i++)
            {
                if (x <= nodes[i]) { return values[i - 1] + (values[i] - values[i - 1]) * ((x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])); }
            }
            return values[^1];
        }
        new BrentSolver<double>(1e-12, 7).Solve(Objective, 0, 4);
        double[] history = [0, 4, 1, 0.8, 0.4, 8.0 / 15, 2.0 / 3, 11.0 / 15];
        for (var i = 0; i < history.Length; i++) { Assert.InRange(Math.Abs(visited[i] - history[i]), 0, 2e-14); }
        // The first inverse interpolation524/675 is rejected: it exceeds the
        // allowed history step. Its midpoint11/15 remains on the negative side.
        // Interpolating the inverse through (f,x)=(-1/64,2/3),(-1/256,11/15),
        // (1/8,4/5) independently gives3356/4455. Its displacement89/4455
        // is below1/30; resetting e to zero would instead force midpoint23/30.
        var a = 2.0 / 3;
        var b = 11.0 / 15;
        var c = 4.0 / 5;
        var fa = -1.0 / 64;
        var fb = -1.0 / 256;
        var fc = 1.0 / 8;
        var expected = a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb));
        Assert.InRange(Math.Abs(expected - 3356.0 / 4455), 0, 5e-16);
        Assert.InRange(Math.Abs(visited[8] - expected), 0, 2e-14);
    }
}
