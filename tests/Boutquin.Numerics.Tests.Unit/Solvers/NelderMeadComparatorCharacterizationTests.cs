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
using Xunit.Abstractions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Analytic minimum checks for <see cref="NelderMeadMinimizer{T}"/>.
/// Expected values come from the objective functions, independently of solver trajectories.
/// </summary>
public sealed class NelderMeadComparatorCharacterizationTests
{
    private readonly ITestOutputHelper _output;

    public NelderMeadComparatorCharacterizationTests(ITestOutputHelper output) => _output = output;

    private static double Rosenbrock(double[] v)
    {
        var a = 1.0 - v[0];
        var b = v[1] - v[0] * v[0];
        return a * a + 100.0 * b * b;
    }

    private static double Sphere(double[] v)
    {
        var s = 0.0;
        foreach (var x in v)
        {
            s += x * x;
        }

        return s;
    }

    private static double Beale(double[] v)
    {
        var x = v[0];
        var y = v[1];
        var t1 = 1.5 - x + x * y;
        var t2 = 2.25 - x + x * y * y;
        var t3 = 2.625 - x + x * y * y * y;
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    [Fact]
    public void Rosenbrock_ReachesDefiningMinimum()
    {
        var r = NelderMeadMinimizer<double>.Minimize(Rosenbrock, new[] { -1.2, 1.0 });
        _output.WriteLine($"ros: [{r.Argmin[0]:R},{r.Argmin[1]:R}] v={r.MinValue:R} it={r.Iterations} c={r.Converged}");
        r.Converged.Should().BeTrue();
        r.Argmin[0].Should().BeApproximately(1, 1e-8);
        r.Argmin[1].Should().BeApproximately(1, 1e-8);
        r.MinValue.Should().BeLessThan(1e-16);
    }

    [Fact]
    public void Sphere3D_ReachesDefiningMinimum()
    {
        var r = NelderMeadMinimizer<double>.Minimize(Sphere, new[] { 3.0, -2.0, 1.5 });
        _output.WriteLine($"sph: [{r.Argmin[0]:R},{r.Argmin[1]:R},{r.Argmin[2]:R}] v={r.MinValue:R} it={r.Iterations} c={r.Converged}");
        r.Converged.Should().BeTrue();
        foreach (var coordinate in r.Argmin)
        {
            coordinate.Should().BeApproximately(0, 1e-8);
        }
        r.MinValue.Should().BeLessThan(1e-16);
    }

    [Fact]
    public void Beale_ReachesDefiningMinimumWithTemporalTies()
    {
        var r = NelderMeadMinimizer<double>.Minimize(Beale, new[] { 1.0, 1.0 });
        _output.WriteLine($"bea: [{r.Argmin[0]:R},{r.Argmin[1]:R}] v={r.MinValue:R} it={r.Iterations} c={r.Converged}");
        r.Converged.Should().BeTrue();
        // The governing temporal tie policy changes the historical 88-iteration
        // physical-index trajectory. Beale's three squared residuals vanish at (3,.5).
        r.Argmin[0].Should().BeApproximately(3, 1e-8);
        r.Argmin[1].Should().BeApproximately(0.5, 1e-8);
        r.MinValue.Should().BeLessThan(1e-16);
    }
}
