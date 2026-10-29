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
public sealed class FidelityC1WriterParityTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void WriterAndAllocatingPaths_HaveIdenticalResultBits(bool analytic, bool shrinkPool)
    {
        // Repository parity contract, with an independent analytic sanity check:
        // minimize .5*((x-1)^2+(x-2)^2), whose optimum is x=1.5 and cost=.25.
        static double[] Residual(double[] x) => [x[0] - 1, x[0] - 2];
        static double[,] Jacobian(double[] _) => new double[,] { { 1 }, { 1 } };
        static void WriteResidual(ReadOnlySpan<double> x, Span<double> r)
        {
            r[0] = x[0] - 1;
            r[1] = x[0] - 2;
        }
        static void WriteJacobian(ReadOnlySpan<double> _, Span<double> j)
        {
            j[0] = 1;
            j[1] = 1;
        }
        var solver = new LevenbergMarquardtSolver<double>(100, 1e-14, 1e-14, 1e-12, 1e-3);
        var reference = solver.Solve(Residual, [0.0], analytic ? Jacobian : null);
        var allocatingPool = new LevenbergMarquardtBuffers<double>(shrinkPool ? 3 : 1, shrinkPool ? 5 : 2);
        var writerPool = new LevenbergMarquardtBuffers<double>(shrinkPool ? 3 : 1, shrinkPool ? 5 : 2);
        allocatingPool.Reset(1, 2);
        writerPool.Reset(1, 2);
        var pooled = solver.Solve(Residual, [0.0], allocatingPool, analytic ? Jacobian : null);
        var writer = solver.Solve(WriteResidual, [0.0], 2, analytic ? WriteJacobian : null);
        var pooledWriter = solver.Solve(WriteResidual, [0.0], writerPool, analytic ? WriteJacobian : null);
        reference.Parameters[0].Should().BeApproximately(1.5, 1e-8);
        reference.FinalCost.Should().BeApproximately(0.25, 1e-14);
        foreach (var actual in new[] { pooled, writer, pooledWriter })
        {
            actual.Converged.Should().Be(reference.Converged);
            actual.Iterations.Should().Be(reference.Iterations);
            actual.TerminationReason.Should().Be(reference.TerminationReason);
            actual.BoundsActive.Should().Be(reference.BoundsActive);
            BitConverter.DoubleToInt64Bits(actual.FinalCost).Should().Be(BitConverter.DoubleToInt64Bits(reference.FinalCost));
            actual.Parameters.Select(BitConverter.DoubleToInt64Bits).Should().Equal(reference.Parameters.Select(BitConverter.DoubleToInt64Bits));
            actual.FinalResiduals.Select(BitConverter.DoubleToInt64Bits).Should().Equal(reference.FinalResiduals.Select(BitConverter.DoubleToInt64Bits));
            var expectedCovariance = reference.ParameterCovariance;
            var actualCovariance = actual.ParameterCovariance;
            actualCovariance.Should().NotBeNull();
            expectedCovariance.Should().NotBeNull();
            actualCovariance!.Cast<double>().Select(BitConverter.DoubleToInt64Bits)
                .Should().Equal(expectedCovariance!.Cast<double>().Select(BitConverter.DoubleToInt64Bits));
        }
    }
}
