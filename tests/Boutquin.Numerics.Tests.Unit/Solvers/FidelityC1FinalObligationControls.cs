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

using System.Runtime.InteropServices;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Solvers.Internal;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

public sealed class FidelityC1FinalObligationControls
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ParameterStopIncludesNonzeroParameterNorm(bool writer)
    {
        static double[] Residual(double[] x) => [x[0] - 3.6, x[1] - 4.8];
        static double[,] Jacobian(double[] _) => new double[,] { { 1, 0 }, { 0, 1 } };
        static void WriteResidual(ReadOnlySpan<double> x, Span<double> r) { r[0] = x[0] - 3.6; r[1] = x[1] - 4.8; }
        static void WriteJacobian(ReadOnlySpan<double> _, Span<double> j) { j[0] = 1; j[1] = 0; j[2] = 0; j[3] = 1; }
        var solver = new LevenbergMarquardtSolver<double>(1, 1e-30, .1, 1e-30, 1);
        var result = writer ? solver.Solve(WriteResidual, [3, 4], 2, WriteJacobian) : solver.Solve(Residual, [3, 4], Jacobian);
        // h=(.3,.4), norm(h)=.5, norm(x)=5. The selected threshold .1*(5+.1)=.51
        // permits stopping; dropping norm(x) gives .01 and incorrectly continues.
        Assert.Equal(LmTerminationReason.ParameterToleranceReached, result.TerminationReason);
        Assert.InRange(Math.Abs(result.Parameters[0] - 3.3), 0, 1e-14);
        Assert.InRange(Math.Abs(result.Parameters[1] - 4.4), 0, 1e-14);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OneSidedLinearDerivativeUsesRepresentedDisplacement(bool writer, bool upper)
    {
        const double x = 1.1;
        var pool = new LevenbergMarquardtBuffers<double>(1, 1);
        pool.Residual[0] = x;
        static double[] Residual(double[] p) => [p[0]];
        static void Write(ReadOnlySpan<double> p, Span<double> r) => r[0] = p[0];
        double[] lower = upper ? [] : [x];
        double[] higher = upper ? [x] : [];
        if (writer) { FiniteDifferenceJacobian<double>.EvaluateInto(Write, [x], pool, lower, higher); }
        else { FiniteDifferenceJacobian<double>.EvaluateInto(Residual, [x], pool, lower, higher); }
        // The two represented abscissae appear unchanged in the numerator;
        // dividing their represented displacement gives exactly one.
        Assert.Equal(1, pool.Jacobian[0, 0]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OneSidedQuadraticExhibitsExactFirstOrderTaylorTerm(bool writer, bool upper)
    {
        var sign = upper ? -1d : 1d;
        var pool = new LevenbergMarquardtBuffers<double>(1, 1);
        pool.Residual[0] = 1;
        double[] Residual(double[] x) => [1 + sign * x[0] + x[0] * x[0]];
        void Write(ReadOnlySpan<double> x, Span<double> r) => r[0] = 1 + sign * x[0] + x[0] * x[0];
        double[] lower = upper ? [] : [0];
        double[] higher = upper ? [0] : [];
        if (writer) { FiniteDifferenceJacobian<double>.EvaluateInto(Write, [0], pool, lower, higher); }
        else { FiniteDifferenceJacobian<double>.EvaluateInto(Residual, [0], pool, lower, higher); }
        // f''=2; signed effective h is +/-2^-26. The exact quotient is
        // sign*(1+2^-26), including the h*f''/2 term, with no higher terms.
        Assert.Equal(sign * (1 + Math.ScaleB(1, -26)), pool.Jacobian[0, 0]);
    }

    [Fact]
    public void ShrinkPreservesOriginalOrderingOfNonbestTies()
    {
        var visits = new List<double[]>();
        double Objective(double[] x)
        {
            visits.Add((double[])x.Clone());
            if (x.All(v => v == 0)) { return 0; }
            for (var j = 0; j < 3; j++)
            {
                if (Enumerable.Range(0, 3).All(k => k == j || x[k] == 0))
                {
                    if (x[j] == 1) { return j + 1; }
                    if (x[j] == .5) { return 1; }
                }
            }
            return x[2] == -.5 ? .5 : 4;
        }
        NelderMeadMinimizer<double>.Minimize(Objective, [0, 0, 0], new SimplexOptions<double>(MaxIterations: 1, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        // Initial values0,1,2,3. Failed inside contraction triggers shrink;
        // all three new nonbest values tie1. Their inherited order leaves the
        // third axis worst. Next centroid(1/6,1/6,0) reflects it to(1/3,1/3,-1/2).
        Assert.Equal(10, visits.Count);
        Assert.Equal(new[] { 1d / 3, 1d / 3, -.5 }, visits[9]);
    }

    [Fact]
    public void WriterCallbacksReceiveCallerPoolStorageInPlace()
    {
        var pool = new LevenbergMarquardtBuffers<double>(2, 3);
        var residualCalls = 0;
        var jacobianCalls = 0;
        void Residual(ReadOnlySpan<double> x, Span<double> r)
        {
            residualCalls++;
            Assert.Equal(2, x.Length);
            Assert.Equal(3, r.Length);
            Assert.True(r.Overlaps(pool.Residual.AsSpan()) || r.Overlaps(pool.TrialResidual.AsSpan()));
            r[0] = x[0] - 2; r[1] = x[1] + 1; r[2] = 3;
        }
        void Jacobian(ReadOnlySpan<double> x, Span<double> j)
        {
            jacobianCalls++;
            Assert.Equal(2, x.Length);
            Assert.Equal(6, j.Length);
            var expected = MemoryMarshal.CreateSpan(ref pool.Jacobian[0, 0], 6);
            Assert.True(j.Overlaps(expected));
            j.Clear(); j[0] = 1; j[3] = 1;
        }
        var solver = new LevenbergMarquardtSolver<double>(10, 1e-14, 1e-14, 1e-12, .001);
        var result = solver.Solve(Residual, [0, 0], pool, Jacobian);
        Assert.True(residualCalls >= 2);
        Assert.True(jacobianCalls >= 2);
        Assert.InRange(Math.Abs(result.Parameters[0] - 2), 0, 1e-7);
        Assert.InRange(Math.Abs(result.Parameters[1] + 1), 0, 1e-7);
    }
    [Fact]
    public void ExactRootAtEitherEndpointIsAccepted()
    {
        foreach (var root in new[] { 0d, 1d })
        {
            RootSolverResult<double>? result = null;
            var error = Record.Exception(() => result = new BrentSolver<double>(1e-12, 10).Solve(x => x - root, 0, 1));
            Assert.Null(error);
            Assert.NotNull(result);
            Assert.True(result.Converged);
            Assert.Equal(root, result.Root);
        }
    }
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NonfiniteWorstPreventsCoordinateOnlyStopping(double invalid)
    {
        var calls = 0;
        var result = NelderMeadMinimizer<double>.Minimize(x => ++calls <= 2 ? x[0] * x[0] : invalid, [0],
            new SimplexOptions<double>(MaxIterations: 2, InitialStep: 1, XTolerance: .6, FTolerance: 0));
        // First shrink leaves coordinates0,.5, within XTolerance, but the worst
        // value is invalid. The finite-worst guard must prevent success next time.
        Assert.False(result.Converged);
        Assert.Equal(0, result.MinValue);
        Assert.Equal(0, result.Argmin[0]);
    }

    [Fact]
    public void DistinctNonfiniteValuesTieAndKeepOriginalOrder()
    {
        var visits = new List<double[]>();
        double Objective(double[] x)
        {
            visits.Add((double[])x.Clone());
            return (x[0], x[1]) switch
            {
                (0, 0) => 0,
                (1, 0) => 1,
                (0, 1) => 2,
                (.5, 0) => double.NaN,
                (0, .5) => double.NegativeInfinity,
                _ => double.PositiveInfinity,
            };
        }
        var result = NelderMeadMinimizer<double>.Minimize(Objective, [0, 0],
            new SimplexOptions<double>(MaxIterations: 2, InitialStep: 1, XTolerance: 0, FTolerance: 0));
        // First shrink gives NaN then-infinity; both share worst rank and retain
        // their temporal order. The next reflection therefore uses(0,.5) as worst.
        Assert.Equal(new[] { .5, -.5 }, visits[7]);
        Assert.Equal(new[] { 0d, 0d }, result.Argmin);
        Assert.False(result.Converged);
    }
}
