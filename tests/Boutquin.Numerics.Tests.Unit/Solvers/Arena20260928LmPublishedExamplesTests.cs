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
public sealed class Arena20260928LmPublishedExamplesTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MoreHelix_OriginalStartReachesPublishedZeroResidual(bool writer)
    {
        double[] R(double[] x)
        {
            var angle = Math.Atan(x[1] / x[0]) / (2 * Math.PI) + (x[0] < 0 ? 0.5 : 0);
            return [10 * (x[2] - 10 * angle), 10 * (Math.Sqrt(x[0] * x[0] + x[1] * x[1]) - 1), x[2]];
        }
        double[,] J(double[] x)
        {
            var squared = x[0] * x[0] + x[1] * x[1];
            return new[,] { { 50 / Math.PI * x[1] / squared, -50 / Math.PI * x[0] / squared, 10 },
                { 10 * x[0] / Math.Sqrt(squared), 10 * x[1] / Math.Sqrt(squared), 0 }, { 0, 0, 1 } };
        }
        var result = Solve(writer, R, [-1, 0, 0], J, 3);
        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(1, 1e-9);
        result.Parameters[1].Should().BeApproximately(0, 1e-9);
        result.Parameters[2].Should().BeApproximately(0, 1e-9);
        Math.Sqrt(2 * result.FinalCost).Should().BeLessThan(1e-9);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MoreBrownDennis_OriginalPositiveFourthStartReachesPrintedResidual(bool writer)
    {
        double[] R(double[] x) => Enumerable.Range(1, 20).Select(i =>
        {
            var t = i / 5.0; var a = x[0] + x[1] * t - Math.Exp(t); var b = x[2] + x[3] * Math.Sin(t) - Math.Cos(t);
            return a * a + b * b;
        }).ToArray();
        double[,] J(double[] x)
        {
            var matrix = new double[20, 4];
            for (var i = 0; i < 20; i++)
            {
                var t = (i + 1) / 5.0; var a = x[0] + x[1] * t - Math.Exp(t); var b = x[2] + x[3] * Math.Sin(t) - Math.Cos(t);
                matrix[i, 0] = 2 * a; matrix[i, 1] = 2 * a * t; matrix[i, 2] = 2 * b; matrix[i, 3] = 2 * b * Math.Sin(t);
            }
            return matrix;
        }
        // More's +1 differs from the later toolbox's -1. Printed norm is truncated to4decimals.
        var result = Solve(writer, R, [25, 5, -5, 1], J, 20);
        result.Converged.Should().BeTrue();
        Math.Sqrt(2 * result.FinalCost).Should().BeApproximately(292.9542, 1e-4);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MadsenPowell_OriginalStartApproachesExactZero(bool writer)
    {
        var result = Solve(writer, x => [x[0], 10 * x[0] / (x[0] + 0.1) + 2 * x[1] * x[1]], [3, 1],
            x => new double[,] { { 1, 0 }, { 1 / Math.Pow(x[0] + 0.1, 2), 4 * x[1] } }, 2);
        result.Converged.Should().BeTrue();
        result.Parameters[0].Should().BeApproximately(0, 1e-10);
        result.Parameters[1].Should().BeApproximately(0, 1e-5);
        Math.Sqrt(2 * result.FinalCost).Should().BeLessThan(1e-10);
    }

    [Theory]
    [InlineData(false, 1e-5)]
    [InlineData(true, 1e-5)]
    [InlineData(false, 1)]
    [InlineData(true, 1)]
    [InlineData(false, 100)]
    [InlineData(true, 100)]
    [InlineData(false, 10000)]
    [InlineData(true, 10000)]
    public void MadsenModifiedRosenbrock_NonzeroResidualRetainsMathematicalMinimum(bool writer, double constant)
    {
        var result = Solve(writer, x => [10 * (x[1] - x[0] * x[0]), 1 - x[0], constant], [-1.2, 1],
            x => new double[,] { { -20 * x[0], 10 }, { -1, 0 }, { 0, 0 } }, 3);
        result.Converged.Should().BeTrue();
        // MNT Example3.13 reports norm error2.37e-4 for identity LM at lambda1e4.
        // This mathematical-result bar does not claim its23iterations or hybrid accuracy.
        var error = Math.Sqrt(Math.Pow(result.Parameters[0] - 1, 2) + Math.Pow(result.Parameters[1] - 1, 2));
        error.Should().BeLessThan(2.5e-4);
        result.FinalCost.Should().BeApproximately(constant * constant / 2, Math.Max(1e-18, constant * constant * 1e-12));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Bard_RecoveredAuthorObservationsReachPublishedResidual(bool writer)
    {
        double[] y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39];
        double[] R(double[] x) => Enumerable.Range(1, 15).Select(i => y[i - 1] - x[0] - i / (x[1] * (16 - i) + x[2] * Math.Min(i, 16 - i))).ToArray();
        double[,] J(double[] x)
        {
            var matrix = new double[15, 3];
            for (var i = 1; i <= 15; i++)
            {
                var v = 16 - i; var w = Math.Min(i, v); var d = x[1] * v + x[2] * w;
                matrix[i - 1, 0] = -1; matrix[i - 1, 1] = i * v / (d * d); matrix[i - 1, 2] = i * w / (d * d);
            }
            return matrix;
        }
        var result = Solve(writer, R, [1, 1, 1], J, 15);
        result.Converged.Should().BeTrue();
        Math.Sqrt(2 * result.FinalCost).Should().BeApproximately(0.0906359, 1e-7);
        double[] author = [0.08241055996, 1.133036099, 2.343695172];
        for (var i = 0; i < 3; i++) { result.Parameters[i].Should().BeApproximately(author[i], 1e-5); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MadsenNoisyExponential_AuthenticAuthorDataReachesPublishedFit(bool writer)
    {
        // Nielsen efit1.dat (1999-03-10), identical to immoptibox uctpget.m hbnefit.
        double[] y = [0.090542, 0.124569, 0.179367, 0.195654, 0.269707, 0.286027, 0.289892, 0.317475, 0.308191, 0.336995, 0.348371, 0.321337, 0.299423, 0.338972, 0.304763, 0.288903, 0.300820, 0.303974, 0.283987, 0.262078, 0.281593, 0.267531, 0.218926, 0.225572, 0.200594, 0.197375, 0.182440, 0.183892, 0.152285, 0.174028, 0.150874, 0.126220, 0.126266, 0.106384, 0.118923, 0.091868, 0.128926, 0.119273, 0.115997, 0.105831, 0.075261, 0.068387, 0.090823, 0.085205, 0.067203];
        double[] R(double[] x) => Enumerable.Range(1, 45).Select(i =>
            x[2] * Math.Exp(x[0] * i * 0.02) + x[3] * Math.Exp(x[1] * i * 0.02) - y[i - 1]).ToArray();
        double[,] J(double[] x)
        {
            var matrix = new double[45, 4];
            for (var i = 0; i < 45; i++)
            {
                var t = (i + 1) * 0.02; var e1 = Math.Exp(x[0] * t); var e2 = Math.Exp(x[1] * t);
                matrix[i, 0] = t * x[2] * e1; matrix[i, 1] = t * x[3] * e2; matrix[i, 2] = e1; matrix[i, 3] = e2;
            }
            return matrix;
        }
        var result = Solve(writer, R, [-1, -2, 1, -1], J, 45);
        result.Converged.Should().BeTrue();
        double[] published = [-4, -5, 4, -4];
        for (var i = 0; i < 4; i++) { result.Parameters[i].Should().BeApproximately(published[i], 1e-3); }
        Math.Sqrt(2 * result.FinalCost).Should().BeApproximately(0.1, 1e-6);
    }

    private static MultivariateSolverResult<double> Solve(bool writer, Func<double[], double[]> residual,
        double[] start, Func<double[], double[,]> jacobian, int count)
    {
        var solver = new LevenbergMarquardtSolver<double>(10000, 1e-16, 1e-14, 1e-14, 0.001);
        if (!writer) { return solver.Solve(residual, start, jacobian); }
        void W(ReadOnlySpan<double> x, Span<double> r) => residual(x.ToArray()).CopyTo(r);
        void D(ReadOnlySpan<double> x, Span<double> j)
        {
            var matrix = jacobian(x.ToArray());
            for (var row = 0; row < count; row++)
            {
                for (var col = 0; col < start.Length; col++) { j[row * start.Length + col] = matrix[row, col]; }
            }
        }
        return solver.Solve(W, start, count, D);
    }
}
