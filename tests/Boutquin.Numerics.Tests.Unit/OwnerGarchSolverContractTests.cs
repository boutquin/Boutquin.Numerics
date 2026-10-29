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

using Boutquin.Numerics.Interpolation;
using Boutquin.Numerics.Solvers;
using Boutquin.Numerics.Statistics;

namespace Boutquin.Numerics.Tests.Unit;

public sealed class OwnerGarchSolverContractTests
{
    [Fact]
    public void PresampleUsesFittedResidualSecondMomentForVarianceAndShock()
    {
        double[] data = [1, 2, -1, 4, -2, 3];
        var fit = GarchOneOneEstimator<double>.Fit(data, new(MinObservations: 2, MaxIterations: 40));
        var seed = data.Sum(x => (x - fit.Mu) * (x - fit.Mu)) / data.Length;
        Assert.Equal(fit.Omega + (fit.Alpha + fit.Beta) * seed, fit.ConditionalVariances[0], 10);
    }

    [Fact]
    public void LikelihoodIsPerObservation()
    {
        double[] data = [1, 2, -1, 4, -2, 3];
        var fit = GarchOneOneEstimator<double>.Fit(data, new(MinObservations: 2, MaxIterations: 40));
        double average = fit.ConditionalVariances.Zip(fit.StandardizedResiduals, (h, z) => -0.5 * (Math.Log(h) + z * z)).Average();
        Assert.Equal(average, fit.LogLikelihood, 10);
    }

    [Fact]
    public void SelectedMeanAndTargetOptionsExist()
    {
        Assert.NotNull(typeof(GarchOptions<double>).GetProperty("FixedZeroMean"));
        Assert.NotNull(typeof(GarchOptions<double>).GetProperty("VarianceTargeting"));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void TargetingConstrainsInterceptAtEveryBudget(bool zeroMean, bool ar1)
    {
        double[] data = [1, 2, -1, 4, -2, 3];
        var center = zeroMean ? 0.0 : data.Average();
        var target = data.Sum(x => (x - center) * (x - center)) / data.Length;
        foreach (var budget in new[] { 2, 3, 7, 40 })
        {
            var fit = GarchOneOneEstimator<double>.Fit(data, new(MinObservations: 2, MaxIterations: budget, IncludeAr1Mean: ar1) { FixedZeroMean = zeroMean, VarianceTargeting = true });
            Assert.Equal((1 - fit.Alpha - fit.Beta) * target, fit.Omega, 12);
            var average = fit.ConditionalVariances.Zip(fit.StandardizedResiduals, (h, z) => -.5 * (Math.Log(h) + z * z)).Average();
            Assert.Equal(average, fit.LogLikelihood, 10);
            if (zeroMean) { Assert.Equal(0.0, fit.Mu); Assert.Equal(0.0, fit.Phi); }
        }
    }

    [Fact]
    public void FixedZeroMeanDoesNotEstimateIntercept()
    {
        var fit = GarchOneOneEstimator<double>.Fit([1, 2, -1, 4, -2, 3], new(MinObservations: 2, MaxIterations: 40) { FixedZeroMean = true });
        Assert.Equal(0.0, fit.Mu);
        Assert.Equal(0.0, fit.Phi);
    }

    [Fact]
    public void Ar1MeanIsStationaryAndResidualSeedUsesItsFittedInterceptAndLag()
    {
        double[] data = [1, 2, 4, 8, 16, 32, 64, 128];
        var fit = GarchOneOneEstimator<double>.Fit(data, new(MinObservations: 2, MaxIterations: 300, IncludeAr1Mean: true));
        Assert.True(Math.Abs(fit.Phi) < 1.0);
        var seed = data.Select((x, i) => Math.Pow(x - fit.Mu - fit.Phi * data[Math.Max(0, i - 1)], 2)).Average();
        Assert.Equal(fit.Omega + (fit.Alpha + fit.Beta) * seed, fit.ConditionalVariances[0], 9);
    }

    [Fact]
    public void ZeroMeanAndAr1AreMutuallyExclusive()
    {
        Assert.Throws<ArgumentException>(() => GarchOneOneEstimator<double>.Fit([1, 2, 3], new(MinObservations: 2, IncludeAr1Mean: true) { FixedZeroMean = true }));
    }

    [Fact]
    public void IterationCounterStopsAfterExceedingBudget()
    {
        var calls = 0;
        var result = NelderMeadMinimizer<double>.Minimize(x => { calls++; return -x[0]; }, [0.0], new(MaxIterations: 1, InitialStep: 1.0, XTolerance: 0.0, FTolerance: 0.0));
        Assert.Equal(2, result.Iterations);
        Assert.Equal(6, calls);
        Assert.Equal(7.0, result.Argmin[0]);
    }

    [Fact]
    public void GeneratedNonfiniteCoordinatesNeverReachObjective()
    {
        var bad = false;
        Assert.Throws<ArithmeticException>(() => NelderMeadMinimizer<double>.Minimize(x => { bad |= !double.IsFinite(x[0]); return -x[0] / double.MaxValue; }, [double.MaxValue * 0.8], new(MaxIterations: 1, InitialStep: double.MaxValue * 0.1, XTolerance: 0.0, FTolerance: 0.0)));
        Assert.False(bad);
    }

    [Theory]
    [InlineData(-1.0, -0.75)]
    [InlineData(3.0, 5.25)]
    public void ExteriorIntegratesEndpointForward(double x, double expected)
    {
        Assert.Equal(expected, MonotoneConvexInterpolator<double>.Instance.Interpolate(x, [0.0, 1.0, 2.0], [0.0, 1.0, 3.0]), 12);
    }
}
