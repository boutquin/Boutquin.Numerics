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

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class C4RmtRawCovarianceTests
{
    [Fact]
    public void Denoised_HeterogeneousSignal_RestoresExactOriginalVariances()
    {
        AssertRawCovariance(new DenoisedCovarianceEstimator<double>(), 0.3998140235478781, 1e-12);
    }

    [Fact]
    public void TracyWidom_HeterogeneousSignal_RestoresExactOriginalVariances()
    {
        AssertRawCovariance(new TracyWidomDenoisedCovarianceEstimator<double>(), 0.3998140235478781, 1e-12);
    }

    [Fact]
    public void Detoned_HeterogeneousSignal_RestoresExactOriginalVariances()
    {
        AssertRawCovariance(new DetonedCovarianceEstimator<double>(1.0), -0.21315168875412208, 1e-10);
    }

    private static void AssertRawCovariance(ICovarianceEstimator<double> estimator, double expectedCorrelation01, double tolerance)
    {
        // The existing builder creates centered orthogonal innovations with Z'Z=119I
        // and a triangular root of the declared covariance, so its variances are
        // the listed squared scales independently of any production covariance routine.
        double[] betas = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        double[] standardDeviations = [0.010, 0.015, 0.020, 0.012, 0.018, 0.025];
        var panel = RmtCleaningFidelityTests.BuildPanel(betas, 120, seed: 17);
        var actual = estimator.Estimate(panel);
        for (var i = 0; i < standardDeviations.Length; i++)
        {
            actual[i, i].Should().BeApproximately(standardDeviations[i] * standardDeviations[i], tolerance);
        }

        // MLAM snippets2.5–2.6 independently evaluated by NumPy on the declared
        // one-signal correlation target. Assert raw covariance; converting actual
        // back to correlation here would erase a missing production rescale.
        var expected01 = expectedCorrelation01 * standardDeviations[0] * standardDeviations[1];
        actual[0, 1].Should().BeApproximately(expected01, tolerance);
        actual[1, 0].Should().BeApproximately(expected01, tolerance);
    }
}
