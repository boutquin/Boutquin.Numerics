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

/// <summary>
/// Fidelity pins for the random-matrix-theory cleaning estimators
/// (<see cref="DenoisedCovarianceEstimator{T}"/>, <see cref="DetonedCovarianceEstimator{T}"/>,
/// <see cref="TracyWidomDenoisedCovarianceEstimator{T}"/>).
/// </summary>
/// <remarks>
/// Every panel is built so that its sample correlation matrix equals a chosen one-factor
/// target <c>rho_ij = beta_i . beta_j</c> exactly (centered, mutually orthogonal
/// innovations scaled by a Cholesky factor). The expected correlation matrices were computed
/// in NumPy from López de Prado's snippets 2.4–2.6 as implemented by emoen,
/// <em>Machine-Learning-for-Asset-Managers</em> (<c>cov2corr</c>, <c>getPCA</c>,
/// <c>denoisedCorr</c>, <c>detoned_corr</c>, copied verbatim), on the same target —
/// they are not read back from this library's output.
/// </remarks>
public sealed class RmtCleaningFidelityTests
{
    private const int Observations = 120;

    private static readonly double[] s_strongBetas = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    private static readonly double[] s_stdDevs = [0.010, 0.015, 0.020, 0.012, 0.018, 0.025];

    [Fact]
    public void Detoned_DefaultConstructor_AppliesFullDetoning()
    {
        var returns = BuildPanel(s_strongBetas, Observations, seed: 11);

        var byDefault = new DetonedCovarianceEstimator<double>().Estimate(returns);
        var explicitFull = new DetonedCovarianceEstimator<double>(1.0).Estimate(returns);

        MaxAbsDifference(byDefault, explicitFull).Should().Be(0.0,
            "the documented default is alpha = 1 (full detoning), matching the decimal facade's 1m default");
    }

    // denoisedCorr(eVal, eVec, nFacts = 1) on the strong one-factor target, q = 120 / 6:
    // lambda = [2.566, 0.898, 0.817, 0.712, 0.582, 0.425], MP edge 1.4972, and the
    // reconstruction is rescaled to a unit diagonal by cov2corr (upper triangle, row-major).
    private static readonly double[] s_denoisedStrongGolden =
    [
        0.9999999999999998, 0.3998140235478781, 0.3778581598490972, 0.34727737950468307, 0.3055275951367473, 0.25001076112616455,
        0.9999999999999998, 0.36381687654738815, 0.334372483996147, 0.29417412980065877, 0.24072031222636991,
        1.0, 0.3160103549790205, 0.27801950110530693, 0.22750111016369617,
        0.9999999999999999, 0.25551885351267395, 0.20908901214044553,
        1.0, 0.18395227221509058,
        1.0,
    ];

    [Fact]
    public void Denoised_RescalesReconstructionToUnitDiagonal_MatchingCov2Corr()
    {
        var returns = BuildPanel(s_strongBetas, Observations, seed: 17);

        var corr = ToCorrelation(new DenoisedCovarianceEstimator<double>().Estimate(returns));

        MaxAbsDifferenceUpper(corr, s_denoisedStrongGolden).Should().BeLessThan(1e-10,
            "snippet 2.5 rescales V.diag(lambda').V^T with cov2corr instead of overwriting its diagonal with 1");
    }

    [Fact]
    public void TracyWidom_RescalesReconstructionToUnitDiagonal_MatchingCov2Corr()
    {
        // lambda_1 = 2.566 lies far above the Tracy-Widom threshold (1.666 at t = 120, n = 6)
        // and lambda_2 = 0.898 far below it, so the signal/noise split is the same one-factor
        // split as the Marcenko-Pastur golden and only the diagonal step is under test.
        var returns = BuildPanel(s_strongBetas, Observations, seed: 19);

        var corr = ToCorrelation(new TracyWidomDenoisedCovarianceEstimator<double>().Estimate(returns));

        MaxAbsDifferenceUpper(corr, s_denoisedStrongGolden).Should().BeLessThan(1e-10,
            "the cleaned reconstruction is rescaled with cov2corr, not diagonal-overwritten");
    }

    // Johnstone (2001) at t = 120 observations, n = 6 variables:
    //   mu_np / t    = (sqrt(119) + sqrt(6))^2 / 120                       = 1.48683
    //   sigma_np / t = (sqrt(119) + sqrt(6)) (1/sqrt(119) + 1/sqrt(6))^(1/3) / 120
    //   threshold    = mu_np / t + (sigma_np / t) . F1^-1(0.99), F1^-1(0.99) = 2.0234  -> 1.66578
    // Centering on the MP edge instead of mu_np / t gives 1.67598, and multiplying the
    // quantile by |E[TW1]| = 1.21 gives 1.71316; lambda_1 = 1.67088 sits between the
    // correct threshold and both of those.
    private static readonly double[] s_twWindowBetas = [0.523595, 0.458146, 0.392696, 0.327247, 0.261798, 0.196348];

    private static readonly double[] s_twWindowGolden =
    [
        0.9999999999999998, 0.18463144904482312, 0.17140074210378167, 0.1540548276398625, 0.13204104139215828, 0.10506521877440422,
        1.0, 0.16280416174665518, 0.14632822920770927, 0.12541854134440963, 0.09979568735436921,
        1.0, 0.13584233458972794, 0.11643103691823924, 0.09264431904633447,
        0.9999999999999999, 0.10464810772819928, 0.08326862782107883,
        0.9999999999999999, 0.07136989149404202,
        1.0,
    ];

    // lambda_1 = 1.65378, just below the 99% Tracy-Widom threshold 1.66578.
    private static readonly double[] s_twBelowBetas = [0.516879, 0.452269, 0.387659, 0.323049, 0.25844, 0.19383];

    [Fact]
    public void TracyWidom_ThresholdIsMuNpOverTPlusSigmaNpOverTTimesTheTw1NinetyNinthPercentile()
    {
        var returns = BuildPanel(s_twWindowBetas, Observations, seed: 23);

        var corr = ToCorrelation(new TracyWidomDenoisedCovarianceEstimator<double>().Estimate(returns));

        MaxAbsDifferenceUpper(corr, s_twWindowGolden).Should().BeLessThan(1e-10,
            "lambda_1 = 1.67088 exceeds the Johnstone 99% threshold 1.66578, so it is signal and the rest are averaged");
    }

    [Fact]
    public void TracyWidom_EigenvalueBelowTheNinetyNinthPercentileThreshold_IsNoise()
    {
        var returns = BuildPanel(s_twBelowBetas, Observations, seed: 29);

        var sample = new SampleCovarianceEstimator<double>().Estimate(returns);
        var tw = new TracyWidomDenoisedCovarianceEstimator<double>().Estimate(returns);

        MaxAbsDifference(ToCorrelation(tw), ToCorrelation(sample)).Should().BeLessThan(1e-9,
            "lambda_1 = 1.65378 is below the threshold, so the all-noise spectrum passes through unchanged");
    }

    [Fact]
    public void Detoned_AllNoiseSpectrum_AveragesWithoutMarketRemoval()
    {
        // q=1 has a singular endpoint density, hence the documented failed-fit
        // variance of 1. Eigenvalues (1.5, 1.5, 0) are all below its edge of 4.
        // The previous weak-factor fixture is not all-noise after fitting variance.
        double[,] returns = { { 1, -1, 0 }, { 0, 1, -1 }, { -1, 0, 1 } };

        var sample = new SampleCovarianceEstimator<double>().Estimate(returns);
        DenoisedCovarianceEstimator<double>.DenoiseCorrelation(sample, 3).SignalCount.Should().Be(0);
        var detoned = new DetonedCovarianceEstimator<double>(1.0).Estimate(returns);

        MaxAbsDifference(ToCorrelation(detoned), new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }).Should().BeLessThan(1e-9,
            "snippet 2.5 averages every residual eigenvalue even when no market component qualifies");
    }

    // corr1 = denoisedCorr(...); eVal1, eVec1 = getPCA(corr1); detoned_corr(corr1, eVal1, eVec1):
    // the market component lambda_1 . v_1 . v_1^T of the denoised correlation is subtracted and
    // the remainder rescaled with cov2corr. The result is singular (min eigenvalue 2.8e-15).
    private static readonly double[] s_detonedStrongGolden =
    [
        1.0, -0.21315168875412208, -0.21627923929524404, -0.21584600870570964, -0.2074548964968825, -0.18500881944207304,
        1.0, -0.2182567500916197, -0.21649466762675795, -0.20679861418210524, -0.18338054048356384,
        1.0, -0.21540580361065495, -0.2042062191863684, -0.17980872362268266,
        1.0, -0.19767795134781704, -0.17263786452855923,
        1.0, -0.15934798639584255,
        1.0,
    ];

    [Fact]
    public void Detoned_FullDetoning_RemovesTheMarketComponentAndRescales()
    {
        var returns = BuildPanel(s_strongBetas, Observations, seed: 37);

        var corr = ToCorrelation(new DetonedCovarianceEstimator<double>(1.0).Estimate(returns));

        // Detoning decomposes the denoised correlation a second time; the Jacobi solver stops at an
        // off-diagonal norm of sqrt(1e-15) . ||A||, so v_1 carries ~1e-9 error that the rescale by the
        // small residual diagonal amplifies. A structural error is of order 1e-1.
        MaxAbsDifferenceUpper(corr, s_detonedStrongGolden).Should().BeLessThan(1e-7,
            "section 2.6 subtracts the first eigenpair of the denoised correlation and rescales with cov2corr");
    }

    internal static double[,] BuildPanel(double[] betas, int t, int seed)
    {
        var n = betas.Length;
        var target = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                target[i, j] = i == j ? 1.0 : betas[i] * betas[j];
            }
        }

        return BuildPanelFromCorrelation(target, t, seed);
    }

    internal static double[,] BuildPanelFromCorrelation(double[,] target, int t, int seed)
    {
        var n = target.GetLength(0);
        var rng = new System.Random(seed);

        // Centered, mutually orthogonal columns with Z^T Z = (t - 1) I.
        var z = new double[t, n];
        for (var i = 0; i < t; i++)
        {
            for (var j = 0; j < n; j++)
            {
                z[i, j] = rng.NextDouble() - 0.5;
            }
        }

        for (var j = 0; j < n; j++)
        {
            var mean = 0.0;
            for (var i = 0; i < t; i++)
            {
                mean += z[i, j];
            }

            mean /= t;
            for (var i = 0; i < t; i++)
            {
                z[i, j] -= mean;
            }

            for (var k = 0; k < j; k++)
            {
                var dot = 0.0;
                for (var i = 0; i < t; i++)
                {
                    dot += z[i, j] * z[i, k];
                }

                for (var i = 0; i < t; i++)
                {
                    z[i, j] -= dot * z[i, k];
                }
            }

            var norm = 0.0;
            for (var i = 0; i < t; i++)
            {
                norm += z[i, j] * z[i, j];
            }

            norm = Math.Sqrt(norm);
            for (var i = 0; i < t; i++)
            {
                z[i, j] /= norm;
            }
        }

        // Cholesky factor of Sigma = S . target . S.
        var sigma = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                sigma[i, j] = target[i, j] * s_stdDevs[i % s_stdDevs.Length] * s_stdDevs[j % s_stdDevs.Length];
            }
        }

        var l = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = sigma[i, j];
                for (var k = 0; k < j; k++)
                {
                    sum -= l[i, k] * l[j, k];
                }

                l[i, j] = i == j ? Math.Sqrt(sum) : sum / l[j, j];
            }
        }

        var scale = Math.Sqrt(t - 1);
        var returns = new double[t, n];
        for (var r = 0; r < t; r++)
        {
            for (var i = 0; i < n; i++)
            {
                var sum = 0.0;
                for (var k = 0; k <= i; k++)
                {
                    sum += z[r, k] * l[i, k];
                }

                returns[r, i] = scale * sum;
            }
        }

        return returns;
    }

    internal static double[,] ToCorrelation(double[,] cov)
    {
        var n = cov.GetLength(0);
        var corr = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                corr[i, j] = cov[i, j] / Math.Sqrt(cov[i, i] * cov[j, j]);
            }
        }

        return corr;
    }

    internal static double MaxAbsDifference(double[,] a, double[,] b)
    {
        var max = 0.0;
        for (var i = 0; i < a.GetLength(0); i++)
        {
            for (var j = 0; j < a.GetLength(1); j++)
            {
                max = Math.Max(max, Math.Abs(a[i, j] - b[i, j]));
            }
        }

        return max;
    }

    internal static double MaxAbsDifferenceUpper(double[,] actual, double[] expectedUpper)
    {
        var n = actual.GetLength(0);
        var max = 0.0;
        var k = 0;
        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                max = Math.Max(max, Math.Abs(actual[i, j] - expectedUpper[k++]));
            }
        }

        return max;
    }
}
