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
/// Range-safety regressions for the quadratic inverse shrinkage and POET covariance estimators:
/// every case has a representable sample covariance and a representable correct result, yet the
/// pre-fix arithmetic left the type's range (or underflowed a small column) on the way.
/// </summary>
public sealed class Arena414074R1QisPoetTests
{
    // ---------------------------------------------------------------- QIS ----

    [Fact]
    public void Qis_IsotropicPanelNearRangeTop_FinalRescaleStaysRepresentable()
    {
        // F-L23-C4-02. Sample covariance = 1.5e308 * I (each column has demeaned sum of
        // squares 2 * 1.5e308 over n = 2). QIS of an isotropic matrix is that matrix. The
        // trace-preserving ratio lambdaSum / deltaSum is 1 / h^2 > 1, and multiplying it into the
        // physical scale before use left the range of double even though every output entry fits.
        var a = Math.Sqrt(1.5e308);
        var b = Math.Sqrt(5e307);
        double[,] returns = { { -a, -b }, { 0, 2 * b }, { a, -b } };

        var sample = new SampleCovarianceEstimator<double>().Estimate(returns);
        var actual = new QuadraticInverseShrinkageEstimator<double>().Estimate(returns);

        actual.Cast<double>().Should().OnlyContain(v => double.IsFinite(v), "the isotropic result fits in double");
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                actual[i, j].Should().BeApproximately(sample[i, j], 1e-12 * 1.5e308);
            }
        }
    }

    [Fact]
    public void Qis_IsotropicPanelNearRangeTop_Float_FinalRescaleStaysRepresentable()
    {
        // F-L23-C4-02, float instantiation: sample covariance = 3.0e38 * I.
        var a = MathF.Sqrt(3.0e38f);
        var b = MathF.Sqrt(1.0e38f);
        float[,] returns = { { -a, -b }, { 0, 2 * b }, { a, -b } };

        var sample = new SampleCovarianceEstimator<float>().Estimate(returns);
        var actual = new QuadraticInverseShrinkageEstimator<float>().Estimate(returns);

        actual.Cast<float>().Should().OnlyContain(v => float.IsFinite(v), "the isotropic result fits in float");
        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                actual[i, j].Should().BeApproximately(sample[i, j], 1e-5f * 3.0e38f);
            }
        }
    }

    [Fact]
    public void Qis_LeadingEigenvalueAboveRange_ReconstructsRepresentableMatrix_Double()
    {
        // F-L1-13. The 2^511-scaled panel has sample entries <= 1.065e308 but leading eigenvalue
        // 3.06e308 > double.MaxValue. QIS is degree-2 homogeneous and power-of-two scaling is exact,
        // so the result is the normal-scale result times 2^1022; restoring the physical scale per
        // eigenvalue (before reconstruction) overflowed all nine entries.
        var panel = CorrelatedPanel();
        var baseResult = new QuadraticInverseShrinkageEstimator<double>().Estimate(panel);

        var scaledPanel = ScaleDouble(panel, 511);
        var sample = new SampleCovarianceEstimator<double>().Estimate(scaledPanel);
        var actual = new QuadraticInverseShrinkageEstimator<double>().Estimate(scaledPanel);

        actual.Cast<double>().Should().OnlyContain(v => double.IsFinite(v), "every correct entry is below 1.07e308");
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var expected = Math.ScaleB(baseResult[i, j], 1022);
                actual[i, j].Should().BeApproximately(expected, 1e-12 * Math.Abs(expected));
            }
        }

        // Independent invariant stated by the algorithm: the trace of the sample matrix is preserved.
        var actualTrace = 0.0;
        var sampleTrace = 0.0;
        for (var i = 0; i < 3; i++)
        {
            actualTrace += Math.ScaleB(actual[i, i], -1022);
            sampleTrace += Math.ScaleB(sample[i, i], -1022);
        }

        actualTrace.Should().BeApproximately(sampleTrace, 1e-12 * sampleTrace);

        // Literal anchors from the finding's evidence (normal-scale result times 2^1022).
        actual[0, 0].Should().BeApproximately(1.0025421775662703E+308, 1e-12 * 1.0025421775662703E+308);
        actual[1, 1].Should().BeApproximately(1.0649775196010143E+308, 1e-12 * 1.0649775196010143E+308);
    }

    [Fact]
    public void Qis_LeadingEigenvalueAboveRange_ReconstructsRepresentableMatrix_Float()
    {
        // F-L1-13, float: entries <= 2.0e38 while the leading eigenvalue is 5.8e38 > float.MaxValue.
        var panel = CorrelatedPanel();
        var floatPanel = new float[panel.GetLength(0), panel.GetLength(1)];
        var scaledPanel = new float[panel.GetLength(0), panel.GetLength(1)];
        for (var i = 0; i < panel.GetLength(0); i++)
        {
            for (var j = 0; j < panel.GetLength(1); j++)
            {
                floatPanel[i, j] = (float)panel[i, j];
                scaledPanel[i, j] = MathF.ScaleB((float)panel[i, j], 63);
            }
        }

        var baseResult = new QuadraticInverseShrinkageEstimator<float>().Estimate(floatPanel);
        var actual = new QuadraticInverseShrinkageEstimator<float>().Estimate(scaledPanel);

        actual.Cast<float>().Should().OnlyContain(v => float.IsFinite(v), "every correct entry is below 2.1e38");
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var expected = MathF.ScaleB(baseResult[i, j], 126);
                actual[i, j].Should().BeApproximately(expected, 1e-5f * MathF.Abs(expected));
            }
        }
    }

    [Fact]
    public void Qis_LeadingEigenvalueAboveRange_ReconstructsRepresentableMatrix_Decimal()
    {
        // F-L1-13, decimal: entries <= 4.65e28 while the leading eigenvalue is 1.34e29 >
        // decimal.MaxValue (decimal overflows with an exception rather than infinity).
        var panel = CorrelatedPanel();
        var basePanel = new decimal[10, 3];
        var scaledPanel = new decimal[10, 3];
        const decimal factor = 140000000000000m;
        for (var i = 0; i < 10; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                basePanel[i, j] = (decimal)panel[i, j];
                scaledPanel[i, j] = basePanel[i, j] * factor;
            }
        }

        var baseResult = new QuadraticInverseShrinkageEstimator<decimal>().Estimate(basePanel);

        decimal[,]? actual = null;
        var estimate = () => actual = new QuadraticInverseShrinkageEstimator<decimal>().Estimate(scaledPanel);
        estimate.Should().NotThrow("every correct entry is below 4.7e28");

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var expected = baseResult[i, j] * factor * factor;
                ((double)actual![i, j]).Should().BeApproximately((double)expected, 1e-12 * (double)Math.Abs(expected));
            }
        }
    }

    // --------------------------------------------------------------- POET ----

    [Fact]
    public void Poet_TopEigenvalueAboveRange_ReturnsRepresentableSampleCovariance_Double()
    {
        // F-L1-10. S = [[9.36e307 x 2x2 block], [0, 0, 4/3]] is representable, the exact POET
        // output with K = 1 is S itself (factor = the rank-one block, residual diagonal), but the
        // top eigenvalue is 1.04 * 2^1024 and the unscaled Jacobi step refused the panel. S also
        // has entries above MaxValue/2, so the final symmetrization must not add two such entries.
        var a = Math.ScaleB(1.25, 511);
        double[,] returns = { { a, a, 1 }, { -a, -a, 1 }, { a, a, -1 }, { -a, -a, -1 } };
        var sample = new SampleCovarianceEstimator<double>().Estimate(returns);
        var scale = sample.Cast<double>().Max(Math.Abs);

        foreach (var multiplier in new[] { 0.0, 0.5 })
        {
            double[,]? actual = null;
            var estimate = () => actual = new PoetCovarianceEstimator<double>(1, multiplier).Estimate(returns);
            estimate.Should().NotThrow($"the exact K=1 output is representable (multiplier {multiplier})");
            for (var i = 0; i < 3; i++)
            {
                for (var j = 0; j < 3; j++)
                {
                    actual![i, j].Should().BeApproximately(sample[i, j], 1e-12 * scale);
                }
            }
        }
    }

    [Fact]
    public void Poet_TopEigenvalueAboveRange_ReturnsRepresentableSampleCovariance_Float()
    {
        // F-L1-10, float: S00 = 1.77e38 is representable but the top eigenvalue 3.55e38 > float.MaxValue.
        var a = MathF.ScaleB(1.25f, 63);
        float[,] returns = { { a, a, 1 }, { -a, -a, 1 }, { a, a, -1 }, { -a, -a, -1 } };
        var sample = new SampleCovarianceEstimator<float>().Estimate(returns);
        var scale = sample.Cast<float>().Max(MathF.Abs);

        float[,]? actual = null;
        var estimate = () => actual = new PoetCovarianceEstimator<float>(1, 0f).Estimate(returns);
        estimate.Should().NotThrow("the exact K=1 output is representable");
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                actual![i, j].Should().BeApproximately(sample[i, j], 1e-5f * scale);
            }
        }
    }

    [Fact]
    public void Poet_TopEigenvalueAboveRange_ReturnsRepresentableSampleCovariance_Decimal()
    {
        // F-L1-10, decimal: S00 = 4.2008e28 is representable, the top eigenvalue 8.4e28 is above
        // decimal.MaxValue (an OverflowException wrapped by the Jacobi step).
        const decimal a = 177500000000000m;
        decimal[,] returns = { { a, a, 1 }, { -a, -a, 1 }, { a, a, -1 }, { -a, -a, -1 } };
        var sample = new SampleCovarianceEstimator<decimal>().Estimate(returns);

        decimal[,]? actual = null;
        var estimate = () => actual = new PoetCovarianceEstimator<decimal>(1, 0m).Estimate(returns);
        estimate.Should().NotThrow("the exact K=1 output is representable");

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                ((double)actual![i, j]).Should().BeApproximately((double)sample[i, j], 1e-14 * 4.2008333333333333e28);
            }
        }
    }

    [Fact]
    public void Poet_SmallColumnBlock_AdaptiveThresholdKeepsItsOwnScale_Double()
    {
        // F-L1-12. Columns A,B (scale 2^-240) and C,D (scale 2^330) live on disjoint rows with
        // exactly zero means, so the K = 1 factor sits in the C,D block and the residual (A,B)
        // entry is S_AB = -1.6713e-145. Its adaptive threshold is 2.7668e-145 (FLM eq. 3.2, high
        // precision oracle): the entry must be thresholded to zero. Normalizing every residual
        // by the global maximum underflowed theta_AB to 0, so the threshold was 0 and the entry
        // survived.
        var panel = DecoupledPanel(Math.ScaleB(1.0, -240), Math.ScaleB(1.0, 330));

        var thresholded = new PoetCovarianceEstimator<double>(1, 0.5).Estimate(panel);
        var untouched = new PoetCovarianceEstimator<double>(1, 0.0).Estimate(panel);

        thresholded[0, 1].Should().Be(0.0, "|S_AB| = 1.6713e-145 < tau_AB = 2.7668e-145");
        thresholded[1, 0].Should().Be(0.0);
        thresholded[0, 0].Should().BeApproximately(1.0306375585638224E-144, 1e-12 * 1.0306375585638224E-144);
        thresholded[1, 1].Should().BeApproximately(5.29246313857098E-145, 1e-12 * 5.29246313857098E-145);

        // Control: with no thresholding the same entry is the retained residual S_AB.
        untouched[0, 1].Should().BeApproximately(-1.6713041490224147E-145, 1e-12 * 1.6713041490224147E-145);
    }

    [Fact]
    public void Poet_SmallColumnBlock_AdaptiveThresholdKeepsItsOwnScale_Decimal()
    {
        // F-L1-12, decimal analogue: A,B = 1e-12 * {U,W}, C,D = 1e8 * {V,V+S}; tau_AB = 8.637e-25
        // exceeds |S_AB| = 5.217e-25, so the entry is zero. The pre-fix global scale left tau = 0.
        var panel = DecoupledPanelDecimal(1e-12m, 1e8m);

        var thresholded = new PoetCovarianceEstimator<decimal>(1, 0.5m).Estimate(panel);
        var untouched = new PoetCovarianceEstimator<decimal>(1, 0m).Estimate(panel);

        thresholded[0, 1].Should().Be(0m, "|S_AB| = 5.217e-25 < tau_AB = 8.637e-25");
        thresholded[1, 0].Should().Be(0m);

        // Control: without thresholding the entry is the retained residual S_AB.
        untouched[0, 1].Should().BeApproximately(-5.217e-25m, 1e-28m);
    }

    // ------------------------------------------------------------ helpers ----

    private static double[,] CorrelatedPanel()
    {
        // Highly correlated 10 x 3 panel (the finding's witness). Entries are small integers plus
        // multiples of 1/16, so they are exact in double, float and decimal.
        var panel = new double[10, 3];
        for (var i = 0; i < 10; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                panel[i, j] = ((i * 7) % 5 - 2) + (0.0625 * ((((i * (j + 3)) + j) % 5) - 2));
            }
        }

        return panel;
    }

    private static double[,] ScaleDouble(double[,] panel, int exponent)
    {
        var scaled = new double[panel.GetLength(0), panel.GetLength(1)];
        for (var i = 0; i < scaled.GetLength(0); i++)
        {
            for (var j = 0; j < scaled.GetLength(1); j++)
            {
                scaled[i, j] = Math.ScaleB(panel[i, j], exponent);
            }
        }

        return scaled;
    }

    private static readonly int[] s_u = [3, -1, 2, -4, 1, 0, -2, 5, -3, 1, -2];
    private static readonly int[] s_w = [1, 2, -3, 1, -2, 3, -1, 0, 2, -1, -2];
    private static readonly int[] s_v = [2, -3, 1, 4, -2, -1, 3, -4, 0, 2, -2];
    private static readonly int[] s_s = [1, 0, -1, 1, -1, 0, 1, -1, 0, 1, -1];

    // 24 x 4 panel: A,B live on rows 1..11 and C,D on rows 13..23, rows 0 and 12 are zero, and
    // every column has an exactly zero mean, so the cross-block covariances are exactly zero.
    private static double[,] DecoupledPanel(double small, double large)
    {
        var panel = new double[24, 4];
        for (var k = 0; k < 11; k++)
        {
            panel[1 + k, 0] = small * s_u[k];
            panel[1 + k, 1] = small * s_w[k];
            panel[13 + k, 2] = large * s_v[k];
            panel[13 + k, 3] = large * (s_v[k] + s_s[k]);
        }

        return panel;
    }

    private static decimal[,] DecoupledPanelDecimal(decimal small, decimal large)
    {
        var panel = new decimal[24, 4];
        for (var k = 0; k < 11; k++)
        {
            panel[1 + k, 0] = small * s_u[k];
            panel[1 + k, 1] = small * s_w[k];
            panel[13 + k, 2] = large * s_v[k];
            panel[13 + k, 3] = large * (s_v[k] + s_s[k]);
        }

        return panel;
    }
}
