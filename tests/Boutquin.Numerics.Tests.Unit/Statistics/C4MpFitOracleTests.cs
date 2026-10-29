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

public sealed class C4MpFitOracleTests
{
    [Fact]
    public void NoiseVarianceFit_IndependentDensityObjective_MatchesNumericalMinimum()
    {
        // Fixed positive spectrum with trace50:49 MP(var=.3,q10) midpoint
        // quantiles plus one signal. Input construction is independent of C#.
        // SciPy differential_evolution seeds17 and53 independently minimize the
        // book's1000-point GaussianKDE/MP squared error, bandwidth.01.
        double[] eigenvalues =
        [
            35.30094219142195,
            0.14801730495492527,
            0.15685308647706228,
            0.1641110145260418,
            0.17070252690317303,
            0.17691292587602309,
            0.1828782098307604,
            0.1886765014011056,
            0.19435796029217897,
            0.1999573300100019,
            0.2055000714767056,
            0.21100568995753355,
            0.21648968702252702,
            0.2219647765430229,
            0.2274416803410901,
            0.23292967066375603,
            0.23843695383307106,
            0.24397095035971408,
            0.24953850654077575,
            0.25514605877221874,
            0.26079976615163936,
            0.2665056204708169,
            0.27226954145369753,
            0.2780974626147774,
            0.2839954113337105,
            0.28996958742500223,
            0.2960264429346536,
            0.3021727660910485,
            0.3084157734918082,
            0.31476321307433425,
            0.32122348375454557,
            0.32780577486173085,
            0.334520236123467,
            0.3413781844028559,
            0.34839236208864965,
            0.35557726543092305,
            0.3629495696847385,
            0.37052868761696345,
            0.3783375257328087,
            0.38640352300485975,
            0.39476012837707686,
            0.40344896163660565,
            0.41252310301122325,
            0.4220523385839244,
            0.4321320342608712,
            0.44289933623928696,
            0.4545659850980771,
            0.4674956355535951,
            0.4824370929392033,
            0.5017200893534647,
        ];
        var actual = CovarianceHelpers<double>.FitNoiseVariance(eigenvalues, 10);
        actual.Should().BeApproximately(0.30148753784615595, 1e-7);
    }

    [Fact]
    public void SquareAspectRatio_EndpointDensityFailure_UsesDeclaredFallback()
    {
        // q1 makes lower edge0 and the endpoint MP density0/0.
        CovarianceHelpers<double>.FitNoiseVariance([1.5, 1.5, 0], 1).Should().Be(1);
    }
}
