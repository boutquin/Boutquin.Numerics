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

using System.Text.Json;

using Boutquin.Numerics.Statistics;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Statistics;

public sealed class RmtPaperReplicationTests
{
    [Fact]
    public void Figure22_SeededBookExperiment_Recovers100Factors()
    {
        // López de Prado (2020), snippets 2.3–2.4: 10,000 observations,
        // 1,000 variables, 100 factors, alpha=.995, q=10, KDE bandwidth=.01.
        // The author gives no seed. This declared realization uses seed 2026;
        // NumPy computes its spectrum and SciPy independently fits the primary
        // MP/KDE objective (tests/Verification/generate_rmt_paper_vectors.py).
        var path = Path.Combine(AppContext.BaseDirectory, "Verification", "vectors", "paper_rmt.json");
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var reference = document.RootElement;
        var spectrum = reference.GetProperty("eigenvalues").EnumerateArray().Select(value => value.GetDouble()).ToArray();
        var q = reference.GetProperty("q").GetDouble();
        var expectedVariance = reference.GetProperty("fitted_noise_variance").GetDouble();
        var expectedEdge = reference.GetProperty("mp_upper_edge").GetDouble();

        spectrum.Should().HaveCount(1000);
        spectrum.Sum().Should().BeApproximately(1000, 1e-9, "a correlation matrix has unit diagonal");
        spectrum.Should().BeInDescendingOrder();
        reference.GetProperty("observations").GetInt32().Should().Be(10000);
        reference.GetProperty("factors").GetInt32().Should().Be(100);
        reference.GetProperty("alpha").GetDouble().Should().Be(0.995);
        reference.GetProperty("bandwidth").GetDouble().Should().Be(0.01);
        q.Should().Be(10);

        var variance = CovarianceHelpers<double>.FitNoiseVariance(spectrum, q);
        var edge = variance * CovarianceHelpers<double>.MarcenkoPasturUpperBound(q);

        variance.Should().BeApproximately(expectedVariance, 1e-8, "the independent dense scan and SciPy fit locate the global minimum");
        edge.Should().BeApproximately(expectedEdge, 2e-8);
        spectrum.Count(value => value > edge).Should().Be(100, "the book's injected factors are recovered");
        reference.GetProperty("signal_count").GetInt32().Should().Be(100);
    }
}
