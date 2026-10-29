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

using Boutquin.Numerics.MonteCarlo;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.MonteCarlo;

public sealed class WeightedBetaTailProductTests
{
    [Theory]
    [InlineData(330, 8.9542202521636598e-170, false)]
    [InlineData(330, 8.9542202521636598e-170, true)]
    [InlineData(340, 1.4214383297796813e-181, false)]
    [InlineData(340, 1.4214383297796813e-181, true)]
    [InlineData(360, 2.3197191277137582e-205, false)]
    [InlineData(360, 2.3197191277137582e-205, true)]
    public void HarrellDavis_SubnormalOrUnrepresentableMass_PreservesItsProduct(int count, double expected, bool negative)
    {
        var sorted = new double[count];
        sorted[negative ? 0 : count - 1] = negative ? -1e150 : 1e150;
        var actual = HarrellDavisPercentile.Compute<double>(sorted, 0.5);

        // Independent mpmath 200-digit oracle: 1e150 * I_(1/count)(a,a), a=(count+1)/2.
        (actual / (negative ? -expected : expected)).Should().BeApproximately(1.0, 2e-12);
    }

    [Theory]
    [InlineData(-1e150, 0.0)]
    [InlineData(-5e149, 7.1071916488984064e-182)]
    public void HarrellDavis_OppositeTailContributions_PreserveCancellation(double lower, double expected)
    {
        var sorted = new double[340];
        sorted[0] = lower;
        sorted[^1] = 1e150;
        var actual = HarrellDavisPercentile.Compute<double>(sorted, 0.5);

        // Equal beta shapes give identical endpoint masses; retain both signed products.
        actual.Should().BeApproximately(expected, 3e-193);
    }

    [Fact]
    public void HarrellDavis_AsymmetricBetaTail_RetainsUnrepresentableMass()
    {
        var sorted = new double[240];
        sorted[^1] = 1e150;
        var actual = HarrellDavisPercentile.Compute<double>(sorted, 0.3);

        // mpmath 150 digits, using the exact binary value of the double probability.
        (actual / 3.0994069704614281e-190).Should().BeApproximately(1.0, 2e-12);
    }

    [Fact]
    public void HarrellDavis_ManyTinyIntervals_PreserveTheirCombinedTail()
    {
        var sorted = new double[10000];
        Array.Fill(sorted, 1e150, 7000, 3000);
        var actual = HarrellDavisPercentile.Compute<double>(sorted, 0.5);

        // The 3000 interval masses telescope to I_0.3(5000.5,5000.5).
        // Each difference is formed from logarithmic tails, rather than rounded CDFs.
        (actual / 2.2759348136733931e-231).Should().BeApproximately(1.0, 1e-10);
    }
}
