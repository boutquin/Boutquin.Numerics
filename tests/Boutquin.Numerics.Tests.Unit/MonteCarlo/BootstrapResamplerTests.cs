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

public sealed class BootstrapResamplerTests
{
    [Fact]
    public void Constructor_RejectsNonPositiveBlockSize()
    {
        Action zero = () => BootstrapResampler.FromSeed(0);
        Action negative = () => BootstrapResampler.FromSeed(-5);
        zero.Should().Throw<ArgumentOutOfRangeException>();
        negative.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Resample_ProducesSameLengthAsInput()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m, 10m];
        var result = BootstrapResampler.FromSeed(blockSize: 3, 1).Resample(source);
        result.Length.Should().Be(source.Length);
    }

    [Fact]
    public void Resample_DrawsOnlyFromSource()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m];
        var result = BootstrapResampler.FromSeed(blockSize: 2, 7).Resample(source);

        var allowed = new HashSet<decimal>(source);
        foreach (var v in result)
        {
            allowed.Should().Contain(v);
        }
    }

    [Fact]
    public void Resample_DeterministicWithSeed()
    {
        decimal[] source = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m];
        var a = BootstrapResampler.FromSeed(3, 42).Resample(source);
        var b = BootstrapResampler.FromSeed(3, 42).Resample(source);

        a.Should().Equal(b);
    }

    [Fact]
    public void Resample_EmptyInput_Throws()
    {
        var resampler = BootstrapResampler.FromSeed(2, 0);
        Action act = () => resampler.Resample([]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ResamplePaired_KeepsSeriesIndexAligned()
    {
        // Construct paired series where B is always A + 100. Any legitimate
        // paired resample must preserve that relationship.
        decimal[] a = [1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m];
        decimal[] b = [101m, 102m, 103m, 104m, 105m, 106m, 107m, 108m];

        var (ra, rb) = BootstrapResampler.FromSeed(3, 99).ResamplePaired(a, b);

        ra.Length.Should().Be(a.Length);
        rb.Length.Should().Be(b.Length);
        for (var i = 0; i < ra.Length; i++)
        {
            rb[i].Should().Be(ra[i] + 100m);
        }
    }

    [Fact]
    public void ResamplePaired_RejectsMismatchedLengths()
    {
        decimal[] a = [1m, 2m, 3m];
        decimal[] b = [1m, 2m];
        var resampler = BootstrapResampler.FromSeed(2);
        Action act = () => resampler.ResamplePaired(a, b);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Resample_LargeBlock_WrapsCircularly()
    {
        // block size > source length → circular wrap around source.
        decimal[] source = [1m, 2m, 3m];
        var result = BootstrapResampler.FromSeed(10, 55).Resample(source);

        result.Length.Should().Be(3);
        foreach (var v in result)
        {
            source.Should().Contain(v);
        }
    }
}
