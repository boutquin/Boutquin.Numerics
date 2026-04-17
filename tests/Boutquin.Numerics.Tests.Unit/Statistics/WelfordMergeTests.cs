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

public sealed class WelfordMergeTests
{
    [Fact]
    public void Merge_MatchesSequentialAccumulation()
    {
        var rng = new System.Random(88);
        var values = new decimal[1000];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = (decimal)(rng.NextDouble() * 10.0 - 5.0);
        }

        var serial = new WelfordMoments();
        foreach (var v in values)
        {
            serial.Add(v);
        }

        // Split roughly evenly.
        var a = new WelfordMoments();
        var b = new WelfordMoments();
        for (var i = 0; i < values.Length; i++)
        {
            if (i < values.Length / 3)
            {
                a.Add(values[i]);
            }
            else
            {
                b.Add(values[i]);
            }
        }

        a.Merge(b);

        a.Count.Should().Be(serial.Count);
        ((double)Math.Abs(a.Mean - serial.Mean)).Should().BeLessThan(1e-20);
        ((double)Math.Abs(a.Variance - serial.Variance)).Should().BeLessThan(1e-18);
    }

    [Fact]
    public void Merge_IntoEmpty_AdoptsOtherState()
    {
        var a = new WelfordMoments();
        var b = new WelfordMoments();
        b.Add(1m);
        b.Add(2m);
        b.Add(3m);

        a.Merge(b);
        a.Count.Should().Be(3);
        a.Mean.Should().Be(2m);
    }

    [Fact]
    public void Merge_FromEmpty_Noop()
    {
        var a = new WelfordMoments();
        a.Add(10m);
        a.Add(20m);

        var empty = new WelfordMoments();
        a.Merge(empty);

        a.Count.Should().Be(2);
        a.Mean.Should().Be(15m);
    }

    [Fact]
    public void Combine_ReturnsMergedResult_WithoutMutatingInputs()
    {
        var a = new WelfordMoments();
        a.Add(1m);
        a.Add(2m);
        var b = new WelfordMoments();
        b.Add(3m);
        b.Add(4m);

        var combined = WelfordMoments.Combine(a, b);

        a.Count.Should().Be(2);
        b.Count.Should().Be(2);
        combined.Count.Should().Be(4);
        combined.Mean.Should().Be(2.5m);
    }

    [Fact]
    public void Merge_RejectsNull()
    {
        FluentActions.Invoking(() => new WelfordMoments().Merge(null!))
            .Should().Throw<ArgumentNullException>();
    }
}
