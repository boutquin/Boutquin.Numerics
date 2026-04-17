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

using Boutquin.Numerics.Collections;
using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Collections;

public sealed class RollingWindowTests
{
    [Fact]
    public void Capacity_ReturnsConstructorValue()
    {
        var sut = new RollingWindow<int>(5);
        sut.Capacity.Should().Be(5);
    }

    [Fact]
    public void Count_StartsAtZero()
    {
        var sut = new RollingWindow<int>(3);
        sut.Count.Should().Be(0);
    }

    [Fact]
    public void IsFull_FalseWhenNotFull()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.IsFull.Should().BeFalse();
    }

    [Fact]
    public void IsFull_TrueWhenFull()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.Add(2);
        sut.Add(3);
        sut.IsFull.Should().BeTrue();
    }

    [Fact]
    public void Add_BelowCapacity_CountIncreases()
    {
        var sut = new RollingWindow<int>(5);
        sut.Add(1);
        sut.Add(2);
        sut.Count.Should().Be(2);
    }

    [Fact]
    public void Add_AtCapacity_CountStaysAtCapacity()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.Add(2);
        sut.Add(3);
        sut.Add(4);
        sut.Count.Should().Be(3);
    }

    [Fact]
    public void Add_BeyondCapacity_DropsOldest()
    {
        var sut = new RollingWindow<int>(2);
        sut.Add(1);
        sut.Add(2);
        sut.Add(3);
        sut.ToArray().Should().Equal(2, 3);
    }

    [Fact]
    public void Indexer_ReturnsChronologicalOrder()
    {
        var sut = new RollingWindow<int>(5);
        sut.Add(10);
        sut.Add(20);
        sut.Add(30);
        sut[0].Should().Be(10);
        sut[1].Should().Be(20);
        sut[2].Should().Be(30);
    }

    [Fact]
    public void Indexer_AfterWrap_ChronologicalOrder()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.Add(2);
        sut.Add(3);
        sut.Add(4);
        sut[0].Should().Be(2);
        sut[1].Should().Be(3);
        sut[2].Should().Be(4);
    }

    [Fact]
    public void Indexer_OutOfRange_Throws()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        var act = () => { var _ = sut[5]; };
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ToArray_MatchesIndexerOrder()
    {
        var sut = new RollingWindow<int>(4);
        sut.Add(10);
        sut.Add(20);
        sut.Add(30);
        var arr = sut.ToArray();
        arr.Should().Equal(sut[0], sut[1], sut[2]);
    }

    [Fact]
    public void Clear_ResetsCountToZero()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.Add(2);
        sut.Clear();
        sut.Count.Should().Be(0);
    }

    [Fact]
    public void Clear_AfterClear_CanAddAgain()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.Add(2);
        sut.Clear();
        sut.Add(99);
        sut.Count.Should().Be(1);
        sut[0].Should().Be(99);
    }

    [Fact]
    public void Enumeration_ChronologicalOrder()
    {
        var sut = new RollingWindow<int>(4);
        sut.Add(5);
        sut.Add(6);
        sut.Add(7);
        sut.Add(8);
        sut.Should().Equal(5, 6, 7, 8);
    }

    [Fact]
    public void Enumeration_AfterWrap_ChronologicalOrder()
    {
        var sut = new RollingWindow<int>(3);
        sut.Add(1);
        sut.Add(2);
        sut.Add(3);
        sut.Add(4);
        sut.Add(5);
        sut.Should().Equal(3, 4, 5);
    }

    [Fact]
    public void Constructor_ZeroCapacity_Throws()
    {
        var act = () => new RollingWindow<int>(0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Constructor_NegativeCapacity_Throws()
    {
        var act = () => new RollingWindow<int>(-1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void WorksWithStringType()
    {
        var sut = new RollingWindow<string>(2);
        sut.Add("a");
        sut.Add("b");
        sut.Add("c");
        sut.ToArray().Should().Equal("b", "c");
    }
}
