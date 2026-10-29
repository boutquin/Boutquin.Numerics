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

using System.Xml.Linq;

using Boutquin.Numerics.Solvers;

using FluentAssertions;

namespace Boutquin.Numerics.Tests.Unit.Solvers;

/// <summary>
/// Pins the accuracy claims the shipped documentation of <see cref="OrdinaryLeastSquares{T}"/>
/// and <see cref="OlsResult{T}"/> makes against what the NIST StRD suites in this project
/// measure: no uniform digit count that one of the eleven problems misses, and no
/// "28-digit" promise for <c>decimal</c> that describes the arithmetic width as if it
/// were the delivered accuracy.
/// </summary>
public sealed class OrdinaryLeastSquaresDocumentationClaimTests
{
    private static readonly XDocument s_docs = XDocument.Load(
        Path.Combine(AppContext.BaseDirectory, "Boutquin.Numerics.xml"));

    [Fact]
    public void OlsResultRemarks_DoNotPromiseUniform28DigitAccuracy()
    {
        var text = MemberText("T:Boutquin.Numerics.Solvers.OlsResult`1");

        text.Should().NotContain("uniform 28-digit precision",
            because: "decimal's 28-digit width is not the accuracy delivered on ill-conditioned problems");
        text.Should().Contain("Filip",
            because: "the remark must name the NIST problem where decimal delivers fewer digits");
        text.Should().Contain("1e-13",
            because: "the remark states the bar NistLinearRegressionDecimalTests asserts");
    }

    [Fact]
    public void OrdinaryLeastSquaresSummary_DoesNotClaimUniformBarOnEveryNistProblem()
    {
        var summary = Collapse(MemberElement("T:Boutquin.Numerics.Solvers.OrdinaryLeastSquares`1").Element("summary")!.Value);

        summary.Should().NotContain("10 significant digits on coefficients, 8 digits on standard errors) on every problem",
            because: "Filip misses the 10-digit coefficient and 8-digit standard-error bar");
        summary.Should().Contain("1e-12").And.Contain("1e-11",
            because: "the summary states the bars NistLinearRegressionTests asserts");
        summary.Should().Contain("Filip");
    }

    private static XElement MemberElement(string name) =>
        s_docs.Descendants("member").Single(e => (string?)e.Attribute("name") == name);

    private static string MemberText(string name) => Collapse(MemberElement(name).Value);

    private static string Collapse(string text) =>
        string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
}
