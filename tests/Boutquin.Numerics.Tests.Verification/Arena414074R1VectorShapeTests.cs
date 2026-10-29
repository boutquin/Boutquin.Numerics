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

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Boutquin.Numerics.Tests.Verification;

/// <summary>
/// Pins the shared vector loader's refusal of empty, missing, or misaligned
/// required families. Every verification consumer reads its reference rows
/// through <see cref="CrossLanguageVerificationBase.LoadVector"/> and bounds its
/// comparison loop by the loaded data, so a family that arrives empty (or
/// short of one of its required identities) would otherwise let the consumer
/// pass on zero compared rows.
/// </summary>
/// <remarks>
/// Each theory mutates an in-memory copy of a shipped fixture, writes it to a
/// temporary directory, and loads it through the production loader entry point
/// <c>LoadVectorFrom</c> (the same code path <c>LoadVector</c> takes). The loader
/// must refuse it with a message naming the vector file and the JSON path of the
/// offending family, so a rejection for an unrelated reason (a parse error, a
/// misspelt path) cannot satisfy the assertion.
/// </remarks>
public sealed class Arena414074R1VectorShapeTests : CrossLanguageVerificationBase
{
    /// <summary>
    /// Array families the verification consumers loop over: (vector file, dotted path
    /// with optional <c>[index]</c> steps). Read off the consumers' own reads and the
    /// Python generators' unconditional emissions.
    /// </summary>
    private static readonly (string Vector, string Path)[] s_consumedFamilies =
    [
        ("sample_moments", "cases"),
        ("qmc", "sobol"),
        ("qmc", "halton"),
        ("qmc_sobol_extended", "indices"),
        ("qmc_sobol_extended", "sobol"),
        ("haircut_fisher", "fisher_z.forward_cases"),
        ("haircut_fisher", "fisher_z.inverse_cases"),
        ("haircut_fisher", "fisher_z.ci_cases"),
        ("welford_pca", "pca.decompose_3x3.covariance_matrix"),
        ("welford_pca", "pca.decompose_3x3.eigenvalues"),
        ("welford_pca", "pca.decompose_3x3.eigenvectors"),
        ("welford_pca", "pca.decompose_3x3.explained_variance_ratio"),
        ("welford_pca", "pca.decompose_3x3.cumulative_explained_variance"),
        ("welford_pca", "pca.from_returns_50x4.returns"),
        ("welford_pca", "pca.from_returns_50x4.eigenvalues"),
        ("welford_pca", "pca.from_returns_50x4.explained_variance_ratio"),
        ("welford_pca", "pca.from_returns_50x4.mean"),
        ("welford_pca", "pca.from_returns_50x4.scores_top2"),
        ("graphical_lasso", "case_3x3_sparse.sample_covariance"),
        ("graphical_lasso", "case_3x3_sparse.sklearn_precision"),
        ("graphical_lasso", "case_3x3_sparse.bic_grid"),
        ("graphical_lasso", "case_5x5_block_diagonal.sample_covariance"),
        ("graphical_lasso", "case_5x5_block_diagonal.sklearn_precision"),
        ("graphical_lasso", "case_3x3_large_lambda.sample_covariance"),
        ("graphical_lasso", "case_3x3_large_lambda.sklearn_precision"),
        ("ppw2009", "rows"),
        ("ppw2009", "rows[0].checkpoints"),
        ("ppw2009", "rows[1].checkpoints"),
        ("ppw2009", "rows[2].checkpoints"),
        ("ppw2009", "rows[3].checkpoints"),
        ("ppw2009", "rows[4].checkpoints"),
        ("ppw2009", "rows[5].checkpoints"),
    ];

    /// <summary>Matrix families whose rows must keep their full width.</summary>
    private static readonly (string Vector, string Path)[] s_consumedMatrices =
    [
        ("qmc", "sobol"),
        ("qmc", "halton"),
        ("qmc_sobol_extended", "sobol"),
        ("welford_pca", "pca.decompose_3x3.covariance_matrix"),
        ("welford_pca", "pca.decompose_3x3.eigenvectors"),
        ("welford_pca", "pca.from_returns_50x4.returns"),
        ("welford_pca", "pca.from_returns_50x4.scores_top2"),
        ("graphical_lasso", "case_3x3_sparse.sample_covariance"),
        ("graphical_lasso", "case_3x3_sparse.sklearn_precision"),
        ("graphical_lasso", "case_5x5_block_diagonal.sample_covariance"),
        ("graphical_lasso", "case_5x5_block_diagonal.sklearn_precision"),
        ("graphical_lasso", "case_3x3_large_lambda.sample_covariance"),
        ("graphical_lasso", "case_3x3_large_lambda.sklearn_precision"),
    ];

    private static readonly string[] s_contractVectors =
    [
        "sample_moments",
        "qmc",
        "qmc_sobol_extended",
        "haircut_fisher",
        "welford_pca",
        "graphical_lasso",
        "ppw2009",
    ];

    public static TheoryData<string, string> Families()
    {
        var data = new TheoryData<string, string>();
        foreach (var (vector, path) in s_consumedFamilies)
        {
            data.Add(vector, path);
        }

        return data;
    }

    public static TheoryData<string, string> Matrices()
    {
        var data = new TheoryData<string, string>();
        foreach (var (vector, path) in s_consumedMatrices)
        {
            data.Add(vector, path);
        }

        return data;
    }

    public static TheoryData<string, bool> ShippedVectors()
    {
        var data = new TheoryData<string, bool>();
        foreach (var name in Directory.EnumerateFiles(GetVectorsDir(), "*.json")
                     .Select(Path.GetFileNameWithoutExtension)
                     .Order(StringComparer.Ordinal))
        {
            data.Add(name!, false);
            data.Add(name!, true);
        }

        return data;
    }

    public static TheoryData<string> Vectors()
    {
        var data = new TheoryData<string>();
        foreach (var vector in s_contractVectors)
        {
            data.Add(vector);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void EmptiedFamily_IsRejectedByLoader(string vector, string path)
    {
        var root = ParseShipped(vector);
        var (parent, key) = Locate(root, path);
        parent[key] = new JsonArray();

        AssertRejected(vector, root, path);
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void RemovedFamily_IsRejectedByLoader(string vector, string path)
    {
        var root = ParseShipped(vector);
        var (parent, key) = Locate(root, path);
        parent.AsObject().Remove(key);

        AssertRejected(vector, root, path);
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void FamilyMissingItsFirstIdentity_IsRejectedByLoader(string vector, string path)
    {
        var root = ParseShipped(vector);
        var (parent, key) = Locate(root, path);
        parent[key]!.AsArray().RemoveAt(0);

        AssertRejected(vector, root, path);
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void FamilyMissingItsLastIdentity_IsRejectedByLoader(string vector, string path)
    {
        var root = ParseShipped(vector);
        var (parent, key) = Locate(root, path);
        var family = parent[key]!.AsArray();
        family.RemoveAt(family.Count - 1);

        AssertRejected(vector, root, path);
    }

    [Theory]
    [MemberData(nameof(Matrices))]
    public void MatrixRowMissingAnEntry_IsRejectedByLoader(string vector, string path)
    {
        var root = ParseShipped(vector);
        var (parent, key) = Locate(root, path);
        parent[key]!.AsArray()[0]!.AsArray().RemoveAt(0);

        AssertRejected(vector, root, path);
    }

    [Fact]
    public void QmcPointCountZeroed_IsRejectedByLoader()
    {
        var root = ParseShipped("qmc");
        root["num_points"] = 0;

        AssertRejected("qmc", root, "num_points");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    [InlineData(12)]
    public void SobolIndexAndRowDroppedTogether_IsRejectedByLoader(int index)
    {
        // The two families stay the same length as each other, so only the required
        // identity set (thirteen indices ending at 1023) can refuse this.
        var root = ParseShipped("qmc_sobol_extended");
        root["indices"]!.AsArray().RemoveAt(index);
        root["sobol"]!.AsArray().RemoveAt(index);

        AssertRejected("qmc_sobol_extended", root, "indices");
    }

    [Fact]
    public void SobolIndexRenamed_IsRejectedByLoader()
    {
        // Same length, same alignment, wrong identity: 1023 re-labelled 1022.
        var root = ParseShipped("qmc_sobol_extended");
        root["indices"]![12] = 1022;

        AssertRejected("qmc_sobol_extended", root, "indices[12]");
    }

    [Fact]
    public void FisherForwardCorrelationRenamed_IsRejectedByLoader()
    {
        var root = ParseShipped("haircut_fisher");
        root["fisher_z"]!["forward_cases"]![0]!["r"] = -0.8;

        AssertRejected("haircut_fisher", root, "fisher_z.forward_cases[0].r");
    }

    [Fact]
    public void SampleMomentsCaseRenamed_IsRejectedByLoader()
    {
        var root = ParseShipped("sample_moments");
        root["cases"]![5]!["name"] = "small_n13";

        AssertRejected("sample_moments", root, "cases[5].name");
    }

    [Fact]
    public void PolitisWhiteCheckpointTrialRenamed_IsRejectedByLoader()
    {
        // Same length, wrong identity: trial 999 re-labelled so the lookup
        // would silently skip it.
        var root = ParseShipped("ppw2009");
        var checkpoints = root["rows"]![0]!["checkpoints"]!.AsArray();
        checkpoints[3]!["trial"] = 998;

        AssertRejected("ppw2009", root, "rows[0].checkpoints");
    }

    [Theory]
    [MemberData(nameof(Vectors))]
    public void ContractVectors_RoundTripCompactAndWithExtraProperty_AreAccepted(string vector)
    {
        var compact = ParseShipped(vector);
        using (var accepted = LoadMutated(vector, compact))
        {
            Assert.Equal(JsonValueKind.Object, accepted.RootElement.ValueKind);
        }

        var extended = ParseShipped(vector);
        extended["arena_unconsumed_note"] = "benign unknown property";
        using var withExtra = LoadMutated(vector, extended);
        Assert.True(withExtra.RootElement.TryGetProperty("arena_unconsumed_note", out _));
    }

    [Theory]
    [MemberData(nameof(ShippedVectors))]
    public void EveryShippedVector_RejectsAnEmptiedArray(string vector, bool lastArray)
    {
        // The empty-family rule is not limited to the vectors with a named contract:
        // every vector's consumers loop over its arrays, so emptying the first (or the
        // last) array in document order must be refused for all of them.
        var root = ParseShipped(vector);
        var arrays = new List<(string Path, JsonNode Parent, string? Key, int Index)>();
        CollectArrays(root, "$", arrays);
        Assert.NotEmpty(arrays);

        var (path, parent, key, index) = lastArray ? arrays[^1] : arrays[0];
        if (key is not null)
        {
            parent.AsObject()[key] = new JsonArray();
        }
        else
        {
            parent.AsArray()[index] = new JsonArray();
        }

        AssertRejected(vector, root, path["$.".Length..]);
    }

    [Fact]
    public void EveryShippedVector_LoadsThroughTheLoader()
    {
        var names = Directory.EnumerateFiles(GetVectorsDir(), "*.json")
            .Select(Path.GetFileNameWithoutExtension)
            .Order()
            .ToArray();

        Assert.NotEmpty(names);
        foreach (var name in names)
        {
            using var document = LoadVector(name!);
            Assert.NotEqual(JsonValueKind.Undefined, document.RootElement.ValueKind);
        }
    }

    private static JsonNode ParseShipped(string vector)
    {
        var text = File.ReadAllText(Path.Combine(GetVectorsDir(), $"{vector}.json"));
        return JsonNode.Parse(text)!;
    }

    private static void CollectArrays(JsonNode node, string path, List<(string Path, JsonNode Parent, string? Key, int Index)> arrays)
    {
        if (node is JsonObject obj)
        {
            foreach (var (name, child) in obj)
            {
                if (child is JsonArray)
                {
                    arrays.Add(($"{path}.{name}", obj, name, -1));
                }

                if (child is not null)
                {
                    CollectArrays(child, $"{path}.{name}", arrays);
                }
            }
        }
        else if (node is JsonArray array)
        {
            for (var i = 0; i < array.Count; i++)
            {
                var child = array[i];
                if (child is JsonArray)
                {
                    arrays.Add(($"{path}[{i}]", array, null, i));
                }

                if (child is not null)
                {
                    CollectArrays(child, $"{path}[{i}]", arrays);
                }
            }
        }
    }

    private static (JsonNode Parent, string Key) Locate(JsonNode root, string path)
    {
        var tokens = path.Split('.');
        var node = root;
        for (var i = 0; i < tokens.Length - 1; i++)
        {
            node = Step(node, tokens[i]);
        }

        return (node, tokens[^1]);
    }

    private static JsonNode Step(JsonNode node, string token)
    {
        var bracket = token.IndexOf('[');
        if (bracket < 0)
        {
            return node[token]!;
        }

        var index = int.Parse(token[(bracket + 1)..^1], CultureInfo.InvariantCulture);
        return node[token[..bracket]]![index]!;
    }

    private static JsonDocument LoadMutated(string vector, JsonNode root)
    {
        var directory = Directory.CreateTempSubdirectory("vector-shape-");
        try
        {
            File.WriteAllText(Path.Combine(directory.FullName, $"{vector}.json"), root.ToJsonString());
            return LoadVectorFrom(directory.FullName, vector);
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }

    private static void AssertRejected(string vector, JsonNode mutated, string path)
    {
        var error = Assert.Throws<InvalidDataException>(() => LoadMutated(vector, mutated).Dispose());

        Assert.Contains($"'{vector}'", error.Message);
        Assert.Contains($"$.{path}", error.Message);
    }
}
