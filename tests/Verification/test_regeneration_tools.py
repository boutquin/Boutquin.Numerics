"""Failure-path checks for regeneration and reused census build inputs."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

import atomic_output
import conftest
import generate_vectors


def load_script(name, relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["serialize", "file_sync", "replace"])
def test_failed_publication_retains_complete_previous_vector(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(conftest, "VECTORS_DIR", tmp_path)
    path = tmp_path / "retained.json"
    original = b'{"complete": "previous"}\n'
    path.write_bytes(original)

    def fail(*args):
        raise OSError("injected publication failure")

    data = {"value": object()} if failure == "serialize" else {"value": [1, 2, 3]}
    if failure == "replace":
        monkeypatch.setattr(os, "replace", fail)
    if failure == "file_sync":
        sync = os.fsync

        def fail_file(fd):
            import stat
            if stat.S_ISREG(os.fstat(fd).st_mode):
                fail()
            sync(fd)

        monkeypatch.setattr(os, "fsync", fail_file)
    with pytest.raises((TypeError, OSError)):
        conftest.save_vector("retained", data)
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corpus_main_preserves_previous_file_when_serialization_fails(monkeypatch, tmp_path):
    corpus = load_script("corpus_regeneration_control", "tests/Verification/qp-corpus/generate_corpus.py")
    cases = [{"T": kind, "family": "control"} for kind in ["double", "decimal"]]
    # Run the real publication path with a bounded oracle panel. The second
    # record fails after the stream has already received its JSON header.
    rich = [{"id": "control", "qualified": True, "unsupported": object()}] * 2

    class Pool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, *args, **kwargs):
            return rich

    monkeypatch.setattr(corpus.multiprocessing, "Pool", Pool)
    monkeypatch.setattr(corpus, "build_cases", lambda: (cases, [{"family": "control"}]))
    monkeypatch.setattr(corpus.sys, "argv", ["generate_corpus.py", str(tmp_path)])
    previous = b'{"complete":true}\n'
    for kind in ["double", "decimal"]:
        (tmp_path / f"corpus-{kind}.json").write_bytes(previous)
    with pytest.raises(TypeError):
        corpus.main()
    for kind in ["double", "decimal"]:
        assert (tmp_path / f"corpus-{kind}.json").read_bytes() == previous
    assert list(tmp_path.glob(".*.tmp")) == []
    for row in rich:
        row.pop("unsupported", None)
    corpus.main()
    for kind in ["double", "decimal"]:
        parsed = json.loads((tmp_path / f"corpus-{kind}.json").read_text())
        assert parsed["type"] == kind
        assert len(parsed["cases"]) == 1




def test_orchestrator_executes_every_consumed_computed_family(monkeypatch):
    called = []
    for name in vars(generate_vectors):
        if name.startswith("gen_") or name == "refresh_selected_contract_vectors":
            monkeypatch.setattr(generate_vectors, name, lambda name=name: called.append(name))
    generate_vectors.main()
    for required in ["gen_graphical_lasso", "refresh_selected_contract_vectors", "gen_jarre",
                     "gen_ppw2009", "gen_rmt", "gen_garch_independent", "gen_lwsf_spike"]:
        assert called.count(required) == 1
    assert called.index("gen_garch") < called.index("gen_garch_independent")
    assert called.index("gen_covariance") < called.index("refresh_selected_contract_vectors")


@pytest.mark.parametrize("module_name,vector", [
    ("generate_jarre_spline_vectors", "jarre_splines"),
    ("generate_ppw2009_vectors", "ppw2009"),
])
def test_standalone_generator_retains_previous_vector_on_replace_failure(monkeypatch, tmp_path, module_name, vector):
    import importlib
    module = importlib.import_module(module_name)
    monkeypatch.setattr(conftest, "VECTORS_DIR", tmp_path)
    path = tmp_path / f"{vector}.json"
    path.write_text('{"previous":true}')

    def refused(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(atomic_output.os, "replace", refused)
    with pytest.raises(OSError, match="injected replace failure"):
        module.generate()
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.glob(".*.tmp")) == []
