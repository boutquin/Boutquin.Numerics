"""Executable failure-boundary regressions for shared verification infrastructure."""
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest

import conftest
import generate_qp_solver_vectors as qp


@pytest.mark.parametrize("kind", ["minimum", "mean"])
def test_iteration_limited_slsqp_is_not_a_reference_vector(monkeypatch, kind):
    original = qp.minimize
    statuses = []

    def limited(*args, **kwargs):
        kwargs["options"] = dict(kwargs["options"], maxiter=1)
        result = original(*args, **kwargs)
        statuses.append((result.success, result.status))
        return result

    monkeypatch.setattr(qp, "minimize", limited)
    cov = np.diag([1.0, 4.0, 9.0])
    with pytest.raises(RuntimeError, match="SLSQP.*9"):
        if kind == "minimum":
            qp.solve_min_variance_slsqp(cov, 0.0, 1.0)
        else:
            qp.solve_mean_variance_slsqp(cov, [0.1, 0.2, 0.3], 1.0, 0.0, 1.0)
    assert statuses == [(False, 9)]


def test_new_vector_directory_syncs_its_parent(monkeypatch, tmp_path):
    target = tmp_path / "vectors"
    monkeypatch.setattr(conftest, "VECTORS_DIR", target)
    original = os.fsync
    calls = []

    def synced(fd):
        stat = os.fstat(fd)
        calls.append((stat.st_dev, stat.st_ino))
        return original(fd)

    monkeypatch.setattr(os, "fsync", synced)
    conftest.save_vector("control", {"values": [1, 2, 3]})
    parent = tmp_path.stat()
    assert (parent.st_dev, parent.st_ino) in calls
    assert conftest.load_vector("control") == {"values": [1, 2, 3]}


@pytest.mark.parametrize("crash", [134, 137, 139])
def test_managed_anomaly_then_process_crash_retains_retry_metadata(tmp_path, crash):
    repo = Path(__file__).resolve().parents[2]
    workflow = (repo / ".github/workflows/pr-verify.yml").read_text()
    body = workflow.split("      - name: NIST/LM stress gate (reproduce-or-bound)\n")[1]
    body = body.split("        run: |\n")[1].split("\n      # always():")[0]
    body = "\n".join(line[10:] for line in body.splitlines())
    stub = r'''
    dotnet() {
      if [ "$1" = test ]; then return 0; fi
      count=0
      if [ -f "$RUNNER_TEMP/count" ]; then count=$(cat "$RUNNER_TEMP/count"); fi
      count=$((count + 1)); echo "$count" > "$RUNNER_TEMP/count"
      if [ "$count" -eq 1 ]; then return 4; fi
      return "$SECOND_EXIT"
    }
    sync() { return 0; }
    '''
    script = tmp_path / "gate.sh"
    script.write_text(stub + body)
    result = subprocess.run(["/bin/bash", "-e", str(script)], capture_output=True,
                            text=True, timeout=30,
                            env=dict(os.environ, RUNNER_TEMP=str(tmp_path), SECOND_EXIT=str(crash)))
    assert result.returncode == crash, result.stdout + result.stderr
    assert (tmp_path / "count").read_text().strip() == "2"
    metadata = tmp_path / "stress-artifacts/attempt-2/host-process-anomaly.txt"
    assert metadata.exists(), result.stdout + result.stderr
    assert f"exit-code: {crash}" in metadata.read_text()
    assert "attempt: 2" in metadata.read_text()


@pytest.mark.parametrize("kind", ["minimum", "mean"])
@pytest.mark.parametrize("weights", [[1.0, 0.0, 0.0], [0.2, 0.2, 0.2], [float("nan"), 0.5, 0.5]])
def test_success_flag_without_feasible_optimal_weights_is_refused(monkeypatch, kind, weights):
    from types import SimpleNamespace

    monkeypatch.setattr(qp, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, status=0, message="reported success", x=np.array(weights)))
    cov = np.diag([1.0, 4.0, 9.0])
    with pytest.raises(RuntimeError, match="certificate"):
        if kind == "minimum":
            qp.solve_min_variance_slsqp(cov, 0.0, 1.0)
        else:
            qp.solve_mean_variance_slsqp(cov, [0.0, 0.0, 0.0], 2.0, 0.0, 1.0)
