"""Regress the direct-consumer lock update path used by dependency updaters."""

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path("tools/Boutquin.Numerics.StressRunner/Boutquin.Numerics.StressRunner.csproj")
UNIT = Path("tests/Boutquin.Numerics.Tests.Unit/Boutquin.Numerics.Tests.Unit.csproj")
DOTNET = "/usr/local/share/dotnet/dotnet"


def test_stress_runner_exposes_every_test_dependency_to_the_updater():
    roots = {node.attrib["Include"] for node in ET.parse(ROOT / UNIT).iter("PackageReference")}
    runner = {node.attrib["Include"]: node for node in ET.parse(ROOT / RUNNER).iter("PackageReference")}
    assert set(runner) == roots
    lock = json.loads((ROOT / RUNNER.parent / "packages.lock.json").read_text())
    for name, reference in runner.items():
        assert "Version" not in reference.attrib
        assert set(reference.attrib["ExcludeAssets"].split(";")) >= {"build", "buildTransitive"}
        assert lock["dependencies"]["net10.0"][name]["type"] == "Direct"


def test_test_package_update_refreshes_the_whole_locked_solution(tmp_path):
    # Restore needs only manifests, not the production/test source or binaries.
    # An already cached adjacent version keeps this replay independent of feeds.
    package, before, after = "xunit.runner.visualstudio", "4.0.0", "4.0.1"
    cache = Path.home() / ".nuget/packages"
    assert (cache / package / after / f"{package}.{after}.nupkg").exists()
    solution = ET.parse(ROOT / "Boutquin.Numerics.slnx")
    projects = [Path(node.attrib["Path"]) for node in solution.iter("Project")]
    for path in [Path("Boutquin.Numerics.slnx"), Path("Directory.Build.props"),
                 Path("Directory.Packages.props"), Path("global.json"), Path("nuget.config")]:
        shutil.copy2(ROOT / path, tmp_path / path)
    for project in projects:
        (tmp_path / project.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / project, tmp_path / project)
        shutil.copy2(ROOT / project.parent / "packages.lock.json", tmp_path / project.parent / "packages.lock.json")
    central_path = tmp_path / "Directory.Packages.props"
    central = ET.parse(central_path)
    version = next(node for node in central.iter("PackageVersion") if node.attrib["Include"] == package)
    assert version.attrib["Version"] == before
    version.attrib["Version"] = after
    central.write(central_path)

    def restore(path, mode):
        return subprocess.run([DOTNET, "restore", str(tmp_path / path), mode, "--disable-parallel", "-m:1",
                               "--source", str(cache), "-p:NuGetAudit=false", "-v", "q"],
                              capture_output=True, text=True, timeout=120)

    # Reproduce the retained defect: only the three original direct test
    # consumers are updated. The unchanged executable lock must fail closed.
    for project in projects:
        if project.parts[0] == "tests":
            result = restore(project, "--force-evaluate")
            assert result.returncode == 0, result.stdout + result.stderr
    stale = restore(Path("Boutquin.Numerics.slnx"), "--locked-mode")
    assert stale.returncode != 0 and "NU1004" in stale.stdout + stale.stderr
    assert "Boutquin.Numerics.StressRunner" in stale.stdout + stale.stderr

    # The executable is now a direct consumer too; the same update operation
    # includes its lock without changing the package versions a second time.
    updated = restore(RUNNER, "--force-evaluate")
    assert updated.returncode == 0, updated.stdout + updated.stderr
    current = restore(Path("Boutquin.Numerics.slnx"), "--locked-mode")
    assert current.returncode == 0, current.stdout + current.stderr
    lock = json.loads((tmp_path / RUNNER.parent / "packages.lock.json").read_text())
    assert lock["dependencies"]["net10.0"][package]["resolved"] == after
