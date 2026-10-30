#!/usr/bin/env python3
"""Verify the immutable Numerics 2.0.0 archives before uploading them."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

SOURCE = "d3ee2a940fa2f2274db845b95468cfba8c8748e8"
ARCHIVES = {
    "Boutquin.Numerics.2.0.0.nupkg": "d63575e38ae185b7f4a9c965c9d6b385fe1c76645286b0de404c6bde565159ed",
    "Boutquin.Numerics.2.0.0.snupkg": "5049fd664680b3702c1432e14e0b4fe5fd46a51efe03a9c3ce79f1a61e8351f5",
}


def verify(directory, commit):
    if commit != SOURCE:
        raise ValueError("Release tag does not name the qualified source")
    if directory.is_symlink() or {p.name for p in directory.iterdir()} != set(ARCHIVES):
        raise ValueError("Release archive inventory differs")
    for name, expected in ARCHIVES.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release archive is not a regular file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Qualified archive bytes changed: " + name)
        with zipfile.ZipFile(path) as archive:
            metadata = ET.fromstring(archive.read("Boutquin.Numerics.nuspec"))
            values = {node.tag.rsplit("}", 1)[-1]: node for node in metadata.iter()}
            if values["id"].text != "Boutquin.Numerics" or values["version"].text != "2.0.0":
                raise ValueError("Package identity differs")
            if values["repository"].attrib.get("commit") != SOURCE:
                raise ValueError("Package source differs")
    return {"status": "PASS_EXACT_QUALIFIED_ARCHIVES", "source": SOURCE, "sha256": ARCHIVES}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]), sys.argv[2]), indent=2))
