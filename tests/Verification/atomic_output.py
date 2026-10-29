"""Durable, atomic publication for generated verification artifacts (stdlib only)."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


def sync_directory(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def atomic_text(path: Path):
    """Sync a complete sibling file before replacing the destination, then sync its directory."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    sync_directory(path.parent.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
