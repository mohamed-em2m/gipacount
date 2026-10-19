"""Small IO helpers (json, timestamps, git provenance)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["git_hash", "load_json", "save_json", "utc_timestamp"]


def utc_timestamp() -> str:
    """Compact UTC timestamp suitable for run names."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def save_json(payload: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=str)
    return path


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def git_hash(cwd: str | Path | None = None) -> str | None:
    """Return the current commit hash, or ``None`` when not in a repository.

    Every run stores this next to its config (prompt requirement: "config and git
    hash saved with every run").
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return out.stdout.strip() or None
