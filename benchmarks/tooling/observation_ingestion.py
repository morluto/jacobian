"""Read Harbor result files into the inputs owned by evidence normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.tooling.errors import HarborSuiteError


@dataclass(frozen=True)
class HarborResult:
    """One selected Harbor result and its per-trial payloads."""

    path: Path
    payload: dict[str, Any]
    trials: tuple[tuple[Path | None, dict[str, Any]], ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read JSON {path}: {exc}") from exc


def _find_result(jobs_dir: Path) -> Path:
    candidates = sorted(
        (path for path in jobs_dir.glob("*/result.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise HarborSuiteError(f"no Harbor result.json found below {jobs_dir}")
    return candidates[0]


def load_harbor_result(jobs_dir: Path, result_path: Path | None = None) -> HarborResult:
    """Select one Harbor result and parse its concrete trial records."""

    path = (result_path or _find_result(jobs_dir)).resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise HarborSuiteError("Harbor result must be an object")
    paths = sorted(
        child for child in path.parent.glob("*/result.json") if child.is_file()
    )
    if paths:
        trials: list[tuple[Path | None, dict[str, Any]]] = []
        for child in paths:
            raw = _read_json(child)
            if not isinstance(raw, dict):
                raise HarborSuiteError(f"trial result must be an object: {child}")
            trials.append((child, raw))
    else:
        inline = payload.get("trial_results", [])
        if not isinstance(inline, list) or not all(
            isinstance(item, dict) for item in inline
        ):
            raise HarborSuiteError("Harbor result has no valid per-trial results")
        trials = [(None, item) for item in inline]
    return HarborResult(path=path, payload=payload, trials=tuple(trials))


__all__ = ["HarborResult", "load_harbor_result"]
