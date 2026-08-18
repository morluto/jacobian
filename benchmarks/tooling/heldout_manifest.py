"""Validate the immutable manifest for a held-out Harbor bundle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import BENCHMARKS
from benchmarks.tooling.heldout_manifest_models import HeldoutManifest
from benchmarks.tooling.strict_boundaries import raise_strict_model

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"invalid JSON {path}: {exc}") from exc


def _digest(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HarborSuiteError(f"unable to digest held-out file {path}: {exc}") from exc


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _tree_digest(root: Path) -> str:
    """Bind a complete regular-file tree, including names and empty-tree state."""
    if not root.is_dir():
        raise HarborSuiteError(f"held-out tree is not a directory: {root}")
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HarborSuiteError(f"held-out tree contains a symlink: {path}")
        if path.is_file():
            entries.append(
                {"path": path.relative_to(root).as_posix(), "digest": _digest(path)}
            )
    return _json_digest(entries)


def _bundle_path(root: Path, declared: str) -> Path:
    path = root / declared
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HarborSuiteError(f"held-out path escapes bundle: {declared}") from exc
    return path


def _validate_task_contracts(manifest: dict[str, Any]) -> set[str]:
    task_ids = [item["id"] for item in manifest["tasks"]]
    if len(set(task_ids)) != len(task_ids):
        raise HarborSuiteError("held-out task ids must be unique")
    families = {item["family"] for item in manifest["tasks"]}
    if len(families) < manifest["dataset"]["minimum_independent_families"]:
        raise HarborSuiteError("held-out bundle has too few independent families")
    for task in manifest["tasks"]:
        expected_prefix = f"dataset/{task['id']}/"
        roots = (task["verifier_root"], task["oracle_root"])
        if any(not root.startswith(expected_prefix) for root in roots):
            raise HarborSuiteError(
                f"held-out verifier/oracle roots must belong to task {task['id']}"
            )
    return set(task_ids)


def _validate_experiment(manifest: dict[str, Any], task_ids: set[str]) -> None:
    stages = manifest["experiment"]["stages"]
    for stage, config in stages.items():
        unknown = sorted(set(config["task_ids"]) - task_ids)
        if unknown:
            raise HarborSuiteError(f"{stage} references unknown task ids: {unknown}")
    if len(stages["pilot"]["task_ids"]) != 3:
        raise HarborSuiteError("pilot must freeze exactly three tasks")
    decision = stages["decision"]
    if len(decision["task_ids"]) < 5 or decision["repetitions"] < 5:
        raise HarborSuiteError(
            "decision stage requires at least five tasks and repetitions"
        )


def _validate_snapshot_lock(manifest: dict[str, Any]) -> None:
    lock = manifest["snapshot_lock"]
    if not _DIGEST_RE.match(lock["lock_id"]):
        raise HarborSuiteError("held-out snapshot_lock.lock_id must be a sha256 digest")
    if not _DIGEST_RE.match(lock["lock_digest"]):
        raise HarborSuiteError(
            "held-out snapshot_lock.lock_digest must be a sha256 digest"
        )
    if not isinstance(lock["lock_uri"], str) or not lock["lock_uri"]:
        raise HarborSuiteError("held-out snapshot_lock.lock_uri must be non-empty")


def validate_manifest(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    # Typed Pydantic boundary first: extra="forbid", strict scalars, no
    # semantic .get/indexing before typed parse.
    model = raise_strict_model(HeldoutManifest, raw, label=str(path))
    manifest = model.model_dump(mode="json", exclude_none=True)
    # JSON Schema as a secondary contract check for pattern constraints
    # (digest format, S3 URI shape, path patterns) not expressible in the
    # strict Pydantic model.
    schema = _read_json(BENCHMARKS / "schemas" / "held-out-manifest.schema.json")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        messages = [
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        ]
        raise HarborSuiteError("held-out manifest is invalid:\n" + "\n".join(messages))
    _validate_snapshot_lock(manifest)
    task_ids = _validate_task_contracts(manifest)
    _validate_experiment(manifest, task_ids)
    conditions = {
        item["id"]: (item["role"], item["jacobian_enabled"])
        for item in manifest["conditions"]
    }
    if conditions != {
        "C1": ("PRIMARY_CONTROL", False),
        "C2": ("PRIMARY_TREATMENT", True),
    }:
        raise HarborSuiteError("held-out conditions must be the frozen C1/C2 pair")
    return manifest
