"""Acquire and verify an immutable local held-out Harbor bundle."""

from __future__ import annotations

import tarfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.tooling.command_runner import operator_environment, run_operator_command
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import task_digest
from benchmarks.tooling.heldout_manifest import (
    _bundle_path,
    _digest,
    _read_json,
    _tree_digest,
    validate_manifest,
)

_AWS_ENVIRONMENT_VARS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    }
)


def _safe_extract(archive: Path, output: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            try:
                target.relative_to(output.resolve())
            except ValueError as exc:
                raise HarborSuiteError(
                    f"held-out archive path escapes output: {member.name}"
                ) from exc
            if member.issym() or member.islnk() or member.isdev():
                raise HarborSuiteError(
                    f"held-out archive contains a forbidden entry: {member.name}"
                )
        tar.extractall(output, members=members, filter="data")


def _verify_dataset_manifest(manifest: dict[str, Any], dataset_manifest: Path) -> None:
    if _digest(dataset_manifest) != manifest["dataset"]["manifest_digest"]:
        raise HarborSuiteError("held-out dataset manifest digest mismatch")
    if dataset_manifest.is_symlink():
        raise HarborSuiteError("held-out dataset manifest is a forbidden symlink")
    try:
        dataset_value = tomllib.loads(dataset_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarborSuiteError(f"held-out dataset manifest is invalid: {exc}") from exc
    entries = dataset_value.get("tasks") if isinstance(dataset_value, dict) else None
    if not isinstance(entries, list):
        raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")
    declared: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")
        name = entry.get("name")
        digest = entry.get("digest")
        if not isinstance(name, str) or not isinstance(digest, str) or name in declared:
            raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")
        declared[name] = digest
    expected = {
        f"jacobian/{task['id']}": str(task["digest"]) for task in manifest["tasks"]
    }
    if declared != expected:
        raise HarborSuiteError("held-out dataset manifest task set/digest mismatch")


def _verify_snapshot_lock(manifest: dict[str, Any], bundle_root: Path) -> None:
    """Verify the archive's snapshot lock agrees with the manifest.

    The snapshot lock is canonical: its task IDs/digests must match the
    manifest's tasks, and its lock_digest/snapshot_id must match the
    manifest's snapshot_lock reference.
    """

    lock_ref = manifest["snapshot_lock"]
    lock_path = bundle_root / "snapshot-lock.json"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise HarborSuiteError("held-out bundle is missing snapshot-lock.json")
    actual_digest = _digest(lock_path)
    if actual_digest != lock_ref["lock_digest"]:
        raise HarborSuiteError("held-out snapshot lock digest mismatch")
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        raise HarborSuiteError("held-out snapshot lock must be a JSON object")
    lock_snapshot_id = lock.get("snapshot_id")
    if not isinstance(lock_snapshot_id, str) or lock_snapshot_id != lock_ref["lock_id"]:
        raise HarborSuiteError("held-out snapshot lock_id mismatch")
    lock_tasks = lock.get("tasks")
    if not isinstance(lock_tasks, list) or not lock_tasks:
        raise HarborSuiteError("held-out snapshot lock has no tasks")
    lock_task_map: dict[str, str] = {}
    for entry in lock_tasks:
        if not isinstance(entry, dict):
            raise HarborSuiteError(
                "held-out snapshot lock task entries must be objects"
            )
        entry_id = entry.get("id")
        entry_digest = entry.get("digest")
        if (
            not isinstance(entry_id, str)
            or not isinstance(entry_digest, str)
            or entry_id in lock_task_map
        ):
            raise HarborSuiteError("held-out snapshot lock task set/digest mismatch")
        lock_task_map[entry_id] = entry_digest
    manifest_task_map = {
        str(task["id"]): str(task["digest"]) for task in manifest["tasks"]
    }
    if lock_task_map != manifest_task_map:
        raise HarborSuiteError("held-out archive tasks do not agree with snapshot lock")


def verify_bundle(manifest: dict[str, Any], root: Path) -> None:
    _verify_snapshot_lock(manifest, root)
    dataset_root = _bundle_path(root, manifest["dataset"]["path"])
    dataset_manifest = dataset_root / "dataset.toml"
    _verify_dataset_manifest(manifest, dataset_manifest)
    prompt = _bundle_path(root, manifest["experiment"]["prompt_path"])
    if _digest(prompt) != manifest["experiment"]["prompt_digest"]:
        raise HarborSuiteError("held-out prompt digest mismatch")
    for task in manifest["tasks"]:
        task_root = dataset_root / task["id"]
        actual = "sha256:" + task_digest(task_root).removeprefix("sha256:")
        if actual != task["digest"]:
            raise HarborSuiteError(f"held-out task digest mismatch: {task['id']}")
        for root_key, digest_key in (
            ("verifier_root", "verifier_tree_digest"),
            ("oracle_root", "oracle_tree_digest"),
        ):
            declared = _bundle_path(root, task[root_key])
            if _tree_digest(declared) != task[digest_key]:
                raise HarborSuiteError(
                    f"held-out {root_key} tree digest mismatch: {task['id']}"
                )


def _run_command(
    command: str,
    arguments: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    result = run_operator_command(
        command,
        arguments,
        cwd=cwd,
        timeout_seconds=600.0,
        environment=environment,
    )
    if result.exit_code is None or result.exit_code != 0:
        diagnostic = result.diagnostic or result.stderr.decode("utf-8", "replace")
        raise HarborSuiteError(f"held-out command {command} failed: {diagnostic}")


def fetch_bundle(manifest_uri: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    aws_env = operator_environment(include=_AWS_ENVIRONMENT_VARS)
    manifest_path = output / "manifest.json"
    _run_command(
        "aws",
        ["s3", "cp", manifest_uri, str(manifest_path)],
        cwd=output,
        environment=aws_env,
    )
    manifest = validate_manifest(manifest_path)
    lock_uri = manifest["snapshot_lock"]["lock_uri"]
    lock_path = output / "snapshot-lock.json"
    _run_command(
        "aws",
        ["s3", "cp", lock_uri, str(lock_path)],
        cwd=output,
        environment=aws_env,
    )
    if _digest(lock_path) != manifest["snapshot_lock"]["lock_digest"]:
        raise HarborSuiteError("held-out snapshot lock digest mismatch")
    archive = output / "bundle.tar.gz"
    _run_command(
        "aws",
        ["s3", "cp", manifest["archive"]["uri"], str(archive)],
        cwd=output,
        environment=aws_env,
    )
    if _digest(archive) != manifest["archive"]["sha256"]:
        raise HarborSuiteError("held-out archive digest mismatch")
    extracted = output / "bundle"
    extracted.mkdir()
    _safe_extract(archive, extracted)
    shutil_lock = extracted / "snapshot-lock.json"
    shutil_lock.write_bytes(lock_path.read_bytes())
    verify_bundle(manifest, extracted)
    return extracted
