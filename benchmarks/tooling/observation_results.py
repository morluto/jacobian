"""Normalize and compare Harbor model-in-the-loop observation results.

This is the strict normalized observation evidence *v2* implementation.  It
replaces v1 atomically and tightens three classes of contract that v1 left
implicit:

* **Dataset/task selection** is normalized to exactly one of the two Harbor job
  forms -- ``datasets[].path`` with optional ``task_names`` or explicit
  ``tasks[].path``.  Mixed selections, outside-dataset paths, unknown task
  names, empty selections, and the v1 implicit "fall back to all known tasks"
  behavior are rejected.
* **Artifact identity** binds ``job``/``trial``/``step``/canonical source
  path/artifact-relative path/digest for every observed trace artifact.
  Identical bytes at distinct canonical source paths remain distinct and
  allowed; reusing the same canonical source path more than once is rejected.
* **Path hygiene** rejects absolute, traversal, escaping-symlink, missing
  manifest, and malformed multistep paths.  The evidence fails closed:
  ``TIMEOUT``/``CANCELLED``/``ERROR`` and incomplete enumeration never produce
  ``VALID`` evidence and never authorize a causal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.harbor_suite import (
    ROOT,
    HarborSuiteError,
    get_suite,
    task_digest,
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_contract(value: dict[str, Any], schema_name: str) -> None:
    schema = _read_json(ROOT / "benchmarks" / "schemas" / schema_name)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise HarborSuiteError(
            f"{schema_name} contract is invalid: {errors[0].message}"
        )


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _find_result(jobs_dir: Path) -> Path:
    candidates = sorted(
        (path for path in jobs_dir.glob("*/result.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise HarborSuiteError(f"no Harbor result.json found below {jobs_dir}")
    return candidates[0]


def _trial_results(
    result_path: Path, payload: dict[str, Any]
) -> list[tuple[Path | None, dict[str, Any]]]:
    paths = sorted(
        path for path in result_path.parent.glob("*/result.json") if path.is_file()
    )
    if paths:
        values: list[tuple[Path | None, dict[str, Any]]] = []
        for path in paths:
            raw = _read_json(path)
            if not isinstance(raw, dict):
                raise HarborSuiteError(f"trial result must be an object: {path}")
            values.append((path, raw))
        return values
    inline = payload.get("trial_results", [])
    if not isinstance(inline, list) or not all(
        isinstance(item, dict) for item in inline
    ):
        raise HarborSuiteError("Harbor result has no valid per-trial results")
    return [(None, item) for item in inline]


def _task_id(name: Any) -> str:
    return name.rsplit("/", 1)[-1] if isinstance(name, str) else ""


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Strict dataset/task selection normalization (v2)
# ---------------------------------------------------------------------------


def _selection_known(
    dataset: str, heldout_manifest: dict[str, Any] | None
) -> tuple[dict[str, str], dict[str, Path], Path | None, str, str]:
    """Return (name->digest, name->task_dir, dataset_path, evidence_class, dataset_id)."""

    if heldout_manifest is not None:
        known = {
            str(item["id"]): str(item["digest"]) for item in heldout_manifest["tasks"]
        }
        dataset_path = None
        evidence_class = "held-out-comparative-evaluation"
        dataset_id = str(heldout_manifest.get("dataset", {}).get("id", dataset))
        return known, {}, dataset_path, evidence_class, dataset_id

    suite = get_suite(dataset)
    known = {
        ref.path.name: "sha256:" + task_digest(ref.path).removeprefix("sha256:")
        for ref in suite.tasks
    }
    task_dirs = {ref.path.name: ref.path for ref in suite.tasks}
    evidence_class = (
        "workflow-observation"
        if suite.claim_class == "workflow-observation"
        else suite.claim_class
    )
    return known, task_dirs, suite.path, evidence_class, suite.id


def _reject_path(value: str, *, label: str) -> list[str]:
    """Reject absolute, traversal, and escaping-symlink paths lexically."""

    failures: list[str] = []
    candidate = Path(value)
    if candidate.is_absolute():
        failures.append(f"{label} must be relative: {value!r}")
        return failures
    parts = candidate.parts
    if any(part == ".." for part in parts):
        failures.append(f"{label} must not traverse parent directories: {value!r}")
        return failures
    if any(part in {"", "."} for part in parts) and len(parts) > 1:
        failures.append(f"{label} is malformed: {value!r}")
    # Walk the lexical chain from ROOT to detect symlinks without resolving.
    current = ROOT
    for part in parts:
        current = current / part
        if current.is_symlink():
            failures.append(f"{label} crosses an escaping symlink: {value!r}")
            return failures
    return failures


def _validate_explicit_task_path(
    value: str,
    *,
    dataset_path: Path | None,
    task_dirs: dict[str, Path],
) -> tuple[str | None, list[str]]:
    failures = _reject_path(value, label="explicit task path")
    if failures:
        return None, failures
    resolved = (ROOT / value).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        failures.append(f"explicit task path escapes repository: {value!r}")
        return None, failures
    if dataset_path is not None:
        try:
            resolved.relative_to(dataset_path.resolve())
        except ValueError:
            failures.append(f"explicit task path is outside the dataset: {value!r}")
            return None, failures
    short = resolved.name
    expected = task_dirs.get(short)
    if expected is None or expected.resolve() != resolved:
        failures.append(f"explicit task path is not a known task: {value!r}")
        return None, failures
    manifest = resolved / "task.toml"
    if not manifest.is_file() or manifest.is_symlink():
        failures.append(f"explicit task is missing its manifest: {value!r}")
        return None, failures
    return short, failures


def _normalize_selection(
    job: dict[str, Any],
    *,
    known: dict[str, str],
    task_dirs: dict[str, Path],
    dataset_path: Path | None,
) -> tuple[list[str], str, dict[str, Any], list[str]]:
    """Normalize exactly one selection form; reject mixed/unknown/empty/fallback."""

    failures: list[str] = []
    has_datasets = job.get("datasets") is not None
    has_tasks = job.get("tasks") is not None
    if has_datasets and has_tasks:
        failures.append(
            "job must select tasks via datasets or explicit tasks, not both"
        )
        return [], "mixed", _eval_args("mixed", [], None, None, 0), failures
    if has_datasets:
        return _normalize_dataset_selection(job["datasets"], known=known)
    if has_tasks:
        return _normalize_explicit_selection(
            job["tasks"], task_dirs=task_dirs, dataset_path=dataset_path
        )
    failures.append(
        "job must select tasks via datasets or explicit tasks; "
        "implicit fallback to all known tasks is forbidden"
    )
    return (
        [],
        "implicit-fallback",
        _eval_args("implicit-fallback", [], None, None, 0),
        failures,
    )


def _normalize_dataset_selection(
    datasets: Any, *, known: dict[str, str]
) -> tuple[list[str], str, dict[str, Any], list[str]]:
    failures: list[str] = []
    if not isinstance(datasets, list) or not datasets:
        failures.append("job datasets must be a non-empty array")
        return (
            [],
            "dataset-task-names",
            _eval_args("dataset-task-names", [], None, None, 0),
            failures,
        )
    selected: set[str] = set()
    norm_datasets: list[dict[str, Any]] = []
    for entry in datasets:
        normalized, names, entry_failures = _normalize_dataset_entry(entry, known)
        failures.extend(entry_failures)
        if normalized is not None:
            norm_datasets.append(normalized)
            selected.update(names)
    selected_sorted = sorted(selected)
    if not selected_sorted and not failures:
        failures.append("dataset selection resolved to no tasks")
    eval_args = _eval_args(
        "dataset-task-names", selected_sorted, norm_datasets, None, 0
    )
    return selected_sorted, "dataset-task-names", eval_args, failures


def _normalize_dataset_entry(
    entry: Any, known: dict[str, str]
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if not isinstance(entry, dict):
        return None, [], ["job dataset entry must be an object"]
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        return None, [], ["job dataset entry must have a non-empty path"]
    task_names = entry.get("task_names")
    if task_names is None:
        return {"path": path, "task_names": None}, list(known), []
    if not isinstance(task_names, list) or not task_names:
        return None, [], ["task_names must be a non-empty array when present"]
    names: list[str] = []
    failures: list[str] = []
    for name in task_names:
        if not isinstance(name, str) or not name:
            failures.append("task_names must be non-empty strings")
        elif name not in known:
            failures.append(f"unknown task name in dataset selection: {name}")
        else:
            names.append(name)
    return {"path": path, "task_names": names}, names, failures


def _normalize_explicit_selection(
    tasks: Any, *, task_dirs: dict[str, Path], dataset_path: Path | None
) -> tuple[list[str], str, dict[str, Any], list[str]]:
    failures: list[str] = []
    if not isinstance(tasks, list) or not tasks:
        failures.append("job tasks must be a non-empty array")
        return (
            [],
            "explicit-tasks",
            _eval_args("explicit-tasks", [], None, None, 0),
            failures,
        )
    selected: set[str] = set()
    norm_tasks: list[dict[str, Any]] = []
    for entry in tasks:
        if not isinstance(entry, dict):
            failures.append("job task entry must be an object")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            failures.append("job task entry must have a non-empty path")
            continue
        short, path_failures = _validate_explicit_task_path(
            path, dataset_path=dataset_path, task_dirs=task_dirs
        )
        failures.extend(path_failures)
        if short is not None:
            if short in selected:
                failures.append(f"explicit task path reused: {path!r}")
            selected.add(short)
        norm_tasks.append({"path": path})
    selected_sorted = sorted(selected)
    if not selected_sorted and not failures:
        failures.append("explicit task selection resolved to no tasks")
    eval_args = _eval_args("explicit-tasks", selected_sorted, None, norm_tasks, 0)
    return selected_sorted, "explicit-tasks", eval_args, failures


def _eval_args(
    mode: str,
    selection: list[str],
    datasets: list[dict[str, Any]] | None,
    tasks: list[dict[str, Any]] | None,
    n_attempts: int,
) -> dict[str, Any]:
    record = {
        "selection_mode": mode,
        "datasets": datasets,
        "tasks": tasks,
        "selection": selection,
        "n_attempts": n_attempts,
    }
    record["selection_digest"] = _json_digest(
        {
            "selection_mode": mode,
            "datasets": datasets,
            "tasks": tasks,
            "selection": selection,
        }
    )
    return record


def _comparison_job(job: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the frozen Jacobian treatment additions.

    Any other condition-specific Compose or MCP change remains in the
    comparison signature and therefore invalidates the pair.
    """

    normalized: dict[str, Any] = json.loads(json.dumps(job))
    normalized.pop("jobs_dir", None)
    environment = normalized.get("environment")
    if isinstance(environment, dict):
        compose = environment.get("extra_docker_compose")
        if isinstance(compose, list):
            environment["extra_docker_compose"] = [
                value for value in compose if Path(str(value)).name != "c2.compose.json"
            ]
    for agent in normalized.get("agents", []):
        if isinstance(agent, dict):
            servers = agent.get("mcp_servers")
            if isinstance(servers, list):
                remaining = [
                    server
                    for server in servers
                    if server
                    != {
                        "name": "jacobian",
                        "transport": "streamable-http",
                        "url": "http://jacobian:8000/mcp",
                    }
                ]
                if remaining:
                    agent["mcp_servers"] = remaining
                else:
                    agent.pop("mcp_servers", None)
    return normalized


def _timing_seconds(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    started = value.get("started_at")
    finished = value.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        from datetime import datetime

        return (
            datetime.fromisoformat(finished.replace("Z", "+00:00"))
            - datetime.fromisoformat(started.replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        return None


def _walk_trace(value: Any, calls: Counter[str]) -> int:
    if isinstance(value, list):
        return sum(_walk_trace(child, calls) for child in value)
    if not isinstance(value, dict):
        return 0
    tool_name = value.get("tool_name")
    if isinstance(tool_name, str):
        calls[tool_name] += 1
    if value.get("type") in {"tool_call", "tool_use"} and isinstance(
        value.get("name"), str
    ):
        calls[str(value["name"])] += 1
    own_error = int(value.get("error") not in {None, False, ""})
    return own_error + sum(_walk_trace(child, calls) for child in value.values())


def _read_trace(path: Path, calls: Counter[str]) -> int:
    try:
        if path.suffix == ".jsonl":
            return sum(
                _walk_trace(json.loads(line), calls)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if path.suffix == ".json":
            return _walk_trace(_read_json(path), calls)
    except (OSError, json.JSONDecodeError, HarborSuiteError):
        return 1
    return 0


_MANIFEST_FILENAME = "manifest.json"
_ARTIFACTS_DIR = "artifacts"
_STEPS_DIR = "steps"
_MANIFEST_STATUSES_OK = {"ok"}
_MANIFEST_STATUSES_NON_CONCLUSION = {"failed", "empty", "skipped"}
_MANIFEST_STATUSES = _MANIFEST_STATUSES_OK | _MANIFEST_STATUSES_NON_CONCLUSION


def _artifact_path_failures(path: Path, trial_root: Path) -> list[str]:
    """Reject absolute, traversal, escaping-symlink, and malformed multistep paths."""

    failures: list[str] = []
    try:
        rel = path.relative_to(trial_root)
    except ValueError:
        return [f"artifact path escapes trial root: {path}"]
    parts = rel.parts
    if not parts:
        return []
    if any(part == ".." for part in parts):
        failures.append(f"artifact path traversal forbidden: {rel.as_posix()}")
        return failures
    if any(part == "" for part in parts):
        failures.append(f"artifact path is malformed: {rel.as_posix()}")
        return failures
    current = trial_root
    for part in parts:
        current = current / part
        if current.is_symlink():
            failures.append(
                f"artifact path crosses an escaping symlink: {rel.as_posix()}"
            )
            return failures
    try:
        path.resolve().relative_to(trial_root.resolve())
    except ValueError:
        failures.append(f"artifact path escapes trial root: {rel.as_posix()}")
    return failures


def _canonical_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _validate_manifest_entry(entry: Any, *, location: str) -> list[str]:
    """Validate one Harbor 0.20 ``ArtifactManifestEntry`` shape."""

    failures: list[str] = []
    if not isinstance(entry, dict):
        return [f"{location}: manifest entry must be an object"]
    for field in ("source", "destination", "type", "status"):
        if field not in entry:
            failures.append(f"{location}: manifest entry missing field {field!r}")
    if failures:
        return failures
    source = entry.get("source")
    destination = entry.get("destination")
    entry_type = entry.get("type")
    status = entry.get("status")
    service = entry.get("service")
    if not isinstance(source, str) or not source:
        failures.append(f"{location}: manifest entry source must be a non-empty string")
    if not isinstance(destination, str) or not destination:
        failures.append(
            f"{location}: manifest entry destination must be a non-empty string"
        )
    if entry_type not in {"file", "directory"}:
        failures.append(f"{location}: manifest entry type must be file or directory")
    if status not in _MANIFEST_STATUSES:
        failures.append(
            f"{location}: manifest entry status must be one of "
            f"{sorted(_MANIFEST_STATUSES)}"
        )
    if service is not None and (not isinstance(service, str) or not service):
        failures.append(
            f"{location}: manifest entry service must be null or a non-empty string"
        )
    return failures


def _manifest_destination(entry: dict[str, Any]) -> str | None:
    destination = entry.get("destination")
    if not isinstance(destination, str) or not destination:
        return None
    if destination.startswith("artifacts/"):
        return destination.removeprefix("artifacts/")
    if destination == "artifacts":
        return ""
    return destination


def _bind_manifest_file(
    host_path: Path,
    *,
    artifacts_dir: Path,
    trial_root: Path,
    rel: str,
    entry: dict[str, Any],
    job_label: str,
    trial_name: str,
    step_index: int,
    step_name: str | None,
    source_prefix: str | None,
) -> dict[str, Any]:
    source = entry.get("source")
    service = entry.get("service")
    return _bind_artifact_file(
        host_path=host_path,
        artifacts_dir=artifacts_dir,
        trial_root=trial_root,
        rel=rel,
        manifest_source=str(source) if isinstance(source, str) else "",
        service=service if isinstance(service, str) else None,
        job_label=job_label,
        trial_name=trial_name,
        step_index=step_index,
        step_name=step_name,
        source_prefix=source_prefix,
    )


def _manifest_file_artifacts(
    host_path: Path, **context: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    trial_name = str(context["trial_name"])
    index = int(context["index"])
    rel = str(context["rel"])
    if not host_path.is_file():
        return [], [
            f"trial {trial_name}: manifest entry {index} file is missing on disk: {rel}"
        ]
    if host_path.is_symlink():
        return [], [
            f"trial {trial_name}: manifest entry {index} file is a forbidden symlink: {rel}"
        ]
    bind_context = {key: value for key, value in context.items() if key != "index"}
    return [_bind_manifest_file(host_path, **bind_context)], []


def _manifest_directory_artifacts(
    host_path: Path, **context: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    trial_name = str(context["trial_name"])
    index = int(context["index"])
    rel = str(context["rel"])
    if not host_path.is_dir():
        return [], [
            f"trial {trial_name}: manifest entry {index} directory is missing on disk: {rel}"
        ]
    if host_path.is_symlink():
        return [], [
            f"trial {trial_name}: manifest entry {index} directory is a forbidden symlink: {rel}"
        ]
    regular_files = sorted(
        child
        for child in host_path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )
    if not regular_files:
        return [], [
            f"trial {trial_name}: manifest entry {index} ok directory is unexpectedly empty: {rel}"
        ]
    artifacts: list[dict[str, Any]] = []
    failures: list[str] = []
    artifacts_dir = context["artifacts_dir"]
    bind_context = {key: value for key, value in context.items() if key != "index"}
    for child in regular_files:
        child_failures = _artifact_path_failures(child, artifacts_dir)
        failures.extend(child_failures)
        if not child_failures:
            bind_context["rel"] = child.relative_to(artifacts_dir).as_posix()
            artifacts.append(_bind_manifest_file(child, **bind_context))
    return artifacts, failures


def _manifest_entry_artifacts(
    entry: Any,
    *,
    index: int,
    artifacts_dir: Path,
    trial_root: Path,
    trial_name: str,
    job_label: str,
    step_index: int,
    step_name: str | None,
    source_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    location = f"trial {trial_name} manifest[{index}]"
    failures = _validate_manifest_entry(entry, location=location)
    if not isinstance(entry, dict):
        return [], failures
    status = entry.get("status")
    if status in _MANIFEST_STATUSES_NON_CONCLUSION:
        failures.append(
            f"trial {trial_name}: artifact manifest entry {index} has "
            f"non-conclusion status {status!r} (source={entry.get('source')!r})"
        )
        return [], failures
    if status != "ok":
        return [], failures
    rel = _manifest_destination(entry)
    if rel is None:
        return [], failures
    if not rel:
        failures.append(
            f"trial {trial_name}: manifest entry {index} destination resolves "
            f"to the artifacts root: {entry.get('destination')!r}"
        )
        return [], failures
    host_path = artifacts_dir / rel
    path_failures = _artifact_path_failures(host_path, artifacts_dir)
    failures.extend(path_failures)
    if path_failures:
        return [], failures
    context = {
        "index": index,
        "artifacts_dir": artifacts_dir,
        "trial_root": trial_root,
        "rel": rel,
        "entry": entry,
        "job_label": job_label,
        "trial_name": trial_name,
        "step_index": step_index,
        "step_name": step_name,
        "source_prefix": source_prefix,
    }
    if entry.get("type") == "file":
        artifacts, entry_failures = _manifest_file_artifacts(host_path, **context)
    elif entry.get("type") == "directory":
        artifacts, entry_failures = _manifest_directory_artifacts(host_path, **context)
    else:
        artifacts, entry_failures = (
            [],
            [
                f"trial {trial_name}: manifest entry {index} has unknown type {entry.get('type')!r}"
            ],
        )
    failures.extend(entry_failures)
    return artifacts, failures


def _manifest_artifacts_for_dir(
    artifacts_dir: Path,
    *,
    trial_root: Path,
    trial_name: str,
    job_label: str,
    step_index: int,
    step_name: str | None,
    source_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read one ``artifacts/manifest.json`` and bind every ``ok`` entry.

    Harbor 0.20 writes ``manifest.json`` as a JSON array of entries
    ``{source, destination, type, status, service}``.  ``destination`` is the
    artifact-relative path (``artifacts/<relative>``); ``source`` is the
    canonical container source path.  Only ``ok`` entries have a collected
    file on disk; ``failed``/``empty``/``skipped`` are non-conclusion records
    that are surfaced as failures because evidence is incomplete.
    """

    failures: list[str] = []
    if not artifacts_dir.is_dir():
        failures.append(
            f"trial {trial_name}: artifacts directory is missing: "
            f"{artifacts_dir.relative_to(trial_root) if trial_root in artifacts_dir.parents or artifacts_dir == trial_root else artifacts_dir}"
        )
        return [], failures
    manifest_path = artifacts_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        failures.append(f"trial {trial_name}: artifact manifest is missing")
        return [], failures
    if manifest_path.is_symlink():
        failures.append(f"trial {trial_name}: artifact manifest is a forbidden symlink")
        return [], failures
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"trial {trial_name}: artifact manifest is unreadable: {exc}")
        return [], failures
    if not isinstance(raw, list):
        failures.append(f"trial {trial_name}: artifact manifest must be a JSON array")
        return [], failures
    if not raw:
        failures.append(f"trial {trial_name}: artifact manifest is empty")
        return [], failures
    artifacts: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        entry_artifacts, entry_failures = _manifest_entry_artifacts(
            entry,
            index=index,
            artifacts_dir=artifacts_dir,
            trial_root=trial_root,
            trial_name=trial_name,
            job_label=job_label,
            step_index=step_index,
            step_name=step_name,
            source_prefix=source_prefix,
        )
        artifacts.extend(entry_artifacts)
        failures.extend(entry_failures)
    return artifacts, failures


def _bind_artifact_file(
    *,
    host_path: Path,
    artifacts_dir: Path,
    trial_root: Path,
    rel: str,
    manifest_source: str,
    service: str | None,
    job_label: str,
    trial_name: str,
    step_index: int,
    step_name: str | None,
    source_prefix: str | None,
) -> dict[str, Any]:
    """Bind one regular on-disk file to a manifest-driven artifact identity."""

    try:
        source_path = host_path.relative_to(trial_root.parent).as_posix()
    except ValueError:
        source_path = rel
    if source_prefix:
        source_path = f"{source_prefix.rstrip('/')}/{source_path}"
    return {
        "job": job_label,
        "trial": trial_name,
        "step": step_index,
        "step_name": step_name,
        "source_path": source_path,
        "manifest_source": manifest_source,
        "service": service,
        "artifact_path": rel,
        "digest": _sha256(host_path),
    }


def _trial_artifacts(
    trial_path: Path | None,
    trial_name: str,
    job_label: str,
    *,
    source_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int, list[str]]:
    """Collect manifest-driven artifacts for one trial.

    Single-step trials expose ``<trial>/artifacts/manifest.json``; multi-step
    trials expose ``<trial>/steps/<step>/artifacts/manifest.json`` per step.
    Both are read; every ``ok`` entry binds ``job``/``trial``/``step``/
    ``step_name``/``source``/``service``/``artifact_path``/``digest``.
    Missing manifests, missing files, malformed entries, and non-conclusion
    statuses are failures (evidence fails closed).
    """

    if trial_path is None:
        return [], {}, 0, []
    root = trial_path.parent
    artifacts: list[dict[str, Any]] = []
    failures: list[str] = []
    steps_dir = root / _STEPS_DIR
    if steps_dir.is_dir():
        step_dirs = sorted(
            path
            for path in steps_dir.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
        if not step_dirs:
            failures.append(f"trial {trial_name}: steps directory is empty")
        for step_index, step_dir in enumerate(step_dirs):
            step_artifacts, step_failures = _manifest_artifacts_for_dir(
                step_dir / _ARTIFACTS_DIR,
                trial_root=root,
                trial_name=trial_name,
                job_label=job_label,
                step_index=step_index,
                step_name=step_dir.name,
                source_prefix=source_prefix,
            )
            artifacts.extend(step_artifacts)
            failures.extend(step_failures)
    else:
        step_artifacts, step_failures = _manifest_artifacts_for_dir(
            root / _ARTIFACTS_DIR,
            trial_root=root,
            trial_name=trial_name,
            job_label=job_label,
            step_index=0,
            step_name=None,
            source_prefix=source_prefix,
        )
        artifacts.extend(step_artifacts)
        failures.extend(step_failures)
    # Tool-call accounting reads trace-shaped files via their canonical host
    # source_path (which includes the artifacts/ or steps/<step>/artifacts/
    # prefix); it never introduces artifacts on its own.
    calls: Counter[str] = Counter()
    trace_paths = [
        root.parent / artifact["source_path"]
        for artifact in artifacts
        if any(
            marker in Path(artifact["artifact_path"]).name.lower()
            for marker in ("trajectory", "atif", "telemetry")
        )
    ]
    errors = sum(_read_trace(path, calls) for path in trace_paths)
    return artifacts, dict(sorted(calls.items())), errors, failures


def _trial_status(trial: dict[str, Any], exception: Any) -> str:
    raw = trial.get("status")
    if isinstance(raw, str) and raw in {"TIMEOUT", "CANCELLED"}:
        return raw
    if exception is not None:
        return "ERROR"
    return "COMPLETED"


def _normalize_trial(
    path: Path | None,
    trial: dict[str, Any],
    repetition: int,
    *,
    job_label: str,
    runtime: dict[str, Any] | None,
    source_prefix: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    agent_result = _object(trial.get("agent_result"))
    agent_info = _object(trial.get("agent_info"))
    model_info = _object(agent_info.get("model_info"))
    verifier = _object(trial.get("verifier_result"))
    rewards = _object(verifier.get("rewards"))
    exception = trial.get("exception_info")
    verifier_state = verifier.get("status")
    if not isinstance(verifier_state, str):
        verifier_state = verifier.get("state")
    if not isinstance(verifier_state, str):
        verifier_state = None
    artifacts, tool_calls, tool_errors, artifact_failures = _trial_artifacts(
        path,
        str(trial.get("trial_name", "")),
        job_label,
        source_prefix=source_prefix,
    )
    budgets: dict[str, Any] | None = None
    if runtime is not None:
        budgets = {
            "max_tokens": runtime.get("max_tokens"),
            "max_cost_usd": runtime.get("max_cost_usd"),
        }
    normalized = {
        "task": _task_id(trial.get("task_name")),
        "task_digest": "sha256:"
        + str(trial.get("task_checksum", "")).removeprefix("sha256:"),
        "repetition": repetition,
        "trial_name": str(trial.get("trial_name", "")),
        "pair_id": None,
        "status": _trial_status(trial, exception),
        "exception_type": exception.get("exception_type")
        if isinstance(exception, dict)
        else None,
        "model": model_info.get("name"),
        "model_provider": model_info.get("provider"),
        "agent": {
            "name": agent_info.get("name"),
            "version": agent_info.get("version"),
        },
        "rewards": rewards,
        "false_certification": rewards.get("false_certification"),
        "verifier_state": verifier_state,
        "tokens": {
            "input": agent_result.get("n_input_tokens"),
            "cache": agent_result.get("n_cache_tokens"),
            "output": agent_result.get("n_output_tokens"),
        },
        "cost_usd": agent_result.get("cost_usd"),
        "agent_seconds": _timing_seconds(trial.get("agent_execution")),
        "budgets": budgets,
        "artifacts": artifacts,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "raw_result_digest": _sha256(path) if path is not None else _json_digest(trial),
    }
    if runtime is not None and isinstance(runtime.get("pair_id"), str):
        normalized["pair_id"] = runtime["pair_id"]
    return normalized, artifact_failures


def _artifact_source_prefix(runtime: dict[str, Any] | None) -> str | None:
    if runtime is None:
        return None
    pair_id = runtime.get("pair_id")
    condition = runtime.get("condition")
    condition_id = condition.get("id") if isinstance(condition, dict) else None
    if isinstance(pair_id, str) and isinstance(condition_id, str):
        return f"{pair_id}/{condition_id}"
    return None


def _observation_failures(
    *,
    counters: Counter[str],
    expected_tasks: set[str],
    attempts: int,
    expected_digests: dict[str, str],
    trials: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if set(counters) != expected_tasks:
        failures.append(
            f"task coverage mismatch: expected={sorted(expected_tasks)}, observed={sorted(counters)}"
        )
    if attempts <= 0:
        failures.append("job n_attempts must be a positive integer")
    failures.extend(
        f"{task}: expected {attempts} repetitions, observed {counters[task]}"
        for task in sorted(expected_tasks)
        if attempts > 0 and counters[task] != attempts
    )
    failures.extend(
        f"{trial['task']} repetition {trial['repetition']}: task digest mismatch"
        for trial in trials
        if expected_digests.get(trial["task"]) is not None
        and trial["task_digest"] != expected_digests[trial["task"]]
    )
    stats = _object(payload.get("stats"))
    incomplete = any(
        stats.get(key, 0)
        for key in (
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
        )
    )
    if incomplete or any(trial["status"] != "COMPLETED" for trial in trials):
        failures.append("execution is incomplete or contains errors")
    return failures


def _artifact_source_reuse(trials: list[dict[str, Any]]) -> list[str]:
    """Reject the same canonical host source path reused across artifacts.

    ``source_path`` is the canonical host artifact file path, stable relative
    to the trial directory.  It differs across independent trials even when
    the Harbor manifest ``manifest_source`` (a container path) repeats.
    Identical bytes at distinct host source paths remain allowed; reusing the
    same host source path more than once is rejected.
    """

    seen: dict[str, str] = {}
    failures: list[str] = []
    for trial in trials:
        for artifact in trial["artifacts"]:
            source_path = artifact["source_path"]
            previous = seen.get(source_path)
            if previous is not None:
                failures.append(
                    f"artifact source path reused: {source_path} "
                    f"(trial {previous} and trial {trial['trial_name']})"
                )
            else:
                seen[source_path] = trial["trial_name"]
    return failures


def _resolve_binding(
    key: str,
    *,
    job: dict[str, Any],
    runtime: dict[str, Any],
    heldout_value: Any = None,
) -> tuple[Any, list[str]]:
    """Read an explicit binding from job/runtime/held-out and require agreement.

    Returns ``(value, failures)``.  ``value`` is the agreed binding or ``None``
    when missing.  Failures are recorded for missing bindings, mismatches, and
    invalid shapes.  The value is never invented from surrounding state.
    """

    job_value = job.get(key)
    runtime_value = runtime.get(key)
    candidates: list[tuple[str, Any]] = []
    if job_value is not None:
        candidates.append(("job", job_value))
    if runtime_value is not None:
        candidates.append(("runtime", runtime_value))
    if heldout_value is not None:
        candidates.append(("held-out manifest", heldout_value))
    if not candidates:
        return None, [f"{key} binding is missing from job, runtime, and manifest"]
    values = [value for _label, value in candidates]
    first = values[0]
    if any(value != first for value in values[1:]):
        labels = ", ".join(f"{label}={value!r}" for label, value in candidates)
        return first, [f"{key} bindings disagree: {labels}"]
    return first, []


def build_observation_evidence(
    *,
    dataset: str,
    condition: str,
    job_path: Path,
    jobs_dir: Path,
    result_path: Path | None = None,
    runtime_snapshot: dict[str, Any] | None = None,
    heldout_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    job = _read_json(job_path)
    if not isinstance(job, dict):
        raise HarborSuiteError("Harbor job must be an object")
    result_path = (result_path or _find_result(jobs_dir)).resolve()
    payload = _read_json(result_path)
    if not isinstance(payload, dict):
        raise HarborSuiteError("Harbor result must be an object")

    known_digests, task_dirs, dataset_path, evidence_class, dataset_id = (
        _selection_known(dataset, heldout_manifest)
    )
    expected_tasks, _mode, eval_args, selection_failures = _normalize_selection(
        job, known=known_digests, task_dirs=task_dirs, dataset_path=dataset_path
    )
    raw_attempts = job.get("n_attempts")
    attempts: int = (
        raw_attempts
        if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool)
        else 0
    )
    eval_args = dict(eval_args)
    eval_args["n_attempts"] = attempts
    eval_args["selection_digest"] = _json_digest(
        {
            "selection_mode": eval_args["selection_mode"],
            "datasets": eval_args["datasets"],
            "tasks": eval_args["tasks"],
            "selection": eval_args["selection"],
            "n_attempts": attempts,
        }
    )

    raw_trials = _trial_results(result_path, payload)
    raw_trials.sort(
        key=lambda pair: (
            _task_id(pair[1].get("task_name")),
            str(pair[1].get("trial_name", "")),
        )
    )
    counters: Counter[str] = Counter()
    trials: list[dict[str, Any]] = []
    artifact_failures: list[str] = []
    job_label = _display_path(job_path)
    source_prefix = _artifact_source_prefix(runtime_snapshot)
    for path, raw in raw_trials:
        task = _task_id(raw.get("task_name"))
        repetition = counters[task]
        if runtime_snapshot is not None and isinstance(
            runtime_snapshot.get("repetition"), int
        ):
            repetition = int(runtime_snapshot["repetition"])
        counters[task] += 1
        normalized, trial_artifact_failures = _normalize_trial(
            path,
            raw,
            repetition,
            job_label=job_label,
            runtime=runtime_snapshot,
            source_prefix=source_prefix,
        )
        artifact_failures.extend(trial_artifact_failures)
        trials.append(normalized)

    expected_digests = {
        task: known_digests[task] for task in expected_tasks if task in known_digests
    }
    failures: list[str] = []
    failures.extend(selection_failures)
    failures.extend(
        _observation_failures(
            counters=counters,
            expected_tasks=set(expected_tasks),
            attempts=attempts,
            expected_digests=expected_digests,
            trials=trials,
            payload=payload,
        )
    )
    failures.extend(artifact_failures)
    failures.extend(_artifact_source_reuse(trials))

    models = sorted(
        {str(trial["model"]) for trial in trials if trial["model"] is not None}
    )
    if len(models) != 1:
        failures.append(f"expected one recorded model identity, observed={models}")
    runtime = runtime_snapshot or {}
    if runtime.get("model") is not None and models != [runtime["model"]]:
        failures.append("recorded model differs from the frozen runtime snapshot")
    snapshot_invariants = {
        key: runtime.get(key)
        for key in (
            "bundle_id",
            "bundle_version",
            "bundle_manifest_digest",
            "dataset_manifest_digest",
            "harbor_version",
            "agent",
            "prompt_path",
            "prompt_digest",
            "reasoning_effort",
            "randomization_seed",
            "stage",
            "max_tokens",
            "max_cost_usd",
        )
        if key in runtime
    }
    heldout_harbor_version = (
        heldout_manifest.get("experiment", {}).get("harbor_version")
        if heldout_manifest is not None
        else None
    )
    heldout_snapshot_id = (
        heldout_manifest.get("dataset", {}).get("snapshot_id")
        if heldout_manifest is not None
        else None
    )
    snapshot_id, snapshot_failures = _resolve_binding(
        "snapshot_id",
        job=job,
        runtime=runtime,
        heldout_value=heldout_snapshot_id,
    )
    harbor_version, harbor_failures = _resolve_binding(
        "harbor_version",
        job=job,
        runtime=runtime,
        heldout_value=heldout_harbor_version,
    )
    failures.extend(snapshot_failures)
    failures.extend(harbor_failures)
    if not isinstance(snapshot_id, str) or not snapshot_id.startswith("sha256:"):
        failures.append(
            "snapshot_id must be a sha256 digest bound by the job or runtime"
        )
        snapshot_id = None
    if not isinstance(harbor_version, str) or not harbor_version:
        failures.append(
            "harbor_version must be a non-empty string bound by the job, runtime, or manifest"
        )
        harbor_version = None
    evidence = {
        "schema_version": "2",
        "evidence_class": evidence_class,
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INCOMPLETE",
        "source_sha": _git_sha(),
        "dataset": dataset_id,
        "condition": condition,
        "snapshot_id": snapshot_id,
        "harbor_version": harbor_version,
        "eval_args": eval_args,
        "job": {
            "path": job_label,
            "digest": _sha256(job_path),
            "comparison_signature": _json_digest(_comparison_job(job)),
            "n_attempts": attempts,
        },
        "runtime_snapshot": runtime,
        "fixed_invariants": {
            "model": models[0] if len(models) == 1 else None,
            "tasks": [
                {"task": task, "digest": expected_digests[task]}
                for task in sorted(expected_digests)
            ],
            "sampling_seed": None,
            "sampling_deterministic": False,
            "runtime": snapshot_invariants,
        },
        "result": {"path": _display_path(result_path), "digest": _sha256(result_path)},
        "trials": trials,
        "validation_failures": failures,
    }
    return evidence, failures


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _trial_metric(trial: dict[str, Any], metric: str) -> float | None:
    if metric in {
        "correctness",
        "evidence_validity",
        "scope_accuracy",
        "assurance_calibration",
        "reward",
    }:
        rewards = trial.get("rewards")
        return _number(rewards.get(metric)) if isinstance(rewards, dict) else None
    if metric == "false_certification":
        return _number(trial.get(metric))
    if metric in {"cost_usd", "agent_seconds"}:
        return _number(trial.get(metric))
    if metric.startswith("tokens."):
        tokens = trial.get("tokens")
        return (
            _number(tokens.get(metric.split(".", 1)[1]))
            if isinstance(tokens, dict)
            else None
        )
    return None


def _bootstrap_interval(deltas: list[float]) -> list[float] | None:
    if len(deltas) < 10:
        return None
    rng = random.Random(0)
    means = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(2000)
    )
    return [means[49], means[1949]]


def _mcnemar_exact(control: list[float], treatment: list[float]) -> float | None:
    discordant = [
        (left >= 1.0, right >= 1.0)
        for left, right in zip(control, treatment, strict=True)
        if left in {0.0, 1.0} and right in {0.0, 1.0}
    ]
    b = sum(left and not right for left, right in discordant)
    c = sum(not left and right for left, right in discordant)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return float(min(1.0, 2 * tail))


def _comparison_failures(
    control: dict[str, Any], treatment: dict[str, Any]
) -> list[str]:
    failures = [
        f"{name} evidence is not VALID"
        for name, value in (("control", control), ("treatment", treatment))
        if value.get("status") != "VALID"
    ]
    if (control.get("condition"), treatment.get("condition")) not in {
        ("control", "treatment"),
        ("C1", "C2"),
    }:
        failures.append("conditions must be a distinct control/treatment or C1/C2 pair")
    failures.extend(
        f"{name} evidence has an invalid public-claim boundary"
        for name, value in (("control", control), ("treatment", treatment))
        if value.get("causal_claim_authorized") is not False
    )
    failures.extend(
        f"fixed invariant differs: {key}"
        for key in ("source_sha", "dataset")
        if control.get(key) != treatment.get(key)
    )
    if control.get("fixed_invariants") != treatment.get("fixed_invariants"):
        failures.append("fixed invariants differ")
    if control.get("job", {}).get("comparison_signature") != treatment.get(
        "job", {}
    ).get("comparison_signature"):
        failures.append("job configuration differs outside the condition allowlist")
    classes = {control.get("evidence_class"), treatment.get("evidence_class")}
    if len(classes) != 1:
        failures.append("evidence classes differ")
    return failures


def _indexed_trials(value: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(item["task"]), int(item["repetition"])): item
        for item in value.get("trials", [])
    }


def _duplicate_pair_keys(value: dict[str, Any]) -> list[tuple[str, int]]:
    keys = [
        (str(item["task"]), int(item["repetition"])) for item in value.get("trials", [])
    ]
    return sorted(key for key, count in Counter(keys).items() if count > 1)


def _derived_comparison_class(
    control: dict[str, Any], treatment: dict[str, Any]
) -> str:
    classes = {control.get("evidence_class"), treatment.get("evidence_class")}
    if classes == {"held-out-comparative-evaluation"}:
        return "held-out-comparison"
    return "public-workflow-comparison"


def _metric_report(
    metric: str,
    pairs: list[tuple[str, int]],
    control_trials: dict[tuple[str, int], dict[str, Any]],
    treatment_trials: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    values = [
        (
            _trial_metric(control_trials[pair], metric),
            _trial_metric(treatment_trials[pair], metric),
        )
        for pair in pairs
    ]
    complete = [
        (left, right)
        for left, right in values
        if left is not None and right is not None
    ]
    left = [item[0] for item in complete]
    right = [item[1] for item in complete]
    deltas = [treatment - control for control, treatment in complete]
    return {
        "pair_count": len(deltas),
        "control_mean": sum(left) / len(left) if left else None,
        "treatment_mean": sum(right) / len(right) if right else None,
        "paired_delta": sum(deltas) / len(deltas) if deltas else None,
        "bootstrap_95_interval": _bootstrap_interval(deltas),
        "mcnemar_exact_p": _mcnemar_exact(left, right)
        if metric in {"correctness", "false_certification"} and left
        else None,
        "interpretation": "descriptive-small-sample"
        if len(deltas) < 10
        else "comparative",
    }


def compare_evidence(
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, Any]:
    _validate_contract(control, "observation-evidence.schema.json")
    _validate_contract(treatment, "observation-evidence.schema.json")
    failures = _comparison_failures(control, treatment)
    for name, value in (("control", control), ("treatment", treatment)):
        duplicates = _duplicate_pair_keys(value)
        if duplicates:
            failures.append(f"{name} evidence has duplicate task/repetition pairs")
    control_trials = _indexed_trials(control)
    treatment_trials = _indexed_trials(treatment)
    if set(control_trials) != set(treatment_trials):
        failures.append("control/treatment trials do not pair exactly")
    pairs = sorted(set(control_trials) & set(treatment_trials))
    metric_names = (
        "correctness",
        "evidence_validity",
        "scope_accuracy",
        "assurance_calibration",
        "false_certification",
        "reward",
        "tokens.input",
        "tokens.output",
        "cost_usd",
        "agent_seconds",
    )
    metrics = {
        metric: _metric_report(metric, pairs, control_trials, treatment_trials)
        for metric in metric_names
    }
    for metric in ("correctness", "false_certification"):
        if metrics[metric]["pair_count"] != len(pairs):
            failures.append(f"core metric is missing from a complete pair: {metric}")
    return {
        "schema_version": "1",
        "evidence_class": _derived_comparison_class(control, treatment),
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INVALID",
        "dataset": control.get("dataset"),
        "source_sha": control.get("source_sha"),
        "conditions": {
            "control": control.get("condition"),
            "treatment": treatment.get("condition"),
        },
        "pair_count": len(pairs),
        "metrics": metrics,
        "validation_failures": failures,
    }


def _heldout_plan_failures(
    plan: dict[str, Any], ledger: dict[str, Any], condition: str
) -> tuple[list[dict[str, Any]], list[str]]:
    from benchmarks.tooling.heldout_runner import _plan_digest

    failures: list[str] = []
    plan_digest = _plan_digest(plan)
    if plan.get("plan_digest") != plan_digest:
        failures.append("held-out plan digest mismatch")
    if ledger.get("plan_digest") != plan_digest:
        failures.append("held-out ledger does not bind the run plan")
    if ledger.get("status") != "COMPLETE":
        failures.append("held-out execution ledger is not COMPLETE")
    selected = [
        run for run in plan.get("runs", []) if run.get("condition") == condition
    ]
    if len(selected) != plan.get("pair_count"):
        failures.append(f"held-out {condition} run coverage is incomplete")
    return selected, failures


def _collect_heldout_runs(
    *,
    selected: list[dict[str, Any]],
    root: Path,
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    trials: list[dict[str, Any]] = []
    signatures: list[dict[str, str]] = []
    failures: list[str] = []
    for run in selected:
        runtime_path = root / run["runtime_snapshot"]
        runtime = _read_json(runtime_path)
        if not isinstance(runtime, dict):
            failures.append(f"invalid runtime snapshot: {run['pair_id']}")
            continue
        evidence, run_failures = build_observation_evidence(
            dataset=manifest["dataset"]["id"],
            condition=condition,
            job_path=root / run["job"],
            jobs_dir=root / run["jobs_dir"],
            runtime_snapshot=runtime,
            heldout_manifest=manifest,
        )
        failures.extend(f"{run['pair_id']}: {failure}" for failure in run_failures)
        run_id = f"{run['pair_id']}/{condition}"
        ledger_run = ledger.get("runs", {}).get(run_id)
        if not isinstance(ledger_run, dict) or ledger_run.get("status") != "COMPLETE":
            failures.append(f"held-out ledger run is not COMPLETE: {run_id}")
        elif ledger_run.get("result_digest") != evidence["result"]["digest"]:
            failures.append(f"held-out result digest differs from ledger: {run_id}")
        trials.extend(evidence["trials"])
        signatures.append(
            {
                "pair_id": str(run["pair_id"]),
                "signature": str(evidence["job"]["comparison_signature"]),
            }
        )
    trials.sort(key=lambda item: (str(item["task"]), int(item["repetition"])))
    return trials, signatures, failures


def _heldout_runtime_invariants(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: plan.get(key)
        for key in (
            "bundle_manifest_digest",
            "snapshot_id",
            "harbor_version",
            "agent",
            "model",
            "prompt_path",
            "prompt_digest",
            "reasoning_effort",
            "randomization_seed",
            "stage",
            "budget",
            "pair_count",
            "plan_digest",
        )
        if key in plan
    }


def collect_heldout_evidence(
    *,
    run_plan_path: Path,
    manifest_path: Path,
    ledger_path: Path,
    condition: str,
) -> tuple[dict[str, Any], list[str]]:
    from benchmarks.tooling.heldout_bundle import validate_manifest

    plan = _read_json(run_plan_path)
    manifest = validate_manifest(manifest_path)
    ledger = _read_json(ledger_path)
    if not isinstance(plan, dict) or not isinstance(ledger, dict):
        raise HarborSuiteError("held-out plan and ledger must be JSON objects")
    selected, failures = _heldout_plan_failures(plan, ledger, condition)
    trials, signatures, run_failures = _collect_heldout_runs(
        selected=selected,
        root=run_plan_path.parent,
        manifest=manifest,
        ledger=ledger,
        condition=condition,
    )
    failures.extend(run_failures)
    experiment = manifest["experiment"]
    stage = str(plan.get("stage", ""))
    stage_tasks = (
        experiment["stages"][stage]["task_ids"] if stage in experiment["stages"] else []
    )
    runtime_invariants = _heldout_runtime_invariants(plan)
    models = sorted(
        {str(item["model"]) for item in trials if item.get("model") is not None}
    )
    if models != [experiment["model"]]:
        failures.append("observed model does not match the frozen experiment")
    task_digests = {str(item["id"]): str(item["digest"]) for item in manifest["tasks"]}
    failures.extend(_artifact_source_reuse(trials))
    heldout_harbor_version = experiment.get("harbor_version")
    bundle_id = manifest.get("bundle_id")
    bundle_version = manifest.get("bundle_version")
    manifest_snapshot_id = manifest.get("dataset", {}).get("snapshot_id")
    snapshot_id, snapshot_failures = _resolve_binding(
        "snapshot_id",
        job=plan,
        runtime=runtime_invariants,
        heldout_value=manifest_snapshot_id,
    )
    harbor_version, harbor_failures = _resolve_binding(
        "harbor_version",
        job=plan,
        runtime=runtime_invariants,
        heldout_value=heldout_harbor_version,
    )
    failures.extend(snapshot_failures)
    failures.extend(harbor_failures)
    if not isinstance(snapshot_id, str) or not snapshot_id.startswith("sha256:"):
        failures.append(
            "snapshot_id must be a sha256 digest bound by the plan, runtime, or manifest dataset"
        )
        snapshot_id = None
    if not isinstance(harbor_version, str) or not harbor_version:
        failures.append(
            "harbor_version must be a non-empty string bound by the plan, runtime, or manifest experiment"
        )
        harbor_version = None
    if bundle_id is not None and isinstance(bundle_id, str):
        # Surface the bundle identity in the runtime snapshot for traceability.
        runtime_invariants.setdefault("bundle_id", bundle_id)
    if bundle_version is not None and isinstance(bundle_version, str):
        runtime_invariants.setdefault("bundle_version", bundle_version)
    eval_args = _eval_args(
        "dataset-task-names",
        sorted(stage_tasks),
        [{"path": manifest["dataset"]["path"], "task_names": sorted(stage_tasks)}],
        None,
        len(selected),
    )
    evidence = {
        "schema_version": "2",
        "evidence_class": "held-out-comparative-evaluation",
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INCOMPLETE",
        "source_sha": _git_sha(),
        "dataset": manifest["dataset"]["id"],
        "condition": condition,
        "snapshot_id": snapshot_id,
        "harbor_version": harbor_version,
        "eval_args": eval_args,
        "job": {
            "path": "run-plan.json",
            "digest": _sha256(run_plan_path),
            "comparison_signature": _json_digest(
                sorted(signatures, key=lambda item: item["pair_id"])
            ),
            "n_attempts": len(selected),
        },
        "runtime_snapshot": runtime_invariants,
        "fixed_invariants": {
            "model": models[0] if len(models) == 1 else None,
            "tasks": [
                {"task": task, "digest": task_digests[task]}
                for task in sorted(stage_tasks)
            ],
            "sampling_seed": experiment["randomization_seed"],
            "sampling_deterministic": False,
            "runtime": runtime_invariants,
        },
        "result": {"path": ledger_path.name, "digest": _sha256(ledger_path)},
        "trials": trials,
        "validation_failures": failures,
    }
    return evidence, failures


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Jacobian workflow comparison",
        "",
        f"Status: **{report['status']}**. This report remains evaluation evidence; it does not itself authorize a causal capability claim.",
        "",
        "| Metric | Pairs | Control | Treatment | Paired delta | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, metric in report["metrics"].items():

        def fmt(value: Any) -> str:
            return "unknown" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| {name} | {metric['pair_count']} | {fmt(metric['control_mean'])} | {fmt(metric['treatment_mean'])} | {fmt(metric['paired_delta'])} | {metric['interpretation']} |"
        )
    if report["validation_failures"]:
        lines.extend(["", "## Validation failures", ""])
        lines.extend(f"- {failure}" for failure in report["validation_failures"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset", required=True)
    validate_parser.add_argument("--condition", required=True)
    validate_parser.add_argument("--job", type=Path, required=True)
    validate_parser.add_argument("--jobs-dir", type=Path, required=True)
    validate_parser.add_argument("--result", type=Path)
    validate_parser.add_argument("--runtime-snapshot", type=Path)
    validate_parser.add_argument("--heldout-manifest", type=Path)
    validate_parser.add_argument("--output", type=Path, required=True)
    collect_parser = subparsers.add_parser("collect-heldout")
    collect_parser.add_argument("--run-plan", type=Path, required=True)
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--ledger", type=Path, required=True)
    collect_parser.add_argument("--condition", choices=("C1", "C2"), required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--control", type=Path, required=True)
    compare_parser.add_argument("--treatment", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        runtime = _read_json(args.runtime_snapshot) if args.runtime_snapshot else None
        heldout = _read_json(args.heldout_manifest) if args.heldout_manifest else None
        evidence, failures = build_observation_evidence(
            dataset=args.dataset,
            condition=args.condition,
            job_path=args.job,
            jobs_dir=args.jobs_dir,
            result_path=args.result,
            runtime_snapshot=runtime,
            heldout_manifest=heldout,
        )
        _validate_contract(evidence, "observation-evidence.schema.json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(args.output)
        return 1 if failures else 0
    if args.command == "collect-heldout":
        evidence, failures = collect_heldout_evidence(
            run_plan_path=args.run_plan,
            manifest_path=args.manifest,
            ledger_path=args.ledger,
            condition=args.condition,
        )
        _validate_contract(evidence, "observation-evidence.schema.json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(args.output)
        return 1 if failures else 0
    control = _read_json(args.control)
    treatment = _read_json(args.treatment)
    if not isinstance(control, dict) or not isinstance(treatment, dict):
        raise HarborSuiteError("comparison inputs must be JSON objects")
    report = compare_evidence(control, treatment)
    _validate_contract(report, "comparison-report.schema.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "comparison-report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(args.output)
    return 0 if report["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_observation_evidence",
    "collect_heldout_evidence",
    "compare_evidence",
    "render_markdown",
]
