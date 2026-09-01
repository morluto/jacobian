"""Manifest-bound artifact identity and trace accounting for observations.

This module owns the filesystem boundary for observation evidence.  It reads
Harbor artifact manifests, validates their paths and statuses, binds each
regular file to its source identity, and accounts for tool calls in trace
artifacts.  Evidence assembly consumes these typed results without owning
manifest semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.tooling.errors import HarborSuiteError

_MCP_TOOL_CALL = re.compile(
    r"\bMCP tool call tool=(math\.(?:find|run))\b"
    r".{0,512}?\bstatus=(success|error)\b"
    r".{0,512}?\brequest_digest=([0-9a-f]{16}|none)\b"
    r".{0,512}?\bargument_digest=(sha256:(?:[0-9a-f]\s*){64})",
    re.DOTALL,
)
_MCP_OPERATION_ATTEMPT = re.compile(
    r"\bMCP operation attempt argument_digest=(sha256:(?:[0-9a-f]\s*){64})"
    r".{0,2048}?\bexecution_status=([A-Z_]+)\b",
    re.DOTALL,
)


def _canonical_tool_name(value: str) -> str:
    aliases = {
        "mcp__jacobian__math_find": "math.find",
        "mcp__jacobian__math_run": "math.run",
    }
    return aliases.get(value, value)


def _read_mcp_runtime_log(path: Path, calls: Counter[str]) -> int:
    """Count server-observed MCP calls and failed mathematical attempts."""

    text = path.read_text(encoding="utf-8")
    failed_attempts: Counter[str] = Counter()
    transport_errors: Counter[str] = Counter()
    for match in _MCP_OPERATION_ATTEMPT.finditer(text):
        argument_digest, execution_status = match.groups()
        if execution_status != "COMPLETED":
            failed_attempts[re.sub(r"\s", "", argument_digest)] += 1
    for match in _MCP_TOOL_CALL.finditer(text):
        tool, status, _request_digest, argument_digest = match.groups()
        calls[tool] += 1
        if status == "error":
            transport_errors[re.sub(r"\s", "", argument_digest)] += 1
    argument_digests = set(failed_attempts) | set(transport_errors)
    return sum(
        max(failed_attempts[digest], transport_errors[digest])
        for digest in argument_digests
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_trace(
    value: Any,
    calls: Counter[str],
    *,
    ignored_tools: frozenset[str] = frozenset(),
) -> int:
    if isinstance(value, list):
        return sum(
            _walk_trace(child, calls, ignored_tools=ignored_tools) for child in value
        )
    if not isinstance(value, dict):
        return 0
    observed_tool: str | None = None
    tool_name = value.get("tool_name")
    if isinstance(tool_name, str):
        observed_tool = _canonical_tool_name(tool_name)
    elif value.get("type") in {"tool_call", "tool_use"} and isinstance(
        value.get("name"), str
    ):
        observed_tool = _canonical_tool_name(str(value["name"]))
    elif value.get("type") == "mcp_tool_call" and isinstance(value.get("tool"), str):
        observed_tool = _canonical_tool_name(str(value["tool"]))
    function_name = value.get("function_name")
    if isinstance(function_name, str) and isinstance(value.get("tool_call_id"), str):
        observed_tool = _canonical_tool_name(function_name)
    if observed_tool in ignored_tools:
        return 0
    if observed_tool is not None:
        calls[observed_tool] += 1
    own_error = int(
        value.get("error") not in {None, False, ""} or value.get("isError") is True
    )
    return own_error + sum(
        _walk_trace(child, calls, ignored_tools=ignored_tools)
        for child in value.values()
    )


def _read_trace(
    path: Path,
    calls: Counter[str],
    *,
    ignored_tools: frozenset[str] = frozenset(),
    mcp_runtime_log: bool = False,
) -> int:
    try:
        if mcp_runtime_log:
            return _read_mcp_runtime_log(path, calls)
        if path.suffix == ".jsonl":
            return sum(
                _walk_trace(json.loads(line), calls, ignored_tools=ignored_tools)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if path.suffix == ".json":
            return _walk_trace(_read_json(path), calls, ignored_tools=ignored_tools)
    except (OSError, UnicodeError, json.JSONDecodeError, HarborSuiteError):
        return 1
    return 0


_MANIFEST_FILENAME = "manifest.json"
_ARTIFACTS_DIR = "artifacts"
_STEPS_DIR = "steps"
_MANIFEST_STATUSES_OK = {"ok"}
_MANIFEST_STATUSES_NON_CONCLUSION = {"failed", "empty", "skipped"}
_MANIFEST_STATUSES = _MANIFEST_STATUSES_OK | _MANIFEST_STATUSES_NON_CONCLUSION
_HARBOR_CONVENTION_SOURCE = "/logs/artifacts"
_HARBOR_CONVENTION_DESTINATION = "artifacts/logs/artifacts"


def _is_empty_harbor_convention(
    entry: dict[str, Any], configured_artifacts: set[tuple[str, str | None]]
) -> bool:
    """Identify Harbor's implicit, optional agent artifact directory."""

    return (
        entry.get("source") == _HARBOR_CONVENTION_SOURCE
        and entry.get("destination") == _HARBOR_CONVENTION_DESTINATION
        and entry.get("type") == "directory"
        and entry.get("status") == "empty"
        and entry.get("service") is None
        and (_HARBOR_CONVENTION_SOURCE, None) not in configured_artifacts
    )


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
    configured_artifacts: set[tuple[str, str | None]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    location = f"trial {trial_name} manifest[{index}]"
    failures = _validate_manifest_entry(entry, location=location)
    if not isinstance(entry, dict):
        return [], failures
    status = entry.get("status")
    # Harbor 0.20 always injects its conventional publish directory. Agents do
    # not have to use it, so its exact `empty` record carries no failed claim.
    if not failures and _is_empty_harbor_convention(
        entry, configured_artifacts or set()
    ):
        return [], []
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
    configured_artifacts: set[tuple[str, str | None]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read one ``artifacts/manifest.json`` and bind every ``ok`` entry.

    Harbor 0.20 writes ``manifest.json`` as a JSON array of entries
    ``{source, destination, type, status, service}``.  ``destination`` is the
    artifact-relative path (``artifacts/<relative>``); ``source`` is the
    canonical container source path. Only ``ok`` entries have a collected
    file on disk; ``failed``/``empty``/``skipped`` are non-conclusion records
    that are surfaced as failures because evidence is incomplete, except for
    Harbor's implicit optional convention directory when it is unused.
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
    configured = configured_artifacts or set()
    observed = {
        (
            entry.get("source"),
            None if entry.get("service") in {None, "main"} else entry.get("service"),
        )
        for entry in raw
        if isinstance(entry, dict)
        and isinstance(entry.get("source"), str)
        and (entry.get("service") is None or isinstance(entry.get("service"), str))
    }
    for source, service in sorted(
        configured - observed, key=lambda item: (item[0], item[1] or "")
    ):
        failures.append(
            f"trial {trial_name}: configured artifact is missing from manifest "
            f"(source={source!r}, service={service!r})"
        )
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
            configured_artifacts=configured_artifacts,
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


def _artifact_host_path(root: Path, artifact: dict[str, Any]) -> Path:
    step_name = artifact.get("step_name")
    base = (
        root / _STEPS_DIR / str(step_name) / _ARTIFACTS_DIR
        if isinstance(step_name, str)
        else root / _ARTIFACTS_DIR
    )
    return base / str(artifact["artifact_path"])


def trial_artifacts(
    trial_path: Path | None,
    trial_name: str,
    job_label: str,
    *,
    source_prefix: str | None = None,
    configured_artifacts: set[tuple[str, str | None]] | None = None,
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
        inline_failures = [
            f"trial {trial_name}: configured artifact is missing without a "
            f"trial manifest (source={source!r}, service={service!r})"
            for source, service in sorted(
                configured_artifacts or set(),
                key=lambda item: (item[0], item[1] or ""),
            )
        ]
        return [], {}, 0, inline_failures
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
                configured_artifacts=configured_artifacts,
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
            configured_artifacts=configured_artifacts,
        )
        artifacts.extend(step_artifacts)
        failures.extend(step_failures)
    # Tool-call accounting reads trace-shaped files via their canonical host
    # source_path (which includes the artifacts/ or steps/<step>/artifacts/
    # prefix); it never introduces artifacts on its own.
    calls: Counter[str] = Counter()
    runtime_logs = [
        _artifact_host_path(root, artifact)
        for artifact in artifacts
        if artifact.get("service") == "jacobian"
        and artifact.get("manifest_source") == "/logs/jacobian/mcp.log"
    ]
    agent_traces = [
        _artifact_host_path(root, artifact)
        for artifact in artifacts
        if artifact.get("service") != "jacobian"
        and any(
            marker in Path(artifact["artifact_path"]).name.lower()
            for marker in ("trajectory", "atif", "telemetry")
        )
    ]
    ignored_tools = (
        frozenset({"math.find", "math.run"}) if runtime_logs else frozenset()
    )
    errors = sum(
        _read_trace(path, calls, ignored_tools=ignored_tools) for path in agent_traces
    )
    errors += sum(
        _read_trace(path, calls, mcp_runtime_log=True) for path in runtime_logs
    )
    return artifacts, dict(sorted(calls.items())), errors, failures


def artifact_source_reuse(trials: list[dict[str, Any]]) -> list[str]:
    """Reject reuse of one canonical host source path across trials."""

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


__all__ = [
    "artifact_source_reuse",
    "trial_artifacts",
]
