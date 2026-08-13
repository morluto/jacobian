"""Normalize and compare Harbor model-in-the-loop observation results.

This is the strict normalized observation evidence *v4* implementation.  It
replaces v2 atomically and tightens three classes of contract that v2 left
implicit:

* **Dataset/task selection** is normalized to exactly one of the two Harbor job
  forms -- ``datasets[].path`` with optional ``task_names`` or explicit
  ``tasks[].path``.  Mixed selections, outside-dataset paths, unknown task
  names, empty selections, and the v2 implicit "fall back to all known tasks"
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
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling import (
    observation_artifacts,
    observation_comparison,
    observation_selection,
)
from benchmarks.tooling.command_runner import git_head_sha
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import (
    ROOT,
    get_suite,
    task_digest,
)
from benchmarks.tooling.strict_boundaries import (
    HarborJobSelection,
    raise_strict_model,
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
    value = git_head_sha(ROOT)
    if value is None:
        raise HarborSuiteError("unable to resolve git HEAD")
    return value


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


def _completed_job_stats(job_stats: dict[str, Any], observed_trial_count: int) -> bool:
    if (
        not isinstance(observed_trial_count, int)
        or isinstance(observed_trial_count, bool)
        or observed_trial_count <= 0
    ):
        return False
    completed = job_stats.get("n_completed_trials")
    total = job_stats.get("n_total_trials")
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or completed <= 0
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or completed != total
        or total != observed_trial_count
    ):
        return False
    return all(
        isinstance(job_stats.get(key), int)
        and not isinstance(job_stats.get(key), bool)
        and job_stats[key] == 0
        for key in (
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
        )
    )


def _comparison_job(job: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the frozen Jacobian treatment additions.

    Any other condition-specific Compose or MCP change remains in the
    comparison signature and therefore invalidates the pair.
    """

    normalized: dict[str, Any] = json.loads(json.dumps(job))
    normalized.pop("jobs_dir", None)
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, list):
        normalized["artifacts"] = [
            entry
            for entry in artifacts
            if entry != {"source": "/logs/jacobian/mcp.log", "service": "jacobian"}
        ]
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
                    not in (
                        {
                            "name": "jacobian",
                            "transport": "streamable-http",
                            "url": "http://127.0.0.1:8000/mcp",
                        },
                        {
                            "name": "jacobian",
                            "transport": "streamable-http",
                            "url": "http://jacobian:8000/mcp",
                        },
                    )
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


_VALID_STATUSES = frozenset(
    {"COMPLETED", "RUNNING", "PENDING", "FAILED", "ERROR", "TIMEOUT", "CANCELLED"}
)


def _resolve_verifier_status(verifier_result: dict[str, Any]) -> tuple[Any, str | None]:
    """Return ``(raw_value, error)`` from the verifier result status/state keys."""
    for key in ("status", "state"):
        if key in verifier_result:
            value = verifier_result[key]
            if value is not None and not isinstance(value, str):
                return value, "ERROR"
            if value is not None:
                return value, None
    return None, None


def _trial_status(
    trial: dict[str, Any],
    exception: Any,
    *,
    job_stats: dict[str, Any] | None = None,
    observed_trial_count: int | None = None,
) -> str:
    raw = trial.get("status")
    if exception is not None:
        return "ERROR"
    if isinstance(raw, str) and raw in _VALID_STATUSES:
        return raw
    if raw is not None:
        return "ERROR"
    verifier_result = _object(trial.get("verifier_result"))
    verifier_status, error = _resolve_verifier_status(verifier_result)
    if error is not None:
        return error
    if isinstance(verifier_status, str) and verifier_status in _VALID_STATUSES:
        if verifier_status == "COMPLETED" and (
            job_stats is None
            or observed_trial_count is None
            or not _completed_job_stats(job_stats, observed_trial_count)
        ):
            return "ERROR"
        return verifier_status
    if verifier_status is not None:
        return "ERROR"
    if (
        job_stats is not None
        and observed_trial_count is not None
        and _completed_job_stats(job_stats, observed_trial_count)
    ):
        return "COMPLETED"
    return "ERROR"


def _normalize_trial(
    path: Path | None,
    trial: dict[str, Any],
    repetition: int,
    *,
    job_label: str,
    runtime: dict[str, Any] | None,
    source_prefix: str | None = None,
    configured_artifacts: set[tuple[str, str | None]] | None = None,
    job_stats: dict[str, Any] | None = None,
    observed_trial_count: int | None = None,
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
    artifacts, tool_calls, tool_errors, artifact_failures = (
        observation_artifacts.trial_artifacts(
            path,
            str(trial.get("trial_name", "")),
            job_label,
            source_prefix=source_prefix,
            configured_artifacts=configured_artifacts,
        )
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
        "status": _trial_status(
            trial,
            exception,
            job_stats=job_stats,
            observed_trial_count=observed_trial_count,
        ),
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


def _configured_artifacts(job: dict[str, Any]) -> set[tuple[str, str | None]]:
    configured: set[tuple[str, str | None]] = set()
    artifacts = job.get("artifacts")
    if not isinstance(artifacts, list):
        return configured
    for artifact in artifacts:
        if isinstance(artifact, str):
            configured.add((artifact, None))
        elif isinstance(artifact, dict) and isinstance(artifact.get("source"), str):
            service = artifact.get("service")
            configured.add(
                (artifact["source"], None if service in {None, "main"} else service)
            )
    return configured


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
    stats = dict(_object(payload.get("stats")))
    if "n_total_trials" not in stats:
        stats["n_total_trials"] = payload.get("n_total_trials")
    if not _completed_job_stats(stats, len(trials)):
        failures.append(
            "execution completion counts are missing, malformed, or disagree with observed trials"
        )
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


def _parse_job_selection(job_path: Path) -> dict[str, Any]:
    """Read and strictly validate the selection root of a Harbor job."""

    job = _read_json(job_path)
    if not isinstance(job, dict):
        raise HarborSuiteError("Harbor job must be an object")
    selection_payload: dict[str, Any] = {}
    if "datasets" in job:
        selection_payload["datasets"] = job["datasets"]
    if "tasks" in job:
        selection_payload["tasks"] = job["tasks"]
    raise_strict_model(
        HarborJobSelection,
        selection_payload,
        label=_display_path(job_path),
    )
    return job


def _resolve_binding_values(
    job: dict[str, Any],
    runtime: dict[str, Any],
    heldout_manifest: dict[str, Any] | None,
) -> tuple[object | None, object | None, list[str]]:
    """Resolve snapshot_id and harbor_version bindings across job/runtime/manifest."""

    heldout_harbor_version = (
        heldout_manifest.get("experiment", {}).get("harbor_version")
        if heldout_manifest is not None
        else None
    )
    heldout_snapshot_id = (
        heldout_manifest.get("snapshot_lock", {}).get("lock_id")
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
    return snapshot_id, harbor_version, snapshot_failures + harbor_failures


def _jacobian_image_failures(runtime: dict[str, Any]) -> list[str]:
    condition = runtime.get("condition")
    if not isinstance(condition, dict) or condition.get("jacobian_enabled") is not True:
        return []
    image = runtime.get("jacobian_image")
    if not isinstance(image, dict):
        return ["Jacobian-enabled runtime snapshot must bind jacobian_image"]
    failures: list[str] = []
    if image.get("source_dirty") is not False:
        failures.append("Jacobian image must come from a clean source revision")
    digest_reference = image.get("digest_reference")
    if not isinstance(digest_reference, str) or not re.fullmatch(
        r"[^@]+@sha256:[0-9a-f]{64}", digest_reference
    ):
        failures.append("Jacobian image must be bound by an OCI digest")
    if not isinstance(image.get("platform"), str) or "/" not in image["platform"]:
        failures.append("Jacobian image platform is not bound")
    if (
        not isinstance(image.get("jacobian_package_version"), str)
        or not image["jacobian_package_version"]
    ):
        failures.append("Jacobian package version is not bound")
    return failures


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
    job = _parse_job_selection(job_path)
    result_path = (result_path or _find_result(jobs_dir)).resolve()
    payload = _read_json(result_path)
    if not isinstance(payload, dict):
        raise HarborSuiteError("Harbor result must be an object")

    known_digests, task_dirs, dataset_path, evidence_class, dataset_id = (
        observation_selection.selection_known(
            dataset,
            heldout_manifest,
            get_suite_fn=get_suite,
            task_digest_fn=task_digest,
        )
    )
    expected_tasks, _mode, eval_args, selection_failures = (
        observation_selection.normalize_selection(
            job,
            known=known_digests,
            task_dirs=task_dirs,
            dataset_path=dataset_path,
            root=ROOT,
        )
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
    job_stats = dict(_object(payload.get("stats")))
    if "n_total_trials" not in job_stats:
        job_stats["n_total_trials"] = payload.get("n_total_trials")
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
    configured_artifacts = _configured_artifacts(job)
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
            configured_artifacts=configured_artifacts,
            job_stats=job_stats,
            observed_trial_count=len(raw_trials),
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
    failures.extend(observation_artifacts.artifact_source_reuse(trials))

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
            "manifest_digest",
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
    snapshot_id, harbor_version, binding_failures = _resolve_binding_values(
        job, runtime, heldout_manifest
    )
    failures.extend(binding_failures)
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
    source_sha = _git_sha()
    failures.extend(_jacobian_image_failures(runtime))
    evidence = {
        "schema_version": "4",
        "evidence_class": evidence_class,
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INCOMPLETE",
        "source_sha": source_sha,
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
    if ledger.get("manifest_digest") != plan.get("manifest_digest"):
        failures.append("held-out ledger does not bind the canonical manifest")
    if ledger.get("status") != "COMPLETE":
        failures.append("held-out execution ledger is not COMPLETE")
    runs = plan.get("runs")
    if not isinstance(runs, list):
        failures.append("held-out plan runs must be a list")
        return [], failures
    malformed = [index for index, run in enumerate(runs) if not isinstance(run, dict)]
    if malformed:
        failures.extend(
            f"held-out plan runs[{index}] must be an object" for index in malformed
        )
    selected = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("condition") == condition
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
            "manifest_digest",
            "stage",
            "budget",
            "pair_count",
            "plan_digest",
        )
        if key in plan
    }


_MATH_RUN_TOOL = "math.run"


def _mark_invoked_if_operation_used(
    ledger: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    contract_dir: Path,
) -> bool:
    """Transition treatment routing_status to AVAILABLE_INVOKED when
    a successful Jacobian operation invocation is observed during
    normalization.  Only actual observed tool_calls evidence this; the
    preflight proves AVAILABLE_UNUSED and the runner must not override it.
    Persists the updated contract to ``routing-status-c2.json``.

    Fail closed: a trial with ``math.run`` calls but non-COMPLETED
    status or nonzero ``tool_errors`` does not evidence a successful
    invocation and must not transition the routing status.
    """

    invoked = any(
        isinstance(trial.get("tool_calls"), dict)
        and trial["tool_calls"].get(_MATH_RUN_TOOL, 0) > 0
        and trial.get("status") == "COMPLETED"
        and trial.get("tool_errors") == 0
        for trial in trials
    )
    if not invoked:
        return False
    routing = ledger.get("routing_status")
    if not isinstance(routing, dict):
        return False
    c2 = routing.get("C2")
    if not isinstance(c2, dict):
        return False
    if c2.get("routing_status") != "AVAILABLE_UNUSED":
        return False
    c2["routing_status"] = "AVAILABLE_INVOKED"
    from benchmarks.tooling.heldout_runner import _write_routing_contract

    _write_routing_contract(contract_dir, "C2", c2)
    return True


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
    if condition == "C2":
        routing_changed = _mark_invoked_if_operation_used(
            ledger, trials, contract_dir=ledger_path.parent
        )
        if routing_changed:
            from benchmarks.tooling.heldout_runner import _write_ledger

            _write_ledger(ledger_path, ledger)
    stage = str(plan.get("stage", ""))
    stage_tasks = (
        experiment["stages"][stage]["task_ids"] if stage in experiment["stages"] else []
    )
    runtime_invariants = _heldout_runtime_invariants(plan)
    evidence_runtime = dict(runtime_invariants)
    if condition == "C2":
        treatment = next(item for item in manifest["conditions"] if item["id"] == "C2")
        evidence_runtime["jacobian_image"] = {
            "source_sha": treatment["source_sha"],
            "source_dirty": False,
            "reference": treatment["image"],
            "digest_reference": treatment["image"],
            "platform": treatment["platform"],
            "jacobian_package_version": treatment["server_version"],
        }
    models = sorted(
        {str(item["model"]) for item in trials if item.get("model") is not None}
    )
    if models != [experiment["model"]]:
        failures.append("observed model does not match the frozen experiment")
    task_digests = {str(item["id"]): str(item["digest"]) for item in manifest["tasks"]}
    failures.extend(observation_artifacts.artifact_source_reuse(trials))
    heldout_harbor_version = experiment.get("harbor_version")
    manifest_snapshot_id = manifest.get("snapshot_lock", {}).get("lock_id")
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
            "snapshot_id must be a sha256 digest bound by the plan, runtime, or manifest snapshot_lock"
        )
        snapshot_id = None
    if not isinstance(harbor_version, str) or not harbor_version:
        failures.append(
            "harbor_version must be a non-empty string bound by the plan, runtime, or manifest experiment"
        )
        harbor_version = None
    eval_args = observation_selection.make_eval_args(
        "dataset-task-names",
        sorted(stage_tasks),
        [{"path": manifest["dataset"]["path"], "task_names": sorted(stage_tasks)}],
        None,
        len(selected),
    )
    evidence = {
        "schema_version": "4",
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
        "runtime_snapshot": evidence_runtime,
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
    report = observation_comparison.compare_evidence(control, treatment)
    _validate_contract(report, "comparison-report.schema.json")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "comparison-report.md").write_text(
        observation_comparison.render_markdown(report), encoding="utf-8"
    )
    print(args.output)
    return 0 if report["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_observation_evidence",
    "collect_heldout_evidence",
]
