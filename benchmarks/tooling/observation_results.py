"""Normalize and compare Harbor model-in-the-loop observation results."""

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


def _expected_tasks(job: dict[str, Any], known: set[str]) -> set[str]:
    selected: set[str] = set()
    for item in job.get("datasets", []):
        if not isinstance(item, dict):
            continue
        names = item.get("task_names")
        selected.update(known if names is None else names)
    return selected or known


def _comparison_job(job: dict[str, Any]) -> dict[str, Any]:
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
    error_value = value.get("error")
    own_error = int(
        error_value is not None and error_value is not False and error_value != ""
    )
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


def _trace_summary(trial_path: Path | None) -> dict[str, Any]:
    if trial_path is None:
        return {"artifacts": [], "tool_calls": {}, "tool_errors": 0}
    root = trial_path.parent
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and any(
            marker in path.name.lower()
            for marker in ("trajectory", "atif", "telemetry")
        )
    )
    calls: Counter[str] = Counter()
    artifacts = [
        {"path": path.relative_to(root).as_posix(), "digest": _sha256(path)}
        for path in candidates
    ]
    errors = sum(_read_trace(path, calls) for path in candidates)
    return {
        "artifacts": artifacts,
        "tool_calls": dict(sorted(calls.items())),
        "tool_errors": errors,
    }


def _normalize_trial(
    path: Path | None, trial: dict[str, Any], repetition: int
) -> dict[str, Any]:
    agent_result = _object(trial.get("agent_result"))
    agent_info = _object(trial.get("agent_info"))
    model_info = _object(agent_info.get("model_info"))
    verifier = _object(trial.get("verifier_result"))
    rewards = _object(verifier.get("rewards"))
    exception = trial.get("exception_info")
    return {
        "task": _task_id(trial.get("task_name")),
        "task_digest": "sha256:"
        + str(trial.get("task_checksum", "")).removeprefix("sha256:"),
        "repetition": repetition,
        "trial_name": str(trial.get("trial_name", "")),
        "status": "COMPLETED" if exception is None else "ERROR",
        "exception_type": exception.get("exception_type")
        if isinstance(exception, dict)
        else None,
        "model": model_info.get("name"),
        "model_provider": model_info.get("provider"),
        "agent_name": agent_info.get("name"),
        "agent_version": agent_info.get("version"),
        "rewards": rewards,
        "false_certification": rewards.get("false_certification"),
        "tokens": {
            "input": agent_result.get("n_input_tokens"),
            "cache": agent_result.get("n_cache_tokens"),
            "output": agent_result.get("n_output_tokens"),
        },
        "cost_usd": agent_result.get("cost_usd"),
        "agent_seconds": _timing_seconds(trial.get("agent_execution")),
        "trace": _trace_summary(path),
        "raw_result_digest": _sha256(path) if path is not None else _json_digest(trial),
    }


def _observation_identity(
    dataset: str, heldout_manifest: dict[str, Any] | None
) -> tuple[dict[str, str], str, str]:
    if heldout_manifest is not None:
        return (
            {
                str(item["id"]): str(item["digest"])
                for item in heldout_manifest.get("tasks", [])
            },
            "held-out-comparative-evaluation",
            str(heldout_manifest.get("dataset", {}).get("id", dataset)),
        )
    suite = get_suite(dataset)
    known = {
        ref.path.name: "sha256:" + task_digest(ref.path).removeprefix("sha256:")
        for ref in suite.tasks
    }
    evidence_class = (
        "workflow-observation"
        if suite.claim_class == "workflow-observation"
        else suite.claim_class
    )
    return known, evidence_class, suite.id


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
    failures.extend(
        f"{task}: expected {attempts} repetitions, observed {counters[task]}"
        for task in sorted(expected_tasks)
        if counters[task] != attempts
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
    raw_trials = _trial_results(result_path, payload)
    raw_trials.sort(
        key=lambda pair: (
            _task_id(pair[1].get("task_name")),
            str(pair[1].get("trial_name", "")),
        )
    )
    counters: Counter[str] = Counter()
    trials: list[dict[str, Any]] = []
    for path, raw in raw_trials:
        task = _task_id(raw.get("task_name"))
        repetition = counters[task]
        if runtime_snapshot is not None and isinstance(
            runtime_snapshot.get("repetition"), int
        ):
            repetition = int(runtime_snapshot["repetition"])
        counters[task] += 1
        normalized = _normalize_trial(path, raw, repetition)
        if runtime_snapshot is not None and isinstance(
            runtime_snapshot.get("pair_id"), str
        ):
            normalized["pair_id"] = runtime_snapshot["pair_id"]
        trials.append(normalized)

    known_digests, evidence_class, dataset_id = _observation_identity(
        dataset, heldout_manifest
    )
    expected_tasks = _expected_tasks(job, set(known_digests))
    raw_attempts = job.get("n_attempts")
    attempts: int = (
        raw_attempts
        if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool)
        else 0
    )
    expected_digests = {
        task: known_digests[task] for task in expected_tasks if task in known_digests
    }
    failures = _observation_failures(
        counters=counters,
        expected_tasks=expected_tasks,
        attempts=attempts,
        expected_digests=expected_digests,
        trials=trials,
        payload=payload,
    )
    models = sorted(
        {str(trial["model"]) for trial in trials if trial["model"] is not None}
    )
    if len(models) != 1:
        failures.append(f"expected one recorded model identity, observed={models}")
    runtime = runtime_snapshot or {}
    if runtime.get("model") is not None and models != [runtime["model"]]:
        failures.append("recorded model differs from the frozen runtime snapshot")
    providers = sorted(
        {
            str(trial["model_provider"])
            for trial in trials
            if trial.get("model_provider") is not None
        }
    )
    if len(providers) > 1:
        failures.append(f"expected one model provider identity, observed={providers}")
    agent_names = sorted(
        {
            str(trial["agent_name"])
            for trial in trials
            if trial.get("agent_name") is not None
        }
    )
    if len(agent_names) != 1:
        failures.append(f"expected one agent name identity, observed={agent_names}")
    agent_versions = sorted(
        {
            str(trial["agent_version"])
            for trial in trials
            if trial.get("agent_version") is not None
        }
    )
    if len(agent_versions) != 1:
        failures.append(
            f"expected one agent version identity, observed={agent_versions}"
        )
    frozen_agent = runtime.get("agent") if isinstance(runtime, dict) else None
    if isinstance(frozen_agent, dict):
        if agent_names and agent_names != [frozen_agent.get("name")]:
            failures.append("observed agent name differs from the frozen runtime")
        if agent_versions and agent_versions != [frozen_agent.get("version")]:
            failures.append("observed agent version differs from the frozen runtime")
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
    evidence = {
        "schema_version": "1",
        "evidence_class": evidence_class,
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INCOMPLETE",
        "source_sha": _git_sha(),
        "dataset": dataset_id,
        "condition": condition,
        "job": {
            "path": _display_path(job_path),
            "digest": _sha256(job_path),
            "comparison_signature": _json_digest(_comparison_job(job)),
            "n_attempts": attempts,
        },
        "runtime_snapshot": runtime,
        "fixed_invariants": {
            "model": models[0] if len(models) == 1 else None,
            "model_provider": providers[0] if len(providers) == 1 else None,
            "agent_name": agent_names[0] if len(agent_names) == 1 else None,
            "agent_version": agent_versions[0] if len(agent_versions) == 1 else None,
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
        return None
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
    for metric in (
        "correctness",
        "evidence_validity",
        "scope_accuracy",
        "assurance_calibration",
        "false_certification",
    ):
        if metrics[metric]["pair_count"] != len(pairs):
            failures.append(
                f"required outcome metric is missing from a complete pair: {metric}"
            )
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
    providers = sorted(
        {
            str(item["model_provider"])
            for item in trials
            if item.get("model_provider") is not None
        }
    )
    if len(providers) > 1:
        failures.append(f"expected one model provider identity, observed={providers}")
    agent_names = sorted(
        {
            str(item["agent_name"])
            for item in trials
            if item.get("agent_name") is not None
        }
    )
    frozen_agent = experiment.get("agent") if isinstance(experiment, dict) else None
    frozen_agent_name = (
        frozen_agent.get("name") if isinstance(frozen_agent, dict) else None
    )
    frozen_agent_version = (
        frozen_agent.get("version") if isinstance(frozen_agent, dict) else None
    )
    if agent_names and agent_names != [frozen_agent_name]:
        failures.append("observed agent name does not match the frozen experiment")
    agent_versions = sorted(
        {
            str(item["agent_version"])
            for item in trials
            if item.get("agent_version") is not None
        }
    )
    if agent_versions and agent_versions != [frozen_agent_version]:
        failures.append("observed agent version does not match the frozen experiment")
    task_digests = {str(item["id"]): str(item["digest"]) for item in manifest["tasks"]}
    evidence = {
        "schema_version": "1",
        "evidence_class": "held-out-comparative-evaluation",
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INCOMPLETE",
        "source_sha": _git_sha(),
        "dataset": manifest["dataset"]["id"],
        "condition": condition,
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
            "model_provider": providers[0] if len(providers) == 1 else None,
            "agent_name": agent_names[0] if len(agent_names) == 1 else None,
            "agent_version": agent_versions[0] if len(agent_versions) == 1 else None,
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
