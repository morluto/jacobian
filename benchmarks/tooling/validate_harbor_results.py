#!/usr/bin/env python3
"""Fail-closed validation and evidence capture for a Harbor Oracle result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_AUGMENTED_DIGEST_MANIFEST = "jacobian-augmented-task-digests.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tooling.command_runner import git_head_sha  # noqa: E402
from benchmarks.tooling.errors import HarborSuiteError  # noqa: E402
from benchmarks.tooling.harbor_digest import durable_task_digest  # noqa: E402
from benchmarks.tooling.harbor_suite import (  # noqa: E402
    ROOT,
    get_suite,
    task_digest,
)


def _git_sha() -> str:
    value = git_head_sha(ROOT)
    if value is None:
        raise HarborSuiteError("unable to resolve git HEAD")
    return value


def _task_id(name: Any) -> str:
    return name.rsplit("/", 1)[-1] if isinstance(name, str) else ""


def _prefixed_digest(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _augmented_job_name(*, dataset: str, digests: dict[str, str]) -> str:
    identity = json.dumps(
        {"dataset": dataset, "tasks": sorted(digests.items())},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"jacobian-oracle-{hashlib.sha256(identity).hexdigest()}"


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_execution_summary(
    payload: dict[str, Any], *, trial_count: int
) -> list[str]:
    failures: list[str] = []
    for key in ("id", "started_at", "finished_at", "n_total_trials", "stats"):
        if key not in payload:
            failures.append(f"result.json: missing {key}")

    total = payload.get("n_total_trials")
    if not _is_nonnegative_integer(total) or total == 0:
        failures.append("result.json: n_total_trials must be positive")
    elif trial_count != total:
        failures.append(
            "result.json: per-trial result count disagrees with n_total_trials"
        )

    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return [*failures, "result.json: stats must be an object"]

    count_keys = (
        "n_completed_trials",
        "n_errored_trials",
        "n_running_trials",
        "n_pending_trials",
        "n_cancelled_trials",
    )
    for key in count_keys:
        if not _is_nonnegative_integer(stats.get(key, 0)):
            failures.append(f"result.json: stats.{key} must be non-negative")
    incomplete_keys = count_keys[1:]
    if any(stats.get(key, 0) for key in incomplete_keys):
        failures.append("result.json: execution is incomplete or contains errors")
    if stats.get("n_completed_trials", 0) != trial_count:
        failures.append(
            "result.json: completed-trial count disagrees with per-trial results"
        )
    return failures


def _validate_reward(
    rewards: dict[str, Any], *, dimension: str, trial_index: int
) -> list[str]:
    value = rewards.get(dimension)
    if dimension == "false_certification" and isinstance(value, bool):
        if value is False:
            return []
        return [f"trial result {trial_index}: {dimension} must be zero"]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return [
            f"trial result {trial_index}: {dimension} reward is missing or not finite"
        ]
    expected = 0.0 if dimension == "false_certification" else 1.0
    if math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        return []
    requirement = "zero" if dimension == "false_certification" else "full reward"
    return [f"trial result {trial_index}: {dimension} must be {requirement}"]


def _validate_trial(
    trial: Any,
    *,
    index: int,
    expected_tasks: set[str],
    expected_digests: dict[str, str],
    trial_digest: Any,
) -> tuple[str | None, list[str]]:
    if not isinstance(trial, dict):
        return None, [f"trial result {index} must be an object"]
    task_id = _task_id(trial.get("task_name"))
    if not task_id:
        return None, [f"trial result {index}: missing task_name"]

    failures: list[str] = []
    if task_id not in expected_tasks:
        failures.append(f"trial result {index}: unexpected task {task_id}")
    expected = expected_digests.get(task_id)
    if not isinstance(trial_digest, str) or _DIGEST.fullmatch(trial_digest) is None:
        failures.append(f"trial result {index}: missing durable task digest")
    elif expected and trial_digest != (
        expected if expected.startswith("sha256:") else f"sha256:{expected}"
    ):
        failures.append(f"trial result {index}: task digest mismatch for {task_id}")
    if trial.get("exception_info") is not None:
        failures.append(f"trial result {index}: exception result is not certifying")

    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if not isinstance(rewards, dict) or not rewards:
        return task_id, [*failures, f"trial result {index}: incomplete verifier reward"]
    dimensions = (
        "reward",
        *sorted(
            dimension
            for dimension, value in rewards.items()
            if dimension != "reward"
            and (
                dimension == "false_certification"
                or (isinstance(value, (int, float)) and not isinstance(value, bool))
            )
        ),
    )
    for dimension in dimensions:
        failures.extend(
            _validate_reward(rewards, dimension=dimension, trial_index=index)
        )
    return task_id, failures


def _validate_payload(
    payload: Any,
    *,
    trial_results: list[Any],
    trial_digests: list[Any],
    expected_tasks: set[str],
    expected_digests: dict[str, str],
) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return ["result.json must contain an object"]
    if not trial_results:
        failures.append("result.json: no per-trial result files were found")
    if len(trial_digests) != len(trial_results):
        failures.append("result.json: per-trial lock count disagrees with results")
    failures.extend(
        _validate_execution_summary(payload, trial_count=len(trial_results))
    )

    observed_task_counts: dict[str, int] = {}
    for index, trial in enumerate(trial_results):
        task_id, trial_failures = _validate_trial(
            trial,
            index=index,
            expected_tasks=expected_tasks,
            expected_digests=expected_digests,
            trial_digest=trial_digests[index] if index < len(trial_digests) else None,
        )
        failures.extend(trial_failures)
        if task_id is None:
            continue
        observed_task_counts[task_id] = observed_task_counts.get(task_id, 0) + 1
    observed_tasks = set(observed_task_counts)
    if observed_tasks != expected_tasks:
        failures.append(
            "result.json: task coverage differs from requested tasks: "
            f"expected={sorted(expected_tasks)}, observed={sorted(observed_tasks)}"
        )
    duplicates = sorted(
        task_id for task_id, count in observed_task_counts.items() if count != 1
    )
    if duplicates:
        failures.append(
            "result.json: expected exactly one trial for each task; duplicates="
            f"{duplicates}"
        )
    return failures


def _find_result(jobs_dir: Path) -> Path:
    candidates = sorted(
        (path for path in jobs_dir.glob("*/result.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise HarborSuiteError(f"no Harbor result.json found below {jobs_dir}")
    return candidates[0]


def _load_trial_results(result_path: Path) -> tuple[list[Any], list[Path], list[Any]]:
    paths = sorted(
        path for path in result_path.parent.glob("*/result.json") if path.is_file()
    )
    results: list[Any] = []
    digests: list[Any] = []
    for path in paths:
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
            lock = json.loads((path.parent / "lock.json").read_text(encoding="utf-8"))
            task = lock.get("task") if isinstance(lock, dict) else None
            digests.append(task.get("digest") if isinstance(task, dict) else None)
        except (OSError, json.JSONDecodeError) as exc:
            raise HarborSuiteError(
                f"unable to read Harbor trial result {path}: {exc}"
            ) from exc
    return results, paths, digests


def _prepare_augmented_digest_manifest(
    *, dataset: str, tasks: tuple[str, ...] | None, jobs_dir: Path
) -> Path:
    """Record the augmented task identity before Harbor starts an Oracle run."""

    suite = get_suite(dataset)
    known = {ref.path.name: ref for ref in suite.tasks}
    requested = set(tasks) if tasks else set(known)
    unknown = sorted(requested - set(known))
    if unknown:
        raise HarborSuiteError(f"unknown task(s) for {dataset}: {', '.join(unknown)}")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    manifest = jobs_dir / _AUGMENTED_DIGEST_MANIFEST
    augmented_digests = {
        task_id: _prefixed_digest(task_digest(known[task_id].path))
        for task_id in sorted(requested)
    }
    manifest.write_text(
        json.dumps(
            {
                "dataset": suite.id,
                "job_name": _augmented_job_name(
                    dataset=suite.id, digests=augmented_digests
                ),
                "prepared_at_ns": time.time_ns(),
                "tasks": [
                    {
                        "task": task_id,
                        "digest": augmented_digests[task_id],
                    }
                    for task_id in sorted(requested)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_augmented_digest_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"augmented task digest manifest is unavailable: {exc}"]
    if not isinstance(manifest, dict):
        return None, ["augmented task digest manifest must contain an object"]
    return manifest, []


def _validate_augmented_manifest_metadata(
    *,
    manifest: dict[str, Any],
    result_path: Path,
    dataset: str,
    expected_job_name: str,
) -> list[str]:
    if manifest.get("dataset") != dataset:
        return ["augmented task digest manifest has the wrong dataset"]
    if manifest.get("job_name") != expected_job_name:
        return ["augmented task digest manifest has the wrong job identity"]
    if result_path.parent.name != expected_job_name:
        return ["Harbor result is not bound to its augmented task identity"]
    prepared_at_ns = manifest.get("prepared_at_ns")
    if not isinstance(prepared_at_ns, int) or isinstance(prepared_at_ns, bool):
        return ["augmented task digest manifest has no preparation timestamp"]
    try:
        result_mtime_ns = result_path.stat().st_mtime_ns
    except OSError as exc:
        return [f"unable to stat Harbor result: {exc}"]
    if result_mtime_ns < prepared_at_ns:
        return ["Harbor result predates its augmented task digest manifest"]
    return []


def _validate_augmented_task_entries(
    entries: Any, expected_digests: dict[str, str]
) -> list[str]:
    if not isinstance(entries, list):
        return ["augmented task digest manifest has no task entries"]
    observed = {
        entry.get("task"): entry.get("digest")
        for entry in entries
        if isinstance(entry, dict)
    }
    if set(observed) != set(expected_digests):
        return ["augmented task digest manifest task coverage does not match"]
    failures: list[str] = []
    for task_id, expected in expected_digests.items():
        digest = observed.get(task_id)
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            failures.append(f"augmented task digest is invalid for {task_id}")
        elif digest != expected:
            failures.append(f"augmented task digest mismatch for {task_id}")
    return failures


def _validate_augmented_digest_manifest(
    *,
    manifest_path: Path,
    result_path: Path,
    dataset: str,
    expected_digests: dict[str, str],
) -> list[str]:
    """Require the result to be newer than and bound to its pre-run identity."""

    manifest, failures = _load_augmented_digest_manifest(manifest_path)
    if manifest is None:
        return failures
    failures = _validate_augmented_manifest_metadata(
        manifest=manifest,
        result_path=result_path,
        dataset=dataset,
        expected_job_name=_augmented_job_name(
            dataset=dataset, digests=expected_digests
        ),
    )
    if failures:
        return failures
    return _validate_augmented_task_entries(manifest.get("tasks"), expected_digests)


def validate(
    *,
    dataset: str,
    tasks: tuple[str, ...] | None,
    jobs_dir: Path,
    result_path: Path | None = None,
) -> Path:
    jobs_dir = jobs_dir.resolve()
    suite = get_suite(dataset)
    known = {ref.path.name: ref for ref in suite.tasks}
    requested = set(tasks) if tasks else set(known)
    unknown = sorted(requested - set(known))
    if unknown:
        raise HarborSuiteError(f"unknown task(s) for {dataset}: {', '.join(unknown)}")
    result_path = (result_path or _find_result(jobs_dir)).resolve()
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(
            f"unable to read Harbor result {result_path}: {exc}"
        ) from exc
    expected_digests = {
        task_id: durable_task_digest(ref.path)
        for task_id, ref in known.items()
        if task_id in requested
    }
    augmented_digests = {
        task_id: _prefixed_digest(task_digest(ref.path))
        for task_id, ref in known.items()
        if task_id in requested
    }
    trial_results, trial_paths, trial_digests = _load_trial_results(result_path)
    failures = _validate_augmented_digest_manifest(
        manifest_path=jobs_dir / _AUGMENTED_DIGEST_MANIFEST,
        result_path=result_path,
        dataset=suite.id,
        expected_digests=augmented_digests,
    )
    failures.extend(
        _validate_payload(
            payload,
            trial_results=trial_results,
            trial_digests=trial_digests,
            expected_tasks=requested,
            expected_digests=expected_digests,
        )
    )
    if failures:
        raise HarborSuiteError("\n".join(failures))
    evidence = result_path.parent / "oracle-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "source_sha": _git_sha(),
                "harbor_version": importlib.metadata.version("harbor"),
                "dataset": suite.id,
                "tasks": [
                    {
                        "task": task_id,
                        "digest": augmented_digests[task_id],
                        "verifier": (known[task_id].path / "tests" / "verifier.py")
                        .relative_to(ROOT)
                        .as_posix(),
                        "verifier_sha256": hashlib.sha256(
                            (known[task_id].path / "tests" / "verifier.py").read_bytes()
                        ).hexdigest(),
                    }
                    for task_id in sorted(requested)
                ],
                "result": result_path.relative_to(ROOT).as_posix(),
                "trial_results": [
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in trial_paths
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    try:
        task_selection = tuple(args.tasks) if args.tasks else None
        if args.prepare:
            evidence = _prepare_augmented_digest_manifest(
                dataset=args.dataset,
                tasks=task_selection,
                jobs_dir=args.jobs_dir,
            )
            manifest = json.loads(evidence.read_text(encoding="utf-8"))
            evidence = manifest["job_name"]
        else:
            evidence = validate(
                dataset=args.dataset,
                tasks=task_selection,
                jobs_dir=args.jobs_dir,
                result_path=args.result,
            )
    except HarborSuiteError as exc:
        parser.error(str(exc))
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
