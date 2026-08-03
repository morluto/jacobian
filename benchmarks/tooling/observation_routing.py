"""Build fail-closed, value-aware MCP routing observations for Harbor runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.tooling.harbor_suite import (
    ROOT,
    HarborSuiteError,
    get_suite,
    task_digest,
)
from benchmarks.tooling.observation_selection import normalize_selection
from jacobian.eval.telemetry import parse_agent_transcript

MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
JACOBIAN_SERVER = {
    "name": "jacobian",
    "transport": "streamable-http",
    "url": "http://127.0.0.1:8000/mcp",
}
ROUTING_STATUSES = (
    "NOT_CONFIGURED",
    "HARNESS_UNAVAILABLE",
    "EVIDENCE_INCOMPLETE",
    "AVAILABLE_NO_CALL",
    "DISCOVERY_FAILED",
    "DISCOVERY_MISS",
    "DESCRIBED_NOT_INVOKED",
    "INVOKE_FAILED",
    "USED_OTHER_CAPABILITY",
    "USED",
)
OPPORTUNITY_VALUES = ("NONE", "OPTIONAL", "HIGH", "UNASSESSED")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _task_id(value: Any) -> str:
    return value.rsplit("/", 1)[-1] if isinstance(value, str) else ""


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


def _trial_status(trial: dict[str, Any]) -> str:
    status = trial.get("status")
    if status in {"TIMEOUT", "CANCELLED", "ERROR", "FAILED"}:
        return str(status)
    return "ERROR" if trial.get("exception_info") is not None else "COMPLETED"


def _config_jobs_dir(config: dict[str, Any]) -> Path:
    value = config.get("jobs_dir")
    if not isinstance(value, str) or not value:
        raise HarborSuiteError("resolved Harbor config must bind jobs_dir")
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _routing_config_state(config: dict[str, Any]) -> tuple[list[Any], bool, list[str]]:
    agents = config.get("agents")
    if not isinstance(agents, list) or len(agents) != 1:
        return [], False, ["resolved config must contain exactly one agent"]
    agent = agents[0]
    if not isinstance(agent, dict) or agent.get("name") != "codex":
        return [], False, ["resolved config must select the Codex agent"]
    servers = agent.get("mcp_servers", [])
    if not isinstance(servers, list):
        return [], False, ["resolved agent mcp_servers must be an array"]
    environment = _object(config.get("environment"))
    compose = environment.get("extra_docker_compose", [])
    if not isinstance(compose, list):
        return [], False, ["resolved extra_docker_compose must be an array"]
    has_sidecar = any(
        Path(str(value)).name == "jacobian-observation.compose.yaml"
        for value in compose
    )
    return servers, has_sidecar, []


def resolved_config_failures(config: dict[str, Any], *, condition: str) -> list[str]:
    """Validate the fully resolved Harbor condition before model execution."""

    if condition not in {"control", "treatment"}:
        return [f"unknown observation condition: {condition}"]
    servers, has_sidecar, failures = _routing_config_state(config)
    if failures:
        return failures
    if condition == "treatment":
        if servers != [JACOBIAN_SERVER]:
            failures.append("treatment does not contain the exact Jacobian MCP server")
        if not has_sidecar:
            failures.append("treatment does not contain the Jacobian sidecar compose")
    else:
        if servers:
            failures.append("control unexpectedly contains MCP servers")
        if has_sidecar:
            failures.append("control unexpectedly contains the Jacobian sidecar")
    return failures


def _opportunity(task_ref: Any) -> dict[str, Any]:
    value = task_ref.tool_opportunity
    if value is None:
        return {"value": "UNASSESSED", "relevant_capability_ids": [], "rationale": None}
    return {
        "value": value.value,
        "relevant_capability_ids": list(value.relevant_capability_ids),
        "rationale": value.rationale,
    }


def _empty_telemetry() -> dict[str, Any]:
    return {
        "transcript": None,
        "turn_usage_present": False,
        "mcp_calls": [],
        "successful_mcp_calls": [],
        "tool_error_count": 0,
        "capability_attempt_ids": [],
        "capability_ids": [],
        "capability_descriptions": [],
    }


def _routing_telemetry(
    trial_path: Path | None, *, jobs_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    telemetry = _empty_telemetry()
    if trial_path is None:
        return telemetry, ["raw agent transcript is unavailable for inline result"]
    transcript = trial_path.parent / "agent" / "codex.txt"
    try:
        if transcript.is_symlink() or not transcript.is_file():
            return telemetry, ["raw agent transcript is missing or symlinked"]
        if transcript.stat().st_size > MAX_TRANSCRIPT_BYTES:
            return telemetry, ["raw agent transcript exceeds 64 MiB"]
        relative = transcript.resolve().relative_to(jobs_dir.resolve()).as_posix()
        parsed = parse_agent_transcript(transcript)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        return telemetry, [f"raw agent transcript is invalid: {exc}"]
    descriptions = parsed.get("capability_descriptions", [])
    if not isinstance(descriptions, list) or not all(
        isinstance(item, dict)
        and set(item)
        == {"kind", "query", "domain", "mode", "capability_id", "match_ids"}
        and isinstance(item.get("match_ids"), list)
        and all(isinstance(value, str) for value in item["match_ids"])
        for item in descriptions
    ):
        return telemetry, ["capability description telemetry is malformed"]

    def strings(key: str) -> list[str]:
        value = parsed.get(key, [])
        return (
            [item for item in value if isinstance(item, str)]
            if isinstance(value, list)
            else []
        )

    raw_errors = parsed.get("tool_error_count", 0)
    telemetry.update(
        transcript={"path": relative, "digest": _sha256(transcript)},
        turn_usage_present=isinstance(parsed.get("usage"), dict),
        mcp_calls=strings("mcp_calls"),
        successful_mcp_calls=strings("successful_tool_calls"),
        tool_error_count=(
            raw_errors
            if isinstance(raw_errors, int) and not isinstance(raw_errors, bool)
            else 0
        ),
        capability_attempt_ids=strings("capability_attempt_ids"),
        capability_ids=strings("capability_ids"),
        capability_descriptions=descriptions,
    )
    return telemetry, []


def _description_ids(descriptions: object) -> set[str]:
    observed: set[str] = set()
    if not isinstance(descriptions, list):
        return observed
    for description in descriptions:
        if not isinstance(description, dict):
            continue
        if isinstance(description.get("capability_id"), str):
            observed.add(description["capability_id"])
        matches = description.get("match_ids")
        if isinstance(matches, list):
            observed.update(item for item in matches if isinstance(item, str))
    return observed


def _matches_opportunity(observed: set[str], relevant: set[str]) -> bool:
    # NONE and UNASSESSED have no relevance set.  A call cannot become a
    # relevant use merely because the evaluator lacks a classification.
    return bool(relevant and observed & relevant)


def _discovery_status(telemetry: dict[str, Any], relevant: set[str]) -> str:
    if _matches_opportunity(
        _description_ids(telemetry["capability_descriptions"]), relevant
    ):
        return "DESCRIBED_NOT_INVOKED"
    if "capability.describe" not in telemetry["mcp_calls"]:
        return "AVAILABLE_NO_CALL"
    if telemetry["tool_error_count"] and not telemetry["capability_descriptions"]:
        return "DISCOVERY_FAILED"
    return "DISCOVERY_MISS"


def _routing_status(
    *,
    condition: str,
    config_failures: list[str],
    telemetry: dict[str, Any],
    opportunity: dict[str, Any],
) -> str:
    if condition == "control":
        return "NOT_CONFIGURED"
    if config_failures:
        return "HARNESS_UNAVAILABLE"
    if telemetry["transcript"] is None or not telemetry["turn_usage_present"]:
        return "EVIDENCE_INCOMPLETE"
    relevant = set(opportunity["relevant_capability_ids"])
    completed = set(telemetry["capability_ids"])
    attempted = set(telemetry["capability_attempt_ids"])
    if _matches_opportunity(completed, relevant):
        return "USED"
    if completed:
        return "USED_OTHER_CAPABILITY"
    if (
        _matches_opportunity(attempted, relevant)
        or "capability.invoke" in telemetry["mcp_calls"]
    ):
        return "INVOKE_FAILED"
    return _discovery_status(telemetry, relevant)


def _routing_rewards(raw: dict[str, Any]) -> dict[str, Any]:
    rewards = _object(_object(raw.get("verifier_result")).get("rewards"))
    return {
        name: rewards.get(name)
        for name in (
            "correctness",
            "evidence_validity",
            "scope_accuracy",
            "assurance_calibration",
            "reward",
            "false_certification",
        )
    }


def _routing_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(trial["routing_status"]) for trial in trials)
    by_opportunity = []
    for value in OPPORTUNITY_VALUES:
        selected = [trial for trial in trials if trial["opportunity"]["value"] == value]
        relevant_used = sum(trial["routing_status"] == "USED" for trial in selected)
        by_opportunity.append(
            {
                "value": value,
                "trials": len(selected),
                "relevant_used": relevant_used,
                "adoption_rate": (
                    relevant_used / len(selected)
                    if selected and value in {"OPTIONAL", "HIGH"}
                    else None
                ),
            }
        )
    return {
        "trial_count": len(trials),
        "routing_status_counts": [
            {"status": status, "count": count}
            for status, count in sorted(statuses.items())
        ],
        "by_tool_opportunity": by_opportunity,
    }


def _build_trials(
    raw_trials: list[tuple[Path | None, dict[str, Any]]],
    *,
    refs: dict[str, Any],
    jobs_dir: Path,
    condition: str,
    config_failures: list[str],
) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    counters: Counter[str] = Counter()
    trials: list[dict[str, Any]] = []
    telemetry_failures: list[str] = []
    for trial_path, raw in raw_trials:
        task = _task_id(raw.get("task_name"))
        repetition = counters[task]
        counters[task] += 1
        telemetry, trace_failures = _routing_telemetry(trial_path, jobs_dir=jobs_dir)
        telemetry_failures.extend(
            f"{task} repetition {repetition}: {failure}" for failure in trace_failures
        )
        opportunity = (
            _opportunity(refs[task])
            if task in refs
            else {
                "value": "UNASSESSED",
                "relevant_capability_ids": [],
                "rationale": None,
            }
        )
        checksum = str(raw.get("task_checksum", ""))
        task_checksum = "sha256:" + checksum.removeprefix("sha256:")
        trials.append(
            {
                "task": task,
                "task_digest": task_checksum,
                "repetition": repetition,
                "opportunity": opportunity,
                "routing_status": _routing_status(
                    condition=condition,
                    config_failures=config_failures,
                    telemetry=telemetry,
                    opportunity=opportunity,
                ),
                "rewards": _routing_rewards(raw),
                "telemetry": telemetry,
                "raw_result_digest": _sha256(trial_path)
                if trial_path is not None
                else _json_digest(raw),
            }
        )
    return trials, counters, telemetry_failures


def _routing_failures(
    *,
    expected_tasks: list[str],
    attempts: int,
    counters: Counter[str],
    known: dict[str, str],
    trials: list[dict[str, Any]],
    raw_trials: list[tuple[Path | None, dict[str, Any]]],
    selection_failures: list[str],
    config_failures: list[str],
    telemetry_failures: list[str],
    condition: str,
) -> list[str]:
    failures = [*config_failures, *selection_failures]
    if set(counters) != set(expected_tasks):
        failures.append(
            f"task coverage mismatch: expected={sorted(expected_tasks)}, observed={sorted(counters)}"
        )
    if attempts <= 0:
        failures.append("resolved config n_attempts must be a positive integer")
    for task in expected_tasks:
        if attempts > 0 and counters[task] != attempts:
            failures.append(
                f"{task}: expected {attempts} repetitions, observed {counters[task]}"
            )
    for trial in trials:
        if (
            known.get(trial["task"]) is not None
            and trial["task_digest"] != known[trial["task"]]
        ):
            failures.append(
                f"{trial['task']} repetition {trial['repetition']}: task digest mismatch"
            )
    if any(_trial_status(raw) != "COMPLETED" for _path, raw in raw_trials):
        failures.append("execution is incomplete or contains errors")
    if condition == "treatment":
        failures.extend(telemetry_failures)
        failures.extend(
            f"{trial['task']} repetition {trial['repetition']}: routing evidence is incomplete"
            for trial in trials
            if trial["routing_status"] == "EVIDENCE_INCOMPLETE"
        )
    return failures


def build_routing_observation(
    *,
    dataset: str,
    condition: str,
    resolved_config_path: Path,
    result_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build routing evidence using the jobs directory bound by the resolved config."""

    config = _read_json(resolved_config_path)
    if not isinstance(config, dict):
        raise HarborSuiteError("resolved Harbor config must be an object")
    jobs_dir = _config_jobs_dir(config)
    config_failures = resolved_config_failures(config, condition=condition)
    result_path = (result_path or _find_result(jobs_dir)).resolve()
    try:
        result_path.relative_to(jobs_dir.resolve())
    except ValueError as exc:
        raise HarborSuiteError(
            "result path must be inside resolved config jobs_dir"
        ) from exc
    payload = _read_json(result_path)
    if not isinstance(payload, dict):
        raise HarborSuiteError("Harbor result must be an object")

    suite = get_suite(dataset)
    refs = {ref.path.name: ref for ref in suite.tasks}
    known = {
        name: "sha256:" + task_digest(ref.path).removeprefix("sha256:")
        for name, ref in refs.items()
    }
    expected_tasks, selection_mode, _eval_args, selection_failures = (
        normalize_selection(
            config,
            known=known,
            task_dirs={name: ref.path for name, ref in refs.items()},
            dataset_path=suite.path,
            root=ROOT,
        )
    )
    raw_attempts = config.get("n_attempts")
    attempts = (
        raw_attempts
        if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool)
        else 0
    )
    raw_trials = _trial_results(result_path, payload)
    raw_trials.sort(
        key=lambda pair: (
            _task_id(pair[1].get("task_name")),
            str(pair[1].get("trial_name", "")),
        )
    )
    trials, counters, telemetry_failures = _build_trials(
        raw_trials,
        refs=refs,
        jobs_dir=jobs_dir,
        condition=condition,
        config_failures=config_failures,
    )
    failures = _routing_failures(
        expected_tasks=expected_tasks,
        attempts=attempts,
        counters=counters,
        known=known,
        trials=trials,
        raw_trials=raw_trials,
        selection_failures=selection_failures,
        config_failures=config_failures,
        telemetry_failures=telemetry_failures,
        condition=condition,
    )

    servers, has_sidecar, _ = _routing_config_state(config)
    report = {
        "schema_version": "2",
        "evidence_class": "workflow-routing-observation",
        "causal_claim_authorized": False,
        "status": "VALID" if not failures else "INCOMPLETE",
        "source_sha": _git_sha(),
        "dataset": suite.dataset_name,
        "condition": condition,
        "resolved_config": {
            "path": _display_path(resolved_config_path),
            "digest": _sha256(resolved_config_path),
            "selection_mode": selection_mode
            if selection_mode in {"dataset-task-names", "explicit-tasks"}
            else "invalid",
            "selection": expected_tasks,
            "n_attempts": attempts,
            "jacobian_mcp_configured": servers == [JACOBIAN_SERVER],
            "jacobian_sidecar_configured": has_sidecar,
        },
        "summary": _routing_summary(trials),
        "trials": trials,
        "validation_failures": failures,
    }
    return report, failures


__all__ = ["build_routing_observation", "resolved_config_failures"]
