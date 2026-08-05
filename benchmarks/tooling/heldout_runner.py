"""Execute a frozen held-out plan with resumable pair-boundary accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.command_runner import operator_environment, run_operator_command
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import BENCHMARKS
from benchmarks.tooling.strict_boundaries import HeldoutRunPlan, raise_strict_model

CommandRunner = Callable[[list[str]], int]

_HARBOR_RUNNER_VARIABLES = (
    "ALL_PROXY",
    "CODEX_FORCE_AUTH_JSON",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"invalid JSON {path}: {exc}") from exc


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise HarborSuiteError("held-out manifest cannot be read for binding") from exc
    return "sha256:" + hashlib.sha256(contents).hexdigest()


def _plan_digest(plan: dict[str, Any]) -> str:
    unsigned = dict(plan)
    unsigned.pop("plan_digest", None)
    return _json_digest(unsigned)


def _result_file(jobs_dir: Path) -> Path:
    candidates = sorted(
        (path for path in jobs_dir.glob("*/result.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (path for path in jobs_dir.rglob("result.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    if not candidates:
        raise HarborSuiteError(f"no Harbor result.json found below {jobs_dir}")
    return candidates[0]


def _trial_payloads(result_path: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    nested = sorted(result_path.parent.glob("*/result.json"))
    if nested:
        values = [_read_json(path) for path in nested]
    else:
        values = result.get("trial_results", [])
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, dict) for value in values)
    ):
        raise HarborSuiteError("Harbor result has no complete per-trial accounting")
    return values


def _usage(result_path: Path) -> tuple[int, float]:
    result = _read_json(result_path)
    if not isinstance(result, dict):
        raise HarborSuiteError("Harbor result must be an object")
    stats = result.get("stats")
    if isinstance(stats, dict) and any(
        stats.get(key, 0)
        for key in (
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
        )
    ):
        raise HarborSuiteError("Harbor result is incomplete")
    tokens = 0
    cost = 0.0
    for trial in _trial_payloads(result_path, result):
        if trial.get("exception_info") is not None:
            raise HarborSuiteError("Harbor trial contains an exception")
        agent_result = trial.get("agent_result")
        if not isinstance(agent_result, dict):
            raise HarborSuiteError("Harbor trial is missing agent accounting")
        input_tokens = agent_result.get("n_input_tokens")
        output_tokens = agent_result.get("n_output_tokens")
        trial_cost = agent_result.get("cost_usd")
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
            or not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
            or not isinstance(trial_cost, (int, float))
            or isinstance(trial_cost, bool)
            or not math.isfinite(float(trial_cost))
            or float(trial_cost) < 0
        ):
            raise HarborSuiteError(
                "Harbor trial has missing or invalid token/cost accounting"
            )
        tokens += input_tokens + output_tokens
        cost += float(trial_cost)
    return tokens, cost


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    schema = _read_json(BENCHMARKS / "schemas" / "held-out-ledger.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(ledger))
    if errors:
        raise HarborSuiteError(f"held-out ledger is invalid: {errors[0].message}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _default_command(command: list[str]) -> int:
    if not command:
        return 1
    result = run_operator_command(
        command[0],
        command[1:],
        cwd=Path.cwd(),
        timeout_seconds=3600.0,
        environment=operator_environment(include=_HARBOR_RUNNER_VARIABLES),
    )
    if result.exit_code is not None:
        return result.exit_code
    return 1


def _validated_plan(plan_path: Path) -> tuple[dict[str, Any], str, list[str]]:
    plan = _read_json(plan_path)
    raise_strict_model(HeldoutRunPlan, plan, label=str(plan_path))
    expected_digest = _plan_digest(plan)
    if plan.get("plan_digest") != expected_digest:
        raise HarborSuiteError("held-out run plan digest mismatch")
    runs = plan["runs"]
    pair_ids = list(dict.fromkeys(str(run["pair_id"]) for run in runs))
    if len(pair_ids) != plan["pair_count"]:
        raise HarborSuiteError("held-out run plan pair count mismatch")
    run_keys = [(run["pair_id"], run["condition"]) for run in runs]
    bad_pairs = any(
        {run["condition"] for run in runs if run["pair_id"] == pair_id} != {"C1", "C2"}
        for pair_id in pair_ids
    )
    if len(set(run_keys)) != len(run_keys) or bad_pairs:
        raise HarborSuiteError(
            "held-out run plan does not contain exactly one C1/C2 pair"
        )
    return plan, expected_digest, pair_ids


def _initial_ledger(
    ledger_path: Path,
    plan: dict[str, Any],
    plan_digest: str,
    manifest_digest: str,
) -> dict[str, Any]:
    if not ledger_path.exists():
        return {
            "schema_version": "2",
            "manifest_digest": manifest_digest,
            "plan_digest": plan_digest,
            "status": "RUNNING",
            "budget": plan["budget"],
            "usage": {"tokens": 0, "cost_usd": 0.0},
            "runs": {},
            "completed_pairs": [],
            "validation_failures": [],
        }
    ledger = _read_json(ledger_path)
    if not isinstance(ledger, dict):
        raise HarborSuiteError("existing ledger must be a JSON object")
    if ledger.get("plan_digest") != plan_digest:
        raise HarborSuiteError("existing ledger belongs to a different run plan")
    if ledger.get("manifest_digest") != manifest_digest:
        raise HarborSuiteError("existing ledger belongs to a different manifest")
    if ledger.get("status") != "COMPLETE":
        ledger["status"] = "RUNNING"
        ledger["validation_failures"] = []
    return ledger


def _incomplete(
    ledger_path: Path,
    ledger: dict[str, Any],
    run_id: str,
    message: str,
    run_status: str,
) -> bool:
    ledger["runs"][run_id] = {"status": run_status}
    ledger["status"] = "INCOMPLETE"
    ledger["validation_failures"].append(message)
    _write_ledger(ledger_path, ledger)
    return False


def _execute_run(
    *,
    run: dict[str, Any],
    root: Path,
    harbor_version: str,
    ledger_path: Path,
    ledger: dict[str, Any],
    command_runner: CommandRunner,
) -> bool:
    run_id = f"{run['pair_id']}/{run['condition']}"
    existing = ledger["runs"].get(run_id)
    if isinstance(existing, dict) and existing.get("status") == "COMPLETE":
        return True
    command = [
        "uvx",
        "--from",
        f"harbor=={harbor_version}",
        "harbor",
        "run",
        "-c",
        str(root / run["job"]),
    ]
    if command_runner(command) != 0:
        return _incomplete(
            ledger_path, ledger, run_id, f"Harbor run failed: {run_id}", "ERROR"
        )
    try:
        result_path = _result_file(root / run["jobs_dir"])
        tokens, cost = _usage(result_path)
    except HarborSuiteError as exc:
        return _incomplete(
            ledger_path, ledger, run_id, f"{run_id}: {exc}", "INCOMPLETE"
        )
    ledger["runs"][run_id] = {
        "status": "COMPLETE",
        "result_digest": _file_digest(result_path),
        "tokens": tokens,
        "cost_usd": cost,
    }
    ledger["usage"]["tokens"] += tokens
    ledger["usage"]["cost_usd"] += cost
    _write_ledger(ledger_path, ledger)
    return True


def _budget_exceeded(ledger: dict[str, Any]) -> bool:
    return bool(
        ledger["usage"]["tokens"] > ledger["budget"]["max_tokens"]
        or ledger["usage"]["cost_usd"] > ledger["budget"]["max_cost_usd"]
    )


def _write_routing_contract(
    directory: Path, condition: str, contract: dict[str, Any]
) -> None:
    """Persist a digest-bound routing status contract next to the ledger."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"routing-status-{condition.lower()}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_harbor_version(manifest_path: Path) -> str:
    """Resolve harbor_version from the canonical manifest."""

    from benchmarks.tooling.heldout_bundle import validate_manifest

    manifest = validate_manifest(manifest_path)
    return str(manifest["experiment"]["harbor_version"])


def _setup_routing_contracts(
    plan: dict[str, Any],
    manifest_path: Path,
    contract_dir: Path,
    probe_url: str,
    probe_fn: Any | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Run preflight and write routing contracts; return (treatment, control, failures)."""

    from benchmarks.tooling.heldout_bundle import (
        control_routing_status,
        treatment_readiness_preflight,
    )

    has_treatment = any(run["condition"] == "C2" for run in plan["runs"])
    has_control = any(run["condition"] == "C1" for run in plan["runs"])
    treatment_contract: dict[str, Any] | None = None
    control_contract: dict[str, Any] | None = None
    failures: list[str] = []
    if has_treatment:
        treatment_contract = treatment_readiness_preflight(
            manifest_path,
            mcp_url=probe_url,
            probe_fn=probe_fn,
        )
        _write_routing_contract(contract_dir, "C2", treatment_contract)
        if treatment_contract["infrastructure_status"] != "READY":
            failures.append(
                "treatment infrastructure is not READY: "
                + treatment_contract["infrastructure_status"]
            )
            if has_control:
                control_contract = control_routing_status(manifest_path)
                _write_routing_contract(contract_dir, "C1", control_contract)
            return treatment_contract, control_contract, failures
    if has_control:
        control_contract = control_routing_status(manifest_path)
        _write_routing_contract(contract_dir, "C1", control_contract)
    return treatment_contract, control_contract, []


def _execute_pairs(
    pair_ids: list[str],
    plan: dict[str, Any],
    root: Path,
    harbor_version: str,
    ledger_path: Path,
    ledger: dict[str, Any],
    command_runner: CommandRunner,
) -> bool:
    """Execute all pairs; return ``False`` if the ledger should be returned early."""

    for pair_id in pair_ids:
        pair_runs = [run for run in plan["runs"] if run["pair_id"] == pair_id]
        for run in pair_runs:
            if not _execute_run(
                run=run,
                root=root,
                harbor_version=harbor_version,
                ledger_path=ledger_path,
                ledger=ledger,
                command_runner=command_runner,
            ):
                return False
        if pair_id not in ledger["completed_pairs"]:
            ledger["completed_pairs"].append(pair_id)
        if _budget_exceeded(ledger):
            ledger["status"] = "INCOMPLETE"
            ledger["validation_failures"].append(
                "frozen budget exceeded at a complete pair boundary"
            )
            _write_ledger(ledger_path, ledger)
            return False
        _write_ledger(ledger_path, ledger)
    return True


def execute_plan(
    plan_path: Path,
    ledger_path: Path,
    *,
    manifest_path: Path,
    probe_url: str = "",
    command_runner: CommandRunner = _default_command,
    probe_fn: Any | None = None,
) -> dict[str, Any]:
    plan, plan_digest, pair_ids = _validated_plan(plan_path)
    manifest_digest = str(plan["manifest_digest"])
    if _file_digest(manifest_path) != manifest_digest:
        raise HarborSuiteError("held-out manifest digest does not match the run plan")
    harbor_version = _resolve_harbor_version(manifest_path)
    ledger = _initial_ledger(ledger_path, plan, plan_digest, manifest_digest)
    if ledger["status"] == "COMPLETE":
        return ledger
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    contract_dir = ledger_path.parent
    treatment_contract, control_contract, routing_failures = _setup_routing_contracts(
        plan, manifest_path, contract_dir, probe_url, probe_fn
    )
    if routing_failures:
        ledger["status"] = "INCOMPLETE"
        ledger["validation_failures"].extend(routing_failures)
        ledger["routing_status"] = {}
        if treatment_contract is not None:
            ledger["routing_status"]["C2"] = treatment_contract
        if control_contract is not None:
            ledger["routing_status"]["C1"] = control_contract
        _write_ledger(ledger_path, ledger)
        return ledger
    if treatment_contract is not None:
        ledger["routing_status"] = {"C2": treatment_contract}
    if control_contract is not None:
        ledger.setdefault("routing_status", {})["C1"] = control_contract
    _write_ledger(ledger_path, ledger)
    root = plan_path.parent
    if not _execute_pairs(
        pair_ids, plan, root, harbor_version, ledger_path, ledger, command_runner
    ):
        return ledger
    ledger["status"] = "COMPLETE"
    _write_ledger(ledger_path, ledger)
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-url", default="")
    args = parser.parse_args()
    ledger = execute_plan(
        args.run_plan,
        args.ledger,
        manifest_path=args.manifest,
        probe_url=args.probe_url,
    )
    print(args.ledger)
    return 0 if ledger["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute_plan"]
