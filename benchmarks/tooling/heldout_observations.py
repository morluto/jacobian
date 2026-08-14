"""Held-out Harbor evidence collection and treatment routing transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.tooling import observation_artifacts, observation_selection
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.heldout_bundle import validate_manifest
from benchmarks.tooling.observation_results import (
    _git_sha,
    _json_digest,
    _read_json,
    _resolve_binding,
    _sha256,
    _validate_contract,
    build_observation_evidence,
)


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
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--condition", choices=("C1", "C2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_heldout_evidence"]
