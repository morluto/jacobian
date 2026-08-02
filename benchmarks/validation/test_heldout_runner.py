from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.harbor_suite import HarborSuiteError
from benchmarks.tooling.heldout_runner import _json_digest, execute_plan


def _plan(tmp_path: Path, *, max_tokens: int = 100) -> Path:
    runs = []
    for condition in ("C2", "C1"):
        root = tmp_path / "runs" / "task-r001" / condition.lower()
        root.mkdir(parents=True)
        job = root / "job.json"
        job.write_text("{}\n", encoding="utf-8")
        runs.append(
            {
                "pair_id": "task-r001",
                "condition": condition,
                "job": job.relative_to(tmp_path).as_posix(),
                "jobs_dir": (root / "results").relative_to(tmp_path).as_posix(),
            }
        )
    plan = {
        "schema_version": "2",
        "harbor_version": "0.20.0",
        "pair_count": 1,
        "budget": {
            "max_tokens": max_tokens,
            "max_cost_usd": 2.0,
            "enforcement": "PAIR_BOUNDARY_POST_RUN",
            "missing_accounting": "INCOMPLETE",
            "overage": "INCOMPLETE",
        },
        "runs": runs,
    }
    plan["plan_digest"] = _json_digest(plan)
    path = tmp_path / "run-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _runner(*, missing_cost: bool = False):
    calls: list[list[str]] = []

    def run(command: list[str]) -> int:
        calls.append(command)
        job = Path(command[-1])
        result_root = job.parent / "results" / "job"
        result_root.mkdir(parents=True)
        agent_result = {"n_input_tokens": 10, "n_output_tokens": 5}
        if not missing_cost:
            agent_result["cost_usd"] = 0.1
        result = {
            "stats": {
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
            "trial_results": [{"agent_result": agent_result, "exception_info": None}],
        }
        (result_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return 0

    return calls, run


def test_runner_executes_whole_pair_and_can_resume_exact_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    calls, runner = _runner()

    ledger = execute_plan(plan, ledger_path, command_runner=runner)
    resumed = execute_plan(plan, ledger_path, command_runner=runner)

    assert ledger["status"] == "COMPLETE"
    assert ledger["completed_pairs"] == ["task-r001"]
    assert ledger["usage"] == {"tokens": 30, "cost_usd": 0.2}
    assert resumed == ledger
    assert len(calls) == 2
    assert all(
        command[:4] == ["uvx", "--from", "harbor==0.20.0", "harbor"]
        for command in calls
    )


def test_runner_marks_pair_boundary_budget_overage_incomplete(tmp_path: Path) -> None:
    plan = _plan(tmp_path, max_tokens=20)
    calls, runner = _runner()

    ledger = execute_plan(plan, tmp_path / "ledger.json", command_runner=runner)

    assert len(calls) == 2
    assert ledger["status"] == "INCOMPLETE"
    assert "pair boundary" in ledger["validation_failures"][0]


def test_runner_fails_closed_when_accounting_is_missing(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _calls, runner = _runner(missing_cost=True)

    ledger = execute_plan(plan, tmp_path / "ledger.json", command_runner=runner)

    assert ledger["status"] == "INCOMPLETE"
    assert "accounting" in ledger["validation_failures"][0]


def test_runner_rejects_ledger_from_another_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"plan_digest": "sha256:" + "0" * 64}))

    with pytest.raises(HarborSuiteError, match="different run plan"):
        execute_plan(plan, ledger)
