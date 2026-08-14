from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.heldout_runner import (
    _default_command,
    _json_digest,
    execute_plan,
)


def _manifest(tmp_path: Path) -> Path:
    manifest = {
        "schema_version": "3",
        "bundle_id": "test-held-out-v1",
        "bundle_version": "1.0.0",
        "snapshot_lock": {
            "lock_id": "sha256:" + "f" * 64,
            "lock_uri": "s3://invalid/lock.json",
            "lock_digest": "sha256:" + "0" * 64,
        },
        "archive": {
            "uri": "s3://invalid/bundle.tar.gz",
            "sha256": "sha256:" + "d" * 64,
        },
        "dataset": {
            "id": "test-held-out-v1",
            "path": "dataset",
            "manifest_digest": "sha256:" + "e" * 64,
            "minimum_independent_families": 2,
        },
        "tasks": [
            {
                "id": f"task-{i}",
                "family": "family-a" if i < 3 else "family-b",
                "digest": "sha256:" + "a" * 64,
                "verifier_root": f"dataset/task-{i}/tests",
                "verifier_tree_digest": "sha256:" + "b" * 64,
                "oracle_root": f"dataset/task-{i}/solution",
                "oracle_tree_digest": "sha256:" + "c" * 64,
            }
            for i in range(5)
        ],
        "conditions": [
            {"id": "C1", "role": "PRIMARY_CONTROL", "jacobian_enabled": False},
            {
                "id": "C2",
                "role": "PRIMARY_TREATMENT",
                "jacobian_enabled": True,
                "image": "registry.invalid/jacobian@sha256:" + "1" * 64,
                "source_sha": "a" * 40,
                "platform": "linux/amd64",
                "server_version": "1.0.0",
                "catalog_digest": "sha256:" + "2" * 64,
            },
        ],
        "experiment": {
            "harbor_version": "0.20.0",
            "agent": {"name": "codex", "version": "1.0.0"},
            "model": "model",
            "prompt_path": "prompts/heldout.md",
            "prompt_digest": "sha256:" + "7" * 64,
            "reasoning_effort": "high",
            "randomization_seed": 104729,
            "max_tokens": 100,
            "max_cost_usd": 2.0,
            "stages": {
                "pilot": {
                    "task_ids": ["task-0", "task-1", "task-2"],
                    "repetitions": 3,
                },
                "decision": {
                    "task_ids": ["task-0", "task-1", "task-2", "task-3", "task-4"],
                    "repetitions": 5,
                },
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


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
        "schema_version": "3",
        "manifest_digest": "sha256:"
        + hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest(),
        "stage": "pilot",
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


def _ready_probe(*, mcp_url, expected_version, timeout_seconds):
    return {
        "reachable": True,
        "report": {
            "server": {"name": "jacobian", "version": "1.0.0"},
            "tool_names": ["math.find", "math.run"],
            "catalog": {
                "catalog_version": "1",
                "operations": 1,
                "catalog_digest": "sha256:" + "2" * 64,
                "sha256": "abc",
            },
            "discovery": {"bytes": 100, "matches": ["cap-1"]},
        },
    }


def _unreachable_probe(*, mcp_url, expected_version, timeout_seconds):
    return {"reachable": False, "diagnostic": "connection refused"}


def test_default_runner_forwards_only_harbor_authorized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "authorized")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("UNRELATED_SECRET", "hidden")
    captured: dict[str, object] = {}

    def fake_run_operator_command(*args: object, **kwargs: object):
        captured["environment"] = kwargs["environment"]
        return type("Result", (), {"exit_code": 0})()

    monkeypatch.setattr(
        "benchmarks.tooling.heldout_runner.run_operator_command",
        fake_run_operator_command,
    )

    assert _default_command(["uvx", "harbor", "run"]) == 0
    environment = captured["environment"]
    assert isinstance(environment, Mapping)
    assert environment["CODEX_FORCE_AUTH_JSON"] == "authorized"
    assert environment["HTTPS_PROXY"] == "http://proxy.invalid"
    assert "UNRELATED_SECRET" not in environment


def test_runner_executes_whole_pair_and_can_resume_exact_plan(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    calls, runner = _runner()

    ledger = execute_plan(
        plan,
        ledger_path,
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=_ready_probe,
    )
    resumed = execute_plan(
        plan,
        ledger_path,
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=_ready_probe,
    )

    assert ledger["status"] == "COMPLETE"
    assert ledger["schema_version"] == "2"
    assert (
        ledger["manifest_digest"]
        == json.loads(plan.read_text(encoding="utf-8"))["manifest_digest"]
    )
    assert ledger["completed_pairs"] == ["task-r001"]
    assert ledger["usage"] == {"tokens": 30, "cost_usd": 0.2}
    assert resumed == ledger
    assert len(calls) == 2
    assert all(
        command[:4] == ["uvx", "--from", "harbor==0.20.0", "harbor"]
        for command in calls
    )


def test_runner_persists_routing_status_contracts(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    _calls, runner = _runner()

    execute_plan(
        plan,
        ledger_path,
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=_ready_probe,
    )

    c1_contract_path = ledger_path.parent / "routing-status-c1.json"
    c2_contract_path = ledger_path.parent / "routing-status-c2.json"
    assert c1_contract_path.is_file()
    assert c2_contract_path.is_file()
    c1 = json.loads(c1_contract_path.read_text(encoding="utf-8"))
    c2 = json.loads(c2_contract_path.read_text(encoding="utf-8"))
    assert c1["infrastructure_status"] == "NOT_CONFIGURED"
    assert c1["routing_status"] == "NOT_APPLICABLE"
    assert c1["condition_id"] == "C1"
    assert c2["infrastructure_status"] == "READY"
    assert c2["routing_status"] == "AVAILABLE_UNUSED"
    assert c2["condition_id"] == "C2"
    assert c2["probe"]["reachable"] is True
    assert c2["probe"]["server_version_observed"] == "1.0.0"
    assert c2["probe"]["catalog_digest_observed"] == "sha256:" + "2" * 64


def test_runner_aborts_when_treatment_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("benchmarks.tooling.heldout_routing.time.sleep", lambda _: None)
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    calls, runner = _runner()

    ledger = execute_plan(
        plan,
        ledger_path,
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=_unreachable_probe,
    )

    assert ledger["status"] == "INCOMPLETE"
    assert "not READY" in ledger["validation_failures"][0]
    assert "UNAVAILABLE" in ledger["validation_failures"][0]
    assert len(calls) == 0
    assert ledger["routing_status"]["C2"]["infrastructure_status"] == "UNAVAILABLE"
    assert ledger["routing_status"]["C2"]["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert ledger["routing_status"]["C1"]["infrastructure_status"] == "NOT_CONFIGURED"


def test_runner_retries_probe_and_succeeds_when_container_becomes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmarks.tooling.heldout_routing.time.sleep", lambda _: None)
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    calls, runner = _runner()
    probe_calls = 0

    def eventually_ready(*, mcp_url, expected_version, timeout_seconds):
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls < 2:
            return {"reachable": False, "diagnostic": "connection refused"}
        return _ready_probe(
            mcp_url=mcp_url,
            expected_version=expected_version,
            timeout_seconds=timeout_seconds,
        )

    ledger = execute_plan(
        plan,
        ledger_path,
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=eventually_ready,
    )

    assert ledger["status"] == "COMPLETE"
    assert probe_calls >= 2
    assert ledger["routing_status"]["C2"]["infrastructure_status"] == "READY"
    assert len(calls) == 2


def test_runner_aborts_when_no_probe_url(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    calls, runner = _runner()

    ledger = execute_plan(
        plan,
        ledger_path,
        manifest_path=manifest_path,
        probe_url="",
        command_runner=runner,
        probe_fn=_ready_probe,
    )

    assert ledger["status"] == "INCOMPLETE"
    assert "not READY" in ledger["validation_failures"][0]
    assert "MISCONFIGURED" in ledger["validation_failures"][0]
    assert len(calls) == 0
    assert ledger["routing_status"]["C2"]["infrastructure_status"] == "MISCONFIGURED"
    assert ledger["routing_status"]["C2"]["routing_status"] == "CONFIGURED_UNCALLABLE"


def test_runner_marks_pair_boundary_budget_overage_incomplete(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path, max_tokens=20)
    calls, runner = _runner()

    ledger = execute_plan(
        plan,
        tmp_path / "ledger.json",
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=_ready_probe,
    )

    assert len(calls) == 2
    assert ledger["status"] == "INCOMPLETE"
    assert "pair boundary" in ledger["validation_failures"][0]


def test_runner_fails_closed_when_accounting_is_missing(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    _calls, runner = _runner(missing_cost=True)

    ledger = execute_plan(
        plan,
        tmp_path / "ledger.json",
        manifest_path=manifest_path,
        probe_url="http://127.0.0.1:8000/mcp",
        command_runner=runner,
        probe_fn=_ready_probe,
    )

    assert ledger["status"] == "INCOMPLETE"
    assert "accounting" in ledger["validation_failures"][0]


def test_runner_rejects_ledger_from_another_plan(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "2",
                "manifest_digest": "sha256:" + "a" * 64,
                "plan_digest": "sha256:" + "0" * 64,
            }
        )
    )

    with pytest.raises(HarborSuiteError, match="different run plan"):
        execute_plan(
            plan,
            ledger,
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )


def test_runner_rejects_ledger_from_different_manifest(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    ledger = tmp_path / "ledger.json"
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "2",
                "manifest_digest": "sha256:" + "z" * 64,
                "plan_digest": plan_data["plan_digest"],
            }
        )
    )

    with pytest.raises(HarborSuiteError, match="different manifest"):
        execute_plan(
            plan,
            ledger,
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )


def test_runner_rejects_manifest_that_does_not_match_plan(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"]["model"] = "different-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HarborSuiteError, match="manifest digest"):
        execute_plan(
            plan,
            tmp_path / "ledger.json",
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )


def test_runner_rejects_v2_plan(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["schema_version"] = "2"
    plan_data["harbor_version"] = "0.20.0"
    plan_data.pop("manifest_digest")
    plan_path = tmp_path / "run-plan-v2.json"
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        execute_plan(
            plan_path,
            tmp_path / "ledger.json",
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )


def test_runner_rejects_plan_with_extra_field(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["legacy_field"] = "should be rejected"
    plan_path = tmp_path / "run-plan-extra.json"
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        execute_plan(
            plan_path,
            tmp_path / "ledger.json",
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )


def test_runner_rejects_plan_with_missing_manifest_digest(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    del plan_data["manifest_digest"]
    plan_path = tmp_path / "run-plan-no-digest.json"
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        execute_plan(
            plan_path,
            tmp_path / "ledger.json",
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )


def test_runner_rejects_plan_with_wrong_condition(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    plan = _plan(tmp_path)
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["runs"][0]["condition"] = "C3"
    plan_path = tmp_path / "run-plan-bad-condition.json"
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        execute_plan(
            plan_path,
            tmp_path / "ledger.json",
            manifest_path=manifest_path,
            probe_url="http://127.0.0.1:8000/mcp",
            probe_fn=_ready_probe,
        )
