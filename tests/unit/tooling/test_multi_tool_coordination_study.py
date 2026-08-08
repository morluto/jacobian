from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from benchmarks.tooling.command_runner import ToolCommandStatus
from benchmarks.tooling.multi_tool_coordination_study import (
    _TASK_ORDER,
    _clean_room_terminal,
    _codex_arguments,
    _prepare_workspace,
    _terminal,
    load_spec,
)

ROOT = Path(__file__).parents[3]
SPEC = ROOT / "benchmarks/config/multi-tool-coordination-pr1.json"
ADJUDICATION = ROOT / "benchmarks/config/multi-tool-coordination-pr1-adjudication.json"


def test_preregistration_freezes_bounded_cross_domain_matrix() -> None:
    spec = load_spec(SPEC)

    assert tuple(task.task_id for task in spec.tasks) == _TASK_ORDER
    assert spec.repetitions_per_task * len(spec.tasks) == 12
    assert len({task.domain for task in spec.tasks}) == 6
    assert spec.wrong_answer_retries == 0
    assert spec.tool_call_reward == 0
    assert spec.reasoning_log_mode == "REQUIRED"
    assert not spec.causal_claim_authorized
    assert not spec.harbor_execution_claimed


def test_workspace_contains_only_public_task_files_and_prompt(tmp_path: Path) -> None:
    spec = load_spec(SPEC)
    task = spec.tasks[0]
    source = ROOT / "benchmarks/datasets/mathematical-benchmarks-v1" / task.task_id

    _prepare_workspace(tmp_path, spec, source)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "input.json",
        "instruction.md",
        "prompt.txt",
        "submission_schema.json",
    ]
    assert json.loads((tmp_path / "input.json").read_text())["task_id"].startswith(
        "jacobian/"
    )
    assert "solution" not in (tmp_path / "prompt.txt").read_text().casefold()
    assert "verifier.py" not in (tmp_path / "prompt.txt").read_text()


def test_codex_arguments_bind_exact_model_reasoning_and_local_mcp(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC)
    args = _codex_arguments(
        workspace=tmp_path,
        spec=spec,
        mcp_url="http://127.0.0.1:8123/mcp",
        prompt="frozen prompt",
    )

    assert "gpt-5.4-mini" in args
    assert 'model_reasoning_effort="medium"' in args
    assert 'web_search="disabled"' in args
    assert 'mcp_servers.jacobian.url="http://127.0.0.1:8123/mcp"' in args
    assert "--ignore-user-config" in args
    assert "--ignore-rules" in args


@pytest.mark.parametrize(
    (
        "status",
        "exit_code",
        "verifier_status",
        "reasoning_status",
        "reward",
        "acceptance",
    ),
    [
        (ToolCommandStatus.EXITED, 0, "COMPLETED", "COMPLETE", 1.0, "ACCEPTED"),
        (ToolCommandStatus.EXITED, 0, "COMPLETED", "COMPLETE", 0.0, "REJECTED"),
        (
            ToolCommandStatus.TIMED_OUT,
            None,
            "COMPLETED",
            "COMPLETE",
            1.0,
            "INCONCLUSIVE",
        ),
        (
            ToolCommandStatus.START_FAILED,
            None,
            "COMPLETED",
            "COMPLETE",
            0.0,
            "INCONCLUSIVE",
        ),
        (ToolCommandStatus.EXITED, 0, "ERROR", "COMPLETE", 0.0, "INCONCLUSIVE"),
        (ToolCommandStatus.EXITED, 0, "COMPLETED", "INCOMPLETE", 1.0, "INCONCLUSIVE"),
    ],
)
def test_terminal_labels_fail_closed(
    status: ToolCommandStatus,
    exit_code: int | None,
    verifier_status: Literal["COMPLETED", "ERROR"],
    reasoning_status: str,
    reward: float,
    acceptance: str,
) -> None:
    observed, _ = _terminal(
        command_status=status,
        exit_code=exit_code,
        verifier_execution_status=verifier_status,
        reasoning_status=reasoning_status,
        reward=reward,
    )

    assert observed == acceptance


def test_legacy_verifier_can_leave_input_binding_unreported() -> None:
    evidence = _clean_room_terminal(
        acceptance="ACCEPTED",
        verifier_digest="sha256:" + "a" * 64,
        verifier_execution_status="COMPLETED",
        details={"evidence_validity": 1.0},
    )

    assert evidence.input_binding_valid is None
    assert evidence.artifact_binding_valid is True


def test_pr1_adjudication_binds_the_complete_frozen_batch() -> None:
    value = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    spec = load_spec(SPEC)
    expected_ids = {
        f"{task.task_id}-r{repetition:02d}"
        for task in spec.tasks
        for repetition in range(1, spec.repetitions_per_task + 1)
    }

    assert value["study_id"] == spec.study_id
    assert value["source_revision"] == "4eaf525a136e4473ddbc015a1d6a94aa0f3dd885"
    assert value["batch"]["run_count"] == len(expected_ids) == 12
    assert value["batch"]["outcomes"] == {"ACCEPTED": 3, "REJECTED": 9}
    assert value["batch"]["reasoning_protocol"] == {"COMPLETE": 12}
    assert {item["trajectory_id"] for item in value["runs"]} == expected_ids
    assert all(item["categories"] for item in value["runs"])
    assert all(
        category in value["taxonomy"]
        for item in value["runs"]
        for category in item["categories"]
    )
    assert (
        sum(
            item["mathematical_adjudication"] == "INVALID_TERMINAL_OBJECT"
            for item in value["runs"]
        )
        == 2
    )
    assert (
        sum(
            "verifier_contract_overconstraint" in item["categories"]
            for item in value["runs"]
        )
        == 3
    )
