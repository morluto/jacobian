"""Generic input-binding contract tests.

Cross-task invariants for workspace input binding: replaced, malformed, and
symlinked inputs must fail closed, and decoupled tasks must report
``input_binding`` independently of mathematical correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support


def test_decoupled_input_binding_contract_metadata_is_preserved() -> None:
    expected = {
        "extremal-subset-sum-semantic-audit",
    }
    assert expected <= set(support.VERIFIER_TASKS)
    for task_name in expected:
        assert support.is_input_binding_decoupled(task_name) is True
        metadata = support.load_task_contract_metadata(task_name)
        assert metadata.get("input_binding_decoupled") is True


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
def test_verifiers_reject_replaced_workspace_inputs(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["task_id"] = "tampered"
    support._write_json(input_path, input_data)

    rejected = support._run_verifier(task, app, logs)
    if support.is_input_binding_decoupled(task_name):
        assert rejected.details["correctness"] == 1.0
        assert rejected.details["input_binding"] == 0.0
    else:
        assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


@pytest.mark.parametrize("task_name", support.VERIFIER_TASKS)
@pytest.mark.parametrize(
    "replacement", ("{", "[]"), ids=("invalid-json", "wrong-shape")
)
def test_verifiers_fail_closed_on_malformed_workspace_inputs(
    tmp_path: Path,
    task_name: str,
    replacement: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    (app / "input.json").write_text(replacement)

    rejected = support._run_verifier(task, app, logs)
    if support.is_input_binding_decoupled(task_name):
        # Mathematical correctness is reported independently of input binding;
        # the result is still canonical, so correctness stays 1.0 while the
        # separate input_binding diagnostic captures the tamper.
        assert rejected.details["correctness"] == 1.0
        assert rejected.details["input_binding"] == 0.0
    else:
        assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_verifier_rejects_symlinked_workspace_input(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, support.RATIONAL_TASK, "computed")
    input_path = app / "input.json"
    input_path.unlink()
    frozen_input = next((task / "tests").glob("*input*.json"))
    input_path.symlink_to(frozen_input)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
