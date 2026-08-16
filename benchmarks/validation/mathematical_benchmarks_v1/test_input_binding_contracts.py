"""Generic input-binding contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _metadata,
    _verifier,
)


def test_decoupled_input_binding_contract_metadata_is_preserved() -> None:
    expected = {
        "extremal-subset-sum-semantic-audit",
        "convergence-mode-separation",
        "cyclic-lipschitz-duality",
        "cyclic-polynomial-sum-audit",
        "edge-pair-ordering-audit",
        "image-complement-commutation",
        "multiplicative-grid-extremum",
        "well-total-domination-counterexample",
    }
    assert expected <= set(_fixtures.VERIFIER_TASKS)
    for task_name in expected:
        assert _metadata.is_input_binding_decoupled(task_name) is True
        metadata = _metadata.load_task_contract_metadata(task_name)
        assert metadata.get("input_binding_decoupled") is True


@pytest.mark.parametrize("task_name", _fixtures.VERIFIER_TASKS)
def test_verifiers_reject_replaced_workspace_inputs(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, task_name, "computed")
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["task_id"] = "tampered"
    _fixtures._write_json(input_path, input_data)

    assert _verifier._run_verifier(task, app, logs).reward == 0.0


@pytest.mark.parametrize("task_name", _fixtures.VERIFIER_TASKS)
@pytest.mark.parametrize(
    "replacement", ("{", "[]"), ids=("invalid-json", "wrong-shape")
)
def test_verifiers_fail_closed_on_malformed_workspace_inputs(
    tmp_path: Path,
    task_name: str,
    replacement: str,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, task_name, "computed")
    (app / "input.json").write_text(replacement)

    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_verifier_rejects_symlinked_workspace_input(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(
        tmp_path, _fixtures.RATIONAL_TASK, "computed"
    )
    input_path = app / "input.json"
    input_path.unlink()
    frozen_input = next((task / "tests").glob("*input*.json"))
    input_path.symlink_to(frozen_input)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
