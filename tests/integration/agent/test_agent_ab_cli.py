from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks import agent_ab as benchmark
from tests.integration.agent._agent_ab_support import _write_private_case

from jacobian.contracts.capabilities import (
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.kernel import JacobianKernel


def test_agent_eval_is_plan_only_without_explicit_execute(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    main = benchmark.main

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plan mode started a model evaluation")

    monkeypatch.setitem(main.__globals__, "_run_condition", unexpected_run)

    assert main(["--case", "ERDOS-STRAUS-AB-001"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan"
    assert plan["execution_requested"] is False
    assert plan["model_run_count"] == 2
    assert plan["maximum_model_wall_seconds"] == 1200


def test_agent_eval_requires_explicit_case_selection() -> None:
    main = benchmark.main

    with pytest.raises(SystemExit):
        main([])


def test_agent_eval_plan_accepts_xhigh_reasoning(
    capsys: Any,
) -> None:
    main = benchmark.main

    assert (
        main(
            [
                "--case",
                "ERDOS-STRAUS-AB-001",
                "--reasoning-effort",
                "xhigh",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["reasoning_effort"] == "xhigh"


def test_agent_eval_requires_sufficient_manual_run_budget(tmp_path: Path) -> None:
    main = benchmark.main
    case_path = _write_private_case(tmp_path)

    with pytest.raises(SystemExit):
        main(
            [
                "--case-file",
                str(case_path),
                "--execute",
                "--max-model-runs",
                "3",
            ]
        )


def test_agent_eval_plan_counts_each_lean_capability_condition(
    tmp_path: Path,
    capsys: Any,
) -> None:
    main = benchmark.main
    case_path = _write_private_case(tmp_path)

    assert (
        main(
            [
                "--case-file",
                str(case_path),
                "--repetitions",
                "2",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["model_run_count"] == 8
    assert plan["cases"][0]["conditions"] == [
        "baseline",
        "tactic",
        "retrieval",
        "combined",
    ]


@pytest.mark.lean_runtime
@pytest.mark.usefixtures("initialized_kernel_store_with_references")
def test_ab_lean_control_ablation_removes_only_declaration_discovery(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(
        tmp_path,
        install_references=True,
        capability_exclusions=frozenset(
            {
                "lean.declaration.search",
                "lean.declaration.inspect",
            }
        ),
    )

    lean_ids = {
        descriptor.capability_id
        for descriptor in kernel.capabilities.catalog().capabilities
        if descriptor.capability_id.startswith("lean.")
    }

    assert lean_ids == {
        "lean.check",
        "lean.declaration.dependencies",
        "lean.proof_edit.validate",
        "lean.proof_state.apply_tactic",
        "lean.retrieve.premises",
        "lean.statement.compare",
        "lean.statement.propose",
    }
    excluded = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.declaration.search",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "MATHLIB",
                "name_contains": "revzip",
                "result_limit": 1,
            },
        )
    )
    assert excluded.execution.status.value == "ERROR"
    assert excluded.diagnostics[0].code == "UNKNOWN_CAPABILITY"


def test_ab_lean_codex_command_uses_same_mcp_with_control_ablation(
    tmp_path: Path,
) -> None:
    codex_command = benchmark._codex_command
    control = codex_command(
        codex_command="codex",
        condition="control",
        workspace=tmp_path / "workspace",
        report_path=tmp_path / "report.json",
        state_dir=tmp_path / "state",
        model="gpt-5.6",
        reasoning_effort="high",
        task_type="lean_declaration",
    )
    treatment = codex_command(
        codex_command="codex",
        condition="treatment",
        workspace=tmp_path / "workspace",
        report_path=tmp_path / "report.json",
        state_dir=tmp_path / "state",
        model="gpt-5.6",
        reasoning_effort="high",
        task_type="lean_declaration",
    )

    assert "agent_ab_mcp.py" in " ".join(control)
    assert "agent_ab_mcp.py" in " ".join(treatment)
    assert " ".join(control).count("--exclude-capability") == 2
    assert "--exclude-capability" not in " ".join(treatment)
