"""Tests for the selected-task Harbor developer workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "harbor_task_workflow.py"
_SPEC = importlib.util.spec_from_loader(
    "harbor_task_workflow", SourceFileLoader("harbor_task_workflow", str(TOOL_PATH))
)
assert _SPEC is not None and _SPEC.loader is not None
workflow = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = workflow
_SPEC.loader.exec_module(workflow)


def test_resolve_selection_uses_planner_owned_host_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = workflow._load_planner()
    monkeypatch.setattr(planner, "_digest", lambda path: f"sha256:{path.name}")
    monkeypatch.setattr(workflow, "_load_planner", lambda: planner)

    selection = workflow.resolve_selection(
        "mathematical-benchmarks-v1", ("parameterized-sharp-bound-audit",)
    )

    assert selection.tasks == ("parameterized-sharp-bound-audit",)
    assert [(item.selector, item.keyword) for item in selection.host_validations] == [
        (
            "benchmarks/validation/mathematical_benchmarks_v1/"
            "test_generic_verifier_contracts.py",
            "parameterized-sharp-bound-audit",
        ),
        (
            "benchmarks/validation/mathematical_benchmarks_v1/"
            "test_parameterized_sharp_bound_audit.py",
            "",
        ),
    ]


def test_prepare_formats_owned_python_and_reports_generated_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = tmp_path / "benchmarks/datasets/dataset-one/task-a"
    verifier = task / "tests/verifier.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("unformatted")
    dockerfile = task / "environment/Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("checksum=old")
    leaf = tmp_path / "benchmarks/validation/dataset_one/test_task_a.py"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("unformatted")
    shared = leaf.parent / "test_generic_verifier_contracts.py"
    shared.write_text("shared")
    selection = workflow.TaskSelection("dataset-one", ("task-a",), (task,), ())
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(
        label: str,
        arguments: tuple[str, ...],
        *,
        timings: list[Any],
        timeout_seconds: float,
        executable: str | None = None,
    ) -> None:
        del timeout_seconds, executable
        calls.append((label, arguments))
        if label == "format":
            verifier.write_text("formatted")
            leaf.write_text("formatted")
        if label == "sync-verifier-checksum":
            dockerfile.write_text("checksum=new")
        timings.append(workflow.StageTiming(label, 0.1))

    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "_run_checked", fake_run)

    changed = workflow.prepare(selection)

    format_arguments = calls[0][1]
    assert verifier.relative_to(tmp_path).as_posix() in format_arguments
    assert leaf.relative_to(tmp_path).as_posix() in format_arguments
    assert shared.relative_to(tmp_path).as_posix() not in format_arguments
    assert changed == (
        "benchmarks/datasets/dataset-one/task-a/environment/Dockerfile",
        "benchmarks/datasets/dataset-one/task-a/tests/verifier.py",
        "benchmarks/validation/dataset_one/test_task_a.py",
    )
    output = capsys.readouterr().out
    assert (
        "formatted: benchmarks/datasets/dataset-one/task-a/tests/verifier.py" in output
    )
    assert (
        "generated: benchmarks/datasets/dataset-one/task-a/environment/Dockerfile"
        in output
    )


def test_validate_orders_stages_isolates_pytest_and_reports_oracle_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selection = workflow.TaskSelection(
        "dataset-one",
        ("task-a",),
        (tmp_path / "benchmarks/datasets/dataset-one/task-a",),
        (
            workflow.HostValidation(
                "specific", "benchmarks/validation/test_leaf.py", ""
            ),
            workflow.HostValidation(
                "generic", "benchmarks/validation/test_generic.py", "task-a"
            ),
        ),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(
        label: str,
        arguments: tuple[str, ...],
        *,
        timings: list[Any],
        timeout_seconds: float,
        executable: str | None = None,
    ) -> None:
        del timeout_seconds, executable
        calls.append((label, arguments))
        timings.append(workflow.StageTiming(label, 0.1))
        if label == "oracle:task-a":
            evidence = (
                tmp_path
                / "benchmarks/results/dataset-one-oracle/job/oracle-evidence.json"
            )
            evidence.parent.mkdir(parents=True)
            evidence.write_text(
                json.dumps(
                    {
                        "dataset": "dataset-one",
                        "tasks": [{"task": "task-a", "digest": "sha256:digest"}],
                    }
                )
            )

    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(
        workflow, "PYTEST_ROOT", tmp_path / ".pytest_cache/harbor-validation"
    )
    monkeypatch.setattr(workflow, "_run_checked", fake_run)

    evidence = workflow.validate(selection)

    assert [label for label, _ in calls] == [
        "static-quality",
        "contracts",
        "host:specific",
        "host:generic",
        "oracle:task-a",
    ]
    host_commands = [
        arguments for label, arguments in calls if label.startswith("host:")
    ]
    assert host_commands[0][-1] == "benchmarks/validation/test_leaf.py"
    assert host_commands[1][-2:] == ("-k", "task-a")
    assert all("-n" in command and "0" in command for command in host_commands)
    assert not list((tmp_path / ".pytest_cache/harbor-validation").glob("run-*"))
    assert evidence == (
        (
            "task-a",
            "sha256:digest",
            "benchmarks/results/dataset-one-oracle/job/oracle-evidence.json",
        ),
    )
    assert "digest=sha256:digest" in capsys.readouterr().out


def test_validate_fails_fast_after_static_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = workflow.TaskSelection("dataset-one", ("task-a",), (tmp_path,), ())
    calls: list[str] = []

    def fail_static(
        label: str,
        arguments: tuple[str, ...],
        *,
        timings: list[Any],
        timeout_seconds: float,
        executable: str | None = None,
    ) -> None:
        del arguments, timings, timeout_seconds, executable
        calls.append(label)
        raise workflow.TaskWorkflowError("static failed")

    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(
        workflow, "PYTEST_ROOT", tmp_path / ".pytest_cache/harbor-validation"
    )
    monkeypatch.setattr(workflow, "_run_checked", fail_static)

    with pytest.raises(workflow.TaskWorkflowError, match="static failed"):
        workflow.validate(selection)

    assert calls == ["static-quality"]


def test_oracle_evidence_freshness_rejects_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = workflow.TaskSelection("dataset-one", ("task-a",), (tmp_path,), ())
    evidence = (
        tmp_path / "benchmarks/results/dataset-one-oracle/job/oracle-evidence.json"
    )
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "dataset": "dataset-one",
                "tasks": [{"task": "task-a", "digest": "sha256:old"}],
            }
        )
    )
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    previous = workflow._oracle_evidence_snapshot(selection)

    with pytest.raises(workflow.TaskWorkflowError, match="no fresh evidence"):
        workflow._fresh_oracle_evidence(selection, "task-a", previous=previous)

    evidence.write_text(
        json.dumps(
            {
                "dataset": "dataset-one",
                "tasks": [{"task": "task-a", "digest": "sha256:new"}],
            }
        )
    )

    assert workflow._fresh_oracle_evidence(selection, "task-a", previous=previous) == (
        "sha256:new",
        "benchmarks/results/dataset-one-oracle/job/oracle-evidence.json",
    )
