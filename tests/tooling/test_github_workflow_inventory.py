from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from tools.command_runner import ToolCommandResult, ToolCommandStatus
from tools.inventory_github_workflows import (
    WORKFLOW_LIST_LIMIT,
    _registered_from_gh,
    _stems_from_workflow_rows,
    _workflow_list_command,
    classify,
    workflow_stems,
)


def test_workflow_stems_read_yaml_files(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    (workflows / "benchmarks.yaml").write_text("name: benchmarks\n", encoding="utf-8")
    (workflows / "notes.txt").write_text("ignored\n", encoding="utf-8")

    assert workflow_stems(tmp_path) == {"ci", "benchmarks"}


def test_classify_historical_agent_leftovers() -> None:
    report = classify(
        {"ci", "agent-port-alpha", "agent-rebase-beta", "retired-other"},
        {"ci"},
    )

    assert report["historical_agent_leftovers"] == (
        "agent-port-alpha",
        "agent-rebase-beta",
    )
    assert report["registered_without_files"] == (
        "agent-port-alpha",
        "agent-rebase-beta",
        "retired-other",
    )
    assert report["files_without_registration"] == ()


def test_workflow_list_command_requests_every_registration() -> None:
    command = _workflow_list_command()

    assert "--all" in command
    assert "--limit" in command
    limit = int(command[command.index("--limit") + 1])
    assert limit >= 1000
    assert limit == WORKFLOW_LIST_LIMIT


def test_workflow_list_fails_closed_when_gh_hits_the_limit() -> None:
    with pytest.raises(SystemExit, match="hit the requested limit"):
        _stems_from_workflow_rows(
            [
                {"path": f".github/workflows/w{index}.yml"}
                for index in range(WORKFLOW_LIST_LIMIT)
            ]
        )


def test_registered_workflows_use_the_bounded_operator_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.inventory_github_workflows as inventory

    observed: dict[str, object] = {}

    def run_operator(
        command: str,
        arguments: list[str],
        **kwargs: object,
    ) -> ToolCommandResult:
        observed.update(command=command, arguments=arguments, **kwargs)
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b'[{"path": ".github/workflows/ci.yml"}]',
            stderr=b"",
        )

    monkeypatch.setattr(inventory, "run_operator_command", run_operator)

    assert _registered_from_gh() == {"ci"}
    assert observed["command"] == "gh"
    assert observed["timeout_seconds"] == 30.0


def test_registered_workflows_preserve_gh_authentication_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.inventory_github_workflows as inventory

    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setenv("UNRELATED_SECRET", "not-forwarded")
    observed: dict[str, object] = {}

    def run_operator(*_args: object, **kwargs: object) -> ToolCommandResult:
        observed.update(kwargs)
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"[]",
            stderr=b"",
        )

    monkeypatch.setattr(inventory, "run_operator_command", run_operator)

    assert _registered_from_gh() == set()
    environment = observed["environment"]
    assert isinstance(environment, Mapping)
    assert environment["GH_TOKEN"] == "token"
    assert "UNRELATED_SECRET" not in environment
