from __future__ import annotations

import subprocess
from pathlib import Path

from tools.command_contract import (
    COMMAND_BY_NAME,
    PUBLIC_COMMAND_NAMES,
    PUBLIC_COMMANDS,
)

ROOT = Path(__file__).parents[4]


def _listed_help_lines(target: str) -> list[str]:
    completed = subprocess.run(
        ["make", "--no-print-directory", target],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith("  ") and line.split()
    ]


def _listed_command_order(target: str) -> list[str]:
    return [line.split()[0] for line in _listed_help_lines(target)]


def test_help_lists_only_the_primary_developer_workflow() -> None:
    assert _listed_command_order("help") == list(PUBLIC_COMMAND_NAMES)


def test_help_descriptions_match_the_command_contract() -> None:
    listed = {
        line.split()[0]: line.split(None, 1)[1]
        for line in _listed_help_lines("help")
        if len(line.split(None, 1)) == 2
    }
    for command in PUBLIC_COMMANDS:
        assert listed[command.name] == command.help
        assert command.scope
        assert command.ci_relationship
        assert command.cost_class
        assert command.name in COMMAND_BY_NAME


def test_makefile_public_commands_match_the_command_contract() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    declared = next(
        line.split(":=", 1)[1].split()
        for line in makefile.splitlines()
        if line.startswith("PUBLIC_COMMANDS :=")
    )
    assert tuple(declared) == PUBLIC_COMMAND_NAMES
    assert COMMAND_BY_NAME["check"].mutates_checkout is False
    assert COMMAND_BY_NAME["fix"].mutates_checkout is True
    assert "not PR-equivalent" in COMMAND_BY_NAME["check"].ci_relationship
    assert "not all CI" in COMMAND_BY_NAME["check-all"].ci_relationship
    assert COMMAND_BY_NAME["check-external"].scope == "tests/boundary/providers/lean"


def test_help_all_retains_specialist_and_compatibility_commands() -> None:
    commands = set(_listed_command_order("help-all"))

    assert commands >= set(PUBLIC_COMMAND_NAMES)
    assert commands >= {
        "test-unit",
        "test-component",
        "test-domain",
        "test-composition",
        "test-storage",
        "test-process",
        "test-mcp",
        "test-provider",
        "test-lean",
        "test-e2e",
        "harbor-plan",
        "harbor-prepare-task",
        "harbor-validate-task",
        "deploy-check",
        "npm-test",
        "test-all-ci",
        "validation-status",
    }
    assert "precommit" in commands
