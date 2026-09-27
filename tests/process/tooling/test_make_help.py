from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[3]


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
    lines = _listed_help_lines("help")
    commands = [line.split()[0] for line in lines]

    assert commands == [
        "setup",
        "test-focused",
        "quick-scoped",
        "handoff-scoped",
        "affected-plan",
        "affected",
        "test-timings",
        "check",
        "check-all",
        "fix",
    ]
    assert all(len(line.split(None, 1)) == 2 for line in lines)


def test_help_all_retains_specialist_and_compatibility_commands() -> None:
    commands = set(_listed_command_order("help-all"))

    assert commands >= {"setup", "test-focused", "affected", "check", "check-all"}
    assert commands >= {
        "test-math",
        "test-catalog",
        "test-dispatch",
        "test-cli",
        "test-tooling",
        "test-integration",
        "test-fast",
        "test-process",
        "test-mcp",
        "harbor-plan",
        "harbor-prepare-task",
        "harbor-validate-task",
        "npm-test",
        "test-full",
        "validation-status",
    }
    assert "precommit" in commands
