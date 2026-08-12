from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[4]
PRIMARY_COMMANDS = {
    "setup",
    "quick",
    "check",
    "check-external",
    "fix",
}
PRIMARY_COMMAND_ORDER = [
    "setup",
    "quick",
    "check",
    "check-external",
    "fix",
]


def _listed_command_order(target: str) -> list[str]:
    completed = subprocess.run(
        ["make", "--no-print-directory", target],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.split()[0]
        for line in completed.stdout.splitlines()
        if line.startswith("  ") and line.split()
    ]


def test_help_lists_only_the_primary_developer_workflow() -> None:
    assert _listed_command_order("help") == PRIMARY_COMMAND_ORDER


def test_help_all_retains_specialist_and_compatibility_commands() -> None:
    commands = set(_listed_command_order("help-all"))

    assert commands >= PRIMARY_COMMANDS
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
    }
