"""Single developer-command contract consumed by help text and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandContract:
    name: str
    help: str
    mutates_checkout: bool
    scope: str
    ci_relationship: str
    cost_class: str


PUBLIC_COMMANDS: tuple[CommandContract, ...] = (
    CommandContract(
        name="setup",
        help="Install the locked Python environment.",
        mutates_checkout=False,
        scope="environment",
        ci_relationship="local setup only; CI uses its own setup actions",
        cost_class="once",
    ),
    CommandContract(
        name="quick",
        help="Cheap iteration: lint and unit tests.",
        mutates_checkout=False,
        scope="lint + tests/unit",
        ci_relationship="subset of the unit CI lane plus lint",
        cost_class="fast",
    ),
    CommandContract(
        name="check",
        help="Routine local handoff: lint, types, and unit tests.",
        mutates_checkout=False,
        scope="lint + typecheck + tests/unit",
        ci_relationship="subset of required PR jobs, not PR-equivalent",
        cost_class="bounded",
    ),
    CommandContract(
        name="check-all",
        help="Reproduce the ordinary Python CI lanes locally.",
        mutates_checkout=False,
        scope="lint + typecheck + unit/component/domain/composition",
        ci_relationship="local equivalent of the python matrix, not all CI jobs",
        cost_class="broad",
    ),
    CommandContract(
        name="check-external",
        help="Pinned Lean specialist lane only.",
        mutates_checkout=False,
        scope="tests/unit/domains/test_logic_operations.py",
        ci_relationship="local equivalent of the Lean job when the toolchain exists",
        cost_class="specialist",
    ),
    CommandContract(
        name="fix",
        help="Apply Ruff fixes and formatting.",
        mutates_checkout=True,
        scope="src tests benchmarks formatting",
        ci_relationship="not a CI job; CI expects an already-formatted tree",
        cost_class="fast",
    ),
)

PUBLIC_COMMAND_NAMES: tuple[str, ...] = tuple(
    command.name for command in PUBLIC_COMMANDS
)
COMMAND_BY_NAME: dict[str, CommandContract] = {
    command.name: command for command in PUBLIC_COMMANDS
}
