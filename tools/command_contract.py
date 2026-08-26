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
        name="affected",
        help="Default local validation: CI-planned affected owners and scoped static checks.",
        mutates_checkout=False,
        scope="changed Python static paths plus CI-planned owner and boundary lanes from AFFECTED_BASE...HEAD",
        ci_relationship="uses the checked-in CI planner; installed-wheel evidence remains CI-owned",
        cost_class="fast",
    ),
    CommandContract(
        name="handoff-scoped",
        help="Focused handoff scoped to declared static PATHS and one owner test path.",
        mutates_checkout=False,
        scope="explicit Ruff and mypy paths + one explicit test path through its declared owner lane",
        ci_relationship="additive local evidence; the broad check and CI static gate remain authoritative",
        cost_class="fast",
    ),
    CommandContract(
        name="test-focused",
        help="Run TESTS through its explicit semantic LANE (for example, LANE=math).",
        mutates_checkout=False,
        scope="one explicit test path through its declared semantic owner lane",
        ci_relationship="uses the same owner-lane command and limits as CI",
        cost_class="fast",
    ),
    CommandContract(
        name="quick-scoped",
        help="Focused edit loop scoped to declared static PATHS and one owner test path.",
        mutates_checkout=False,
        scope="explicit Ruff paths + one explicit test path through its declared owner lane",
        ci_relationship="additive local evidence; the broad check and CI static gate remain authoritative",
        cost_class="fast",
    ),
    CommandContract(
        name="affected-plan",
        help="Show the CI-planned local validation selected from AFFECTED_BASE...HEAD.",
        mutates_checkout=False,
        scope="prints changed owners, boundary lanes, and scoped static paths without running them",
        ci_relationship="same checked-in CI planner as affected and pull-request CI",
        cost_class="fast",
    ),
    CommandContract(
        name="test-timings",
        help="Summarize pytest JUnit timing evidence; set TIMING for worker skew.",
        mutates_checkout=False,
        scope="one JUnit artifact plus optional xdist worker timing evidence",
        ci_relationship="reads the JUnit and timing artifacts retained by each CI lane",
        cost_class="fast",
    ),
    CommandContract(
        name="check",
        help="Final broad gate: lint, types, and all non-integration owner tests.",
        mutates_checkout=False,
        scope="lint + typecheck + math/catalog/dispatch/CLI/tooling tests",
        ci_relationship="subset of required PR jobs, not PR-equivalent",
        cost_class="bounded",
    ),
    CommandContract(
        name="check-all",
        help="Escalation: reproduce all ordinary Python CI lanes locally.",
        mutates_checkout=False,
        scope="lint + typecheck + all ordinary semantic owners",
        ci_relationship="local equivalent of the python matrix, not all CI jobs",
        cost_class="broad",
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
