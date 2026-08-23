"""Bounded source-indexed SMT unsatisfiable cores."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, ValidationError, model_validator

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.logic._operations import (
    SmtSolveRequest,
    _tokenize_smtlib,
    _top_level_smtlib_commands,
)

_MAX_CORE_ASSERTIONS = 512
_MAX_CORE_TOKENS = 32_768
_MAX_CORE_NESTING_DEPTH = 256
_MAX_CORE_NUMERAL_DIGITS = 256
_MAX_CORE_AST_NODES = 4_096
_MAX_CORE_RLIMIT = 10_000_000
_MAX_CORE_MEMORY_MB = 512
_NUMERAL = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|#x[0-9a-fA-F]+|#b[01]+)")


class SmtUnsatCoreRequest(SmtSolveRequest):
    """One bounded SMT-LIB query whose top-level assertions are indexed from zero.

    The source uses the SMT-LIB 2 command and term grammar admitted by ``smt.solve``.
    Its inherited 128,000-byte ASCII envelope is further limited to 32,768 tokens,
    nesting depth 256, 512 top-level ``assert`` commands, 4,096 distinct parsed AST
    nodes, and 256 digits per numeral. Assertion source order is the public
    zero-based index. Parsing constructs Z3 syntax trees only; it does not evaluate
    the source as Python or as a host command. Execution performs one tracked check
    and at most one direct result-validation replay, each under the supplied timeout,
    rlimit, and memory bound.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "logic": "QF_LIA",
                    "smtlib": (
                        "(set-logic QF_LIA)\n"
                        "(declare-const x Int)\n"
                        "(assert (>= x 1))\n"
                        "(assert (<= x 0))\n"
                        "(check-sat)"
                    ),
                    "timeout_ms": 1_000,
                    "rlimit": 100_000,
                    "max_memory_mb": 128,
                }
            ]
        }
    )

    rlimit: StrictInt = Field(
        default=100_000,
        ge=1,
        le=_MAX_CORE_RLIMIT,
        description="Z3 deterministic resource-unit limit for each solver check.",
    )
    max_memory_mb: StrictInt = Field(
        default=128,
        ge=16,
        le=_MAX_CORE_MEMORY_MB,
        description="Z3 memory limit in megabytes for each solver check.",
    )

    @property
    def assertion_count(self) -> int:
        return sum(
            command[:1] == ("assert",)
            for command in _top_level_smtlib_commands(self.smtlib)
        )

    @model_validator(mode="after")
    def require_bounded_indexed_assertions(self) -> Self:
        tokens = _tokenize_smtlib(self.smtlib)
        if len(tokens) > _MAX_CORE_TOKENS:
            raise ValueError(
                f"SMT core source may contain at most {_MAX_CORE_TOKENS} tokens"
            )
        _validate_nesting(tokens)
        _validate_numerals(tokens)
        if self.assertion_count > _MAX_CORE_ASSERTIONS:
            raise ValueError(
                f"SMT core source may contain at most {_MAX_CORE_ASSERTIONS} source assertions"
            )
        assertions = _parse_assertions(self.smtlib)
        if len(assertions) != self.assertion_count:
            raise ValueError(
                "SMT-LIB parser output must preserve one term per source assertion"
            )
        node_count, _boolean_constants = _ast_facts(assertions)
        if node_count > _MAX_CORE_AST_NODES:
            raise ValueError(
                f"SMT core source may contain at most {_MAX_CORE_AST_NODES} distinct AST nodes"
            )
        return self


class SmtUnsatCoreResult(StrictModel):
    """A source-bound satisfiability outcome and replayable UNSAT core."""

    source: SmtUnsatCoreRequest
    outcome: Literal["SAT", "UNSAT", "UNKNOWN"]
    core_indices: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=_MAX_CORE_ASSERTIONS,
        description=(
            "Strictly increasing zero-based source-assertion indices. Present only "
            "for UNSAT, when the selected assertions independently replay as UNSAT."
        ),
    )
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_outcome_to_source(self) -> Self:
        if self.core_indices != tuple(sorted(set(self.core_indices))):
            raise ValueError("core indices must be distinct and strictly increasing")
        if any(
            index < 0 or index >= self.source.assertion_count
            for index in self.core_indices
        ):
            raise ValueError("core index does not refer to a source assertion")

        if self.outcome == "UNSAT":
            if not self.core_indices:
                raise ValueError(
                    "UNSAT requires a nonempty core because the source has no untracked assertions"
                )
            replay, _detail = _replay_source(self.source, self.core_indices)
            if replay != "UNSAT":
                raise ValueError(
                    "selected source assertions must independently replay as UNSAT"
                )
        elif self.outcome == "SAT":
            if self.core_indices:
                raise ValueError("SAT cannot carry core indices")
            replay, _detail = _replay_source(self.source, None)
            if replay != "SAT":
                raise ValueError(
                    "complete source assertions must independently replay as SAT"
                )
        else:
            if self.core_indices:
                raise ValueError("UNKNOWN cannot carry core indices")
            if not self.detail:
                raise ValueError("UNKNOWN must explain the bounded execution outcome")

        if self.outcome != "UNKNOWN" and self.detail is not None:
            raise ValueError("only UNKNOWN may carry execution detail")
        return self


def _validate_nesting(tokens: tuple[str, ...]) -> None:
    depth = 0
    for token in tokens:
        if token == "(":
            depth += 1
            if depth > _MAX_CORE_NESTING_DEPTH:
                raise ValueError(
                    f"SMT core source nesting may not exceed {_MAX_CORE_NESTING_DEPTH}"
                )
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("SMT core source parentheses must be balanced")
    if depth:
        raise ValueError("SMT core source parentheses must be balanced")


def _validate_numerals(tokens: tuple[str, ...]) -> None:
    for token in tokens:
        if _NUMERAL.fullmatch(token) is None:
            continue
        digits = (
            len(token[2:])
            if token.startswith(("#x", "#b"))
            else sum(character.isdigit() for character in token)
        )
        if digits > _MAX_CORE_NUMERAL_DIGITS:
            raise ValueError(
                f"SMT numeral may contain at most {_MAX_CORE_NUMERAL_DIGITS} digits"
            )


def _parse_assertions(smtlib: str, *, context: Any | None = None) -> tuple[Any, ...]:
    import z3  # type: ignore[import-untyped]

    ctx = context if context is not None else z3.Context()
    try:
        return tuple(z3.parse_smt2_string(smtlib, ctx=ctx))
    except z3.Z3Exception as exc:
        raise ValueError("SMT core source could not be parsed as SMT-LIB") from exc


def _ast_facts(assertions: tuple[Any, ...]) -> tuple[int, frozenset[int]]:
    import z3

    seen: set[int] = set()
    boolean_constants: set[int] = set()
    stack = list(assertions)
    while stack:
        expression = stack.pop()
        expression_id = expression.get_id()
        if expression_id in seen:
            continue
        seen.add(expression_id)
        if z3.is_const(expression) and z3.is_bool(expression):
            boolean_constants.add(expression_id)
        if len(seen) > _MAX_CORE_AST_NODES:
            return len(seen), frozenset(boolean_constants)
        stack.extend(expression.children())
    return len(seen), frozenset(boolean_constants)


def _tracking_literals(assertions: tuple[Any, ...], *, context: Any) -> tuple[Any, ...]:
    import z3

    _node_count, used_boolean_constants = _ast_facts(assertions)
    trackers: list[Any] = []
    for index in range(len(assertions)):
        name = f"jacobian_unsat_core_{index}"
        tracker = z3.Bool(name, ctx=context)
        while tracker.get_id() in used_boolean_constants:
            name += "_"
            tracker = z3.Bool(name, ctx=context)
        trackers.append(tracker)
    return tuple(trackers)


def _configured_solver(source: SmtUnsatCoreRequest, *, context: Any) -> Any:
    import z3

    solver = z3.SolverFor(source.logic.value, ctx=context)
    solver.set(
        timeout=source.timeout_ms,
        rlimit=source.rlimit,
        max_memory=source.max_memory_mb,
        random_seed=0,
        unsat_core=True,
    )
    return solver


def _bounded_outcome(
    solver: Any,
) -> tuple[Literal["SAT", "UNSAT", "UNKNOWN"], str | None]:
    import z3

    outcome = solver.check()
    if outcome == z3.sat:
        return "SAT", None
    if outcome == z3.unsat:
        return "UNSAT", None
    detail = solver.reason_unknown().strip() or "Z3 returned UNKNOWN."
    return "UNKNOWN", detail[:1_024]


def _extract_source_core(
    source: SmtUnsatCoreRequest,
) -> tuple[Literal["SAT", "UNSAT", "UNKNOWN"], tuple[int, ...], str | None]:
    import z3

    context = z3.Context()
    try:
        assertions = _parse_assertions(source.smtlib, context=context)
        solver = _configured_solver(source, context=context)
        trackers = _tracking_literals(assertions, context=context)
        for assertion, tracker in zip(assertions, trackers, strict=True):
            solver.assert_and_track(assertion, tracker)
        outcome, detail = _bounded_outcome(solver)
    except (ValueError, z3.Z3Exception):
        return "UNKNOWN", (), "Z3 could not complete the bounded source check."

    if outcome != "UNSAT":
        return outcome, (), detail

    core_ids = {literal.get_id() for literal in solver.unsat_core()}
    tracker_ids = {tracker.get_id() for tracker in trackers}
    if not core_ids or not core_ids <= tracker_ids:
        return "UNKNOWN", (), "Z3 returned an unusable UNSAT core."
    core_indices = tuple(
        index for index, tracker in enumerate(trackers) if tracker.get_id() in core_ids
    )
    return "UNSAT", core_indices, None


def _replay_source(
    source: SmtUnsatCoreRequest,
    selected_indices: tuple[int, ...] | None,
) -> tuple[Literal["SAT", "UNSAT", "UNKNOWN"], str | None]:
    import z3

    context = z3.Context()
    try:
        assertions = _parse_assertions(source.smtlib, context=context)
        selected = (
            tuple(range(len(assertions)))
            if selected_indices is None
            else selected_indices
        )
        solver = _configured_solver(source, context=context)
        solver.add(*(assertions[index] for index in selected))
        return _bounded_outcome(solver)
    except (ValueError, z3.Z3Exception):
        return "UNKNOWN", "Z3 could not complete the bounded source replay."


def compute_smt_unsat_core(request: SmtUnsatCoreRequest) -> SmtUnsatCoreResult:
    """Return a bounded core of source-assertion indices when Z3 proves UNSAT."""

    outcome, core_indices, detail = _extract_source_core(request)
    try:
        return SmtUnsatCoreResult(
            source=request,
            outcome=outcome,
            core_indices=core_indices,
            detail=detail,
        )
    except ValidationError:
        return SmtUnsatCoreResult(
            source=request,
            outcome="UNKNOWN",
            detail="The bounded result-validation replay did not establish the conclusion.",
        )


SMT_UNSAT_CORE_OPERATION = MathTool(
    operation_id="smt.unsat_core",
    version="1",
    title="Extract a bounded indexed SMT UNSAT core",
    description=(
        "Return SAT, UNKNOWN, or source-order indices of assertions whose exact "
        "subsystem replays as UNSAT through the maintained Z3 Python binding."
    ),
    request_type=SmtUnsatCoreRequest,
    result_type=SmtUnsatCoreResult,
    run=compute_smt_unsat_core,
    tags=("smt", "unsat", "core", "constraints", "z3"),
    examples=(
        example(
            "contradictory_integer_bounds",
            (
                "Extract an indexed core from contradictory integer bounds; the "
                "SMT-LIB source must end in one check-sat and uses zero-based assertion indices."
            ),
            {
                "logic": "QF_LIA",
                "smtlib": (
                    "(set-logic QF_LIA)\n"
                    "(declare-const x Int)\n"
                    "(assert (>= x 1))\n"
                    "(assert (<= x 0))\n"
                    "(assert (<= x 10))\n"
                    "(check-sat)"
                ),
            },
        ),
    ),
)

__all__ = [
    "SMT_UNSAT_CORE_OPERATION",
    "SmtUnsatCoreRequest",
    "SmtUnsatCoreResult",
    "compute_smt_unsat_core",
]
