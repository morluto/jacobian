"""Bounded source-indexed SMT unsatisfiable cores."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, ValidationError, model_validator

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.logic._operations import (
    SmtLogic,
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
_NUMERAL = re.compile(r"(?:-?[0-9]+(?:\.[0-9]+)?|#x[0-9a-fA-F]+|#b[01]+)")


class SmtUnsatCoreRequest(SmtSolveRequest):
    """One bounded SMT-LIB query whose top-level assertions are indexed from zero.

    The source uses the SMT-LIB 2 command and term grammar admitted by ``smt.solve``.
    Terms must stay inside the selected quantifier-free fragment: QF_UF admits
    Boolean-sorted constants and uninterpreted functions, while QF_LIA and QF_LRA
    admit pure linear integer and real arithmetic respectively, together with
    Boolean structure.
    Its inherited 128,000-byte ASCII envelope is further limited to 32,768 tokens,
    nesting depth 256, 512 top-level ``assert`` commands, 4,096 distinct parsed AST
    nodes, 256 digits per numeral, and 256 digits in the numerator or denominator
    of any closed arithmetic coefficient, including every scalar coefficient and
    translated constant obtained by flattening nested products through
    coefficient-preserving wrappers (negation, additions and subtractions whose
    remaining operands are closed, and exact division by closed constants).
    Assertion source order is the public zero-based index. Parsing constructs Z3
    syntax trees only; it does not evaluate the source as Python or as a host
    command. Execution performs one tracked check and at most one direct
    tracked check and at most one direct result-validation replay, each under the
    supplied deterministic rlimit and timeout safety net.
    Z3 5.0 exposes only a process-global memory threshold, so this in-process
    operation does not claim a hard request-local byte limit; fixed source, AST,
    coefficient, and resource-unit bounds control its admitted memory envelope.
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
                }
            ]
        }
    )

    logic: SmtLogic = Field(
        description=(
            "Declared quantifier-free fragment. QF_UF is Boolean-sorted "
            "uninterpreted functions; QF_LIA and QF_LRA are pure linear integer "
            "and real arithmetic respectively, with Boolean structure."
        )
    )
    rlimit: StrictInt = Field(
        default=100_000,
        ge=1,
        le=_MAX_CORE_RLIMIT,
        description="Z3 deterministic resource-unit limit for each solver check.",
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
        parsed_error = _parsed_request_error(
            self.smtlib,
            assertion_count=self.assertion_count,
            logic=self.logic,
        )
        if parsed_error is not None:
            raise ValueError(parsed_error)
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
            "for UNSAT, when the selected assertions independently replay as UNSAT; "
            "the core is not promised to be minimum or inclusion-minimal."
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

    try:
        ctx = context if context is not None else z3.Context()
        return tuple(z3.parse_smt2_string(smtlib, ctx=ctx))
    except z3.Z3Exception as exc:
        raise ValueError("SMT core source could not be parsed as SMT-LIB") from exc


def _parsed_request_error(
    smtlib: str,
    *,
    assertion_count: int,
    logic: SmtLogic,
) -> str | None:
    """Inspect parsed terms without attaching Z3 objects to validation errors."""

    try:
        assertions = _parse_assertions(smtlib)
        if len(assertions) != assertion_count:
            return "SMT-LIB parser output must preserve one term per source assertion"
        node_count = _ast_node_count(assertions)
        if node_count > _MAX_CORE_AST_NODES:
            return (
                "SMT core source may contain at most "
                f"{_MAX_CORE_AST_NODES} distinct AST nodes"
            )
        _require_declared_logic(assertions, logic)
    except ValueError as exc:
        return str(exc)
    return None


def _ast_node_count(assertions: tuple[Any, ...]) -> int:
    seen: set[int] = set()
    stack = list(assertions)
    while stack:
        expression = stack.pop()
        expression_id = expression.get_id()
        if expression_id in seen:
            continue
        seen.add(expression_id)
        if len(seen) > _MAX_CORE_AST_NODES:
            return len(seen)
        stack.extend(expression.children())
    return len(seen)


def _require_declared_logic(
    assertions: tuple[Any, ...],
    logic: SmtLogic,
) -> None:
    """Reject parsed terms outside the request's advertised QF fragment."""

    if not assertions:
        return

    import z3

    try:
        if logic is SmtLogic.QF_UF:
            if not _is_boolean_uninterpreted_fragment(assertions):
                raise ValueError(
                    "SMT core terms must belong to the declared QF_UF fragment"
                )
            return

        classified_assertions = _normalize_closed_coefficients(assertions)
        goal = z3.Goal(ctx=assertions[0].ctx)
        goal.add(*classified_assertions)
        probe_name = "is-lia" if logic is SmtLogic.QF_LIA else "is-lra"
        has_quantifiers = float(
            z3.Probe("has-quantifiers", ctx=assertions[0].ctx)(goal)
        )
        belongs_to_fragment = float(z3.Probe(probe_name, ctx=assertions[0].ctx)(goal))
        if has_quantifiers != 0.0 or belongs_to_fragment != 1.0:
            raise ValueError(
                f"SMT core terms must belong to the declared {logic.value} fragment"
            )
    except z3.Z3Exception as exc:
        raise ValueError(
            f"SMT core terms could not be classified as {logic.value}"
        ) from exc


def _normalize_closed_coefficients(assertions: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize only bounded exact arithmetic subterms with no variables."""

    normalized_by_id: dict[int, Any] = {}
    closed_value_by_id: dict[int, Fraction | None] = {}
    scalar_by_id: dict[int, Fraction | None] = {}
    offset_by_id: dict[int, Fraction] = {}
    residual_by_id: dict[int, Any] = {}
    return tuple(
        _normalize_closed_expression(
            assertion,
            normalized_by_id=normalized_by_id,
            closed_value_by_id=closed_value_by_id,
            scalar_by_id=scalar_by_id,
            offset_by_id=offset_by_id,
            residual_by_id=residual_by_id,
        )
        for assertion in assertions
    )


def _fold_scaled_product(
    candidate: Any,
    children: tuple[Any, ...],
    normalized_children: tuple[Any, ...],
    child_values: tuple[Fraction | None, ...],
    *,
    scalar_by_id: dict[int, Fraction | None],
    offset_by_id: dict[int, Fraction],
    residual_by_id: dict[int, Any],
) -> tuple[Any, Fraction | None, Any | None, Fraction]:
    """Flatten one product with a single variable-carrying affine child."""

    import z3

    if candidate.decl().kind() != z3.Z3_OP_MUL:
        return candidate, None, None, Fraction(0)
    dependent_children = tuple(
        normalized_child
        for normalized_child, value in zip(
            normalized_children, child_values, strict=True
        )
        if value is None
    )
    closed_factors = tuple(value for value in child_values if value is not None)
    dependent_scalar: Fraction | None = None
    dependent_offset = Fraction(0)
    for child, child_value in zip(children, child_values, strict=True):
        if child_value is None:
            dependent_scalar = scalar_by_id[child.get_id()]
            if dependent_scalar is not None:
                dependent_offset = offset_by_id[child.get_id()]
            break
    if len(dependent_children) != 1 or not closed_factors:
        return candidate, None, None, Fraction(0)
    factors = (
        (*closed_factors, dependent_scalar)
        if dependent_scalar is not None
        else closed_factors
    )
    coefficient = _bounded_fraction_product(factors)
    scaled_offset = Fraction(0)
    if dependent_offset:
        scaled_offset = _bounded_fraction_product((*closed_factors, dependent_offset))
    dependent = dependent_children[0]
    while (residual := residual_by_id.get(dependent.get_id())) is not None:
        dependent = residual
    folded_term = candidate.decl()(
        _exact_arithmetic_value(coefficient, template=candidate),
        dependent,
    )
    folded_offset = Fraction(0)
    folded_candidate: Any = folded_term
    if dependent_offset:
        folded_candidate = folded_term + _exact_arithmetic_value(
            scaled_offset, template=folded_term
        )
        folded_offset = scaled_offset
    return folded_candidate, coefficient, dependent, folded_offset


def _normalize_closed_expression(
    expression: Any,
    *,
    normalized_by_id: dict[int, Any],
    closed_value_by_id: dict[int, Fraction | None],
    scalar_by_id: dict[int, Fraction | None],
    offset_by_id: dict[int, Fraction],
    residual_by_id: dict[int, Any],
) -> Any:
    """Normalize one expression without retaining a recursive closure over its AST."""

    import z3

    expression_id = expression.get_id()
    if expression_id in normalized_by_id:
        return normalized_by_id[expression_id]
    if z3.is_int_value(expression):
        value = Fraction(expression.as_long())
        normalized_by_id[expression_id] = expression
        closed_value_by_id[expression_id] = value
        scalar_by_id[expression_id] = value
        offset_by_id[expression_id] = Fraction(0)
        return expression
    if z3.is_rational_value(expression):
        value = expression.as_fraction()
        normalized_by_id[expression_id] = expression
        closed_value_by_id[expression_id] = value
        scalar_by_id[expression_id] = value
        offset_by_id[expression_id] = Fraction(0)
        return expression
    if not z3.is_app(expression):
        normalized_by_id[expression_id] = expression
        closed_value_by_id[expression_id] = None
        scalar_by_id[expression_id] = None
        return expression

    children = expression.children()
    normalized_children = tuple(
        _normalize_closed_expression(
            child,
            normalized_by_id=normalized_by_id,
            closed_value_by_id=closed_value_by_id,
            scalar_by_id=scalar_by_id,
            offset_by_id=offset_by_id,
            residual_by_id=residual_by_id,
        )
        for child in children
    )
    if any(
        not original.eq(normalized)
        for original, normalized in zip(children, normalized_children, strict=True)
    ):
        candidate = expression.decl()(*normalized_children)
    else:
        candidate = expression

    child_values = tuple(closed_value_by_id[child.get_id()] for child in children)
    candidate, folded_coefficient, folded_dependent, folded_offset = (
        _fold_scaled_product(
            candidate,
            children,
            normalized_children,
            child_values,
            scalar_by_id=scalar_by_id,
            offset_by_id=offset_by_id,
            residual_by_id=residual_by_id,
        )
    )

    closed_value = _evaluate_closed_arithmetic(candidate.decl().kind(), child_values)
    if closed_value is not None:
        candidate = _exact_arithmetic_value(closed_value, template=candidate)

    scalar = closed_value if closed_value is not None else folded_coefficient
    offset: Fraction | None = Fraction(0) if closed_value is not None else folded_offset
    if scalar is None:
        scalar, offset = _wrapped_linear_scalar(
            candidate,
            children,
            child_values,
            scalar_by_id=scalar_by_id,
            offset_by_id=offset_by_id,
            residual_by_id=residual_by_id,
        )

    normalized_by_id[expression_id] = candidate
    closed_value_by_id[expression_id] = closed_value
    scalar_by_id[expression_id] = scalar
    if offset is not None:
        offset_by_id[expression_id] = offset
    if folded_dependent is not None:
        residual_by_id[expression_id] = folded_dependent
        residual_by_id[candidate.get_id()] = folded_dependent
        offset_by_id[candidate.get_id()] = folded_offset
    return candidate


def _chased_residual(dependent: Any, residual_by_id: dict[int, Any]) -> Any:
    core = dependent
    while (residual := residual_by_id.get(core.get_id())) is not None:
        core = residual
    return core


def _folded_offset(offset_by_id: dict[int, Fraction], child: Any) -> Fraction:
    """Return the recorded translated constant of a scalar-bearing child."""

    return offset_by_id.get(child.get_id(), Fraction(0))


def _wrapped_linear_scalar(
    candidate: Any,
    children: tuple[Any, ...],
    child_values: tuple[Fraction | None, ...],
    *,
    scalar_by_id: dict[int, Fraction | None],
    offset_by_id: dict[int, Fraction],
    residual_by_id: dict[int, Any],
) -> tuple[Fraction | None, Fraction | None]:
    """Carry one child's folded linear scalar through coefficient-preserving wrappers."""

    import z3

    kind = candidate.decl().kind()
    if kind == z3.Z3_OP_ADD and len(children) >= 2:
        signs: tuple[int, ...] = (1,) * len(children)
    elif kind == z3.Z3_OP_SUB and len(children) >= 2:
        signs = (1,) + (-1,) * (len(children) - 1)
    else:
        signs = ()
    if len(children) == 1 and kind in (z3.Z3_OP_UMINUS, z3.Z3_OP_ADD):
        child_scalar = scalar_by_id[children[0].get_id()]
        if child_scalar is None:
            return None, None
        scalar = -child_scalar if kind == z3.Z3_OP_UMINUS else child_scalar
        child_offset = _folded_offset(offset_by_id, children[0])
        offset = -child_offset if kind == z3.Z3_OP_UMINUS else child_offset
        residual = _chased_residual(children[0], residual_by_id)
    elif signs:
        open_indices = [
            index for index, value in enumerate(child_values) if value is None
        ]
        if len(open_indices) != 1:
            return None, None
        open_index = open_indices[0]
        child_scalar = scalar_by_id[children[open_index].get_id()]
        if child_scalar is None:
            return None, None
        sign = signs[open_index]
        scalar = sign * child_scalar
        total: Fraction = sign * _folded_offset(offset_by_id, children[open_index])
        for index, value in enumerate(child_values):
            if value is not None:
                total += signs[index] * value
        _require_bounded_normalized_coefficient(total)
        offset = total
        residual = _chased_residual(children[open_index], residual_by_id)
    elif (
        len(children) == 2
        and kind == z3.Z3_OP_DIV
        and child_values[0] is None
        and child_values[1] is not None
        and child_values[1] != 0
    ):
        dividend_scalar = scalar_by_id[children[0].get_id()]
        if dividend_scalar is None:
            return None, None
        divisor = child_values[1]
        scalar = dividend_scalar / divisor
        _require_bounded_normalized_coefficient(scalar)
        offset = _folded_offset(offset_by_id, children[0]) / divisor
        _require_bounded_normalized_coefficient(offset)
        residual = _chased_residual(children[0], residual_by_id)
    else:
        return None, None
    residual_by_id[candidate.get_id()] = residual
    return scalar, offset


def _evaluate_closed_arithmetic(
    kind: int,
    values: tuple[Fraction | None, ...],
) -> Fraction | None:
    import z3

    if not values or any(value is None for value in values):
        return None
    exact_values = tuple(value for value in values if value is not None)
    if kind == z3.Z3_OP_ADD:
        total = Fraction()
        for value in exact_values:
            total += value
            _require_bounded_normalized_coefficient(total)
        return total
    if kind == z3.Z3_OP_SUB:
        difference = exact_values[0]
        for value in exact_values[1:]:
            difference -= value
            _require_bounded_normalized_coefficient(difference)
        return difference
    if kind == z3.Z3_OP_UMINUS:
        return -exact_values[0]
    if kind == z3.Z3_OP_MUL:
        return _bounded_fraction_product(exact_values)
    if kind == z3.Z3_OP_DIV and len(exact_values) == 2 and exact_values[1]:
        return exact_values[0] / exact_values[1]
    return None


def _bounded_fraction_product(values: tuple[Fraction, ...]) -> Fraction:
    product = Fraction(1)
    for value in values:
        product *= value
        _require_bounded_normalized_coefficient(product)
    return product


def _exact_arithmetic_value(value: Fraction, *, template: Any) -> Any:
    import z3

    _require_bounded_normalized_coefficient(value)
    if template.sort().kind() == z3.Z3_INT_SORT:
        if value.denominator != 1:
            raise ValueError("normalized integer coefficient must remain integral")
        return z3.IntVal(value.numerator, ctx=template.ctx)
    if template.sort().kind() == z3.Z3_REAL_SORT:
        return z3.RealVal(
            f"{value.numerator}/{value.denominator}",
            ctx=template.ctx,
        )
    raise ValueError("normalized coefficient must have an arithmetic sort")


def _require_bounded_normalized_coefficient(value: Fraction) -> None:
    if any(
        len(str(abs(component))) > _MAX_CORE_NUMERAL_DIGITS
        for component in (value.numerator, value.denominator)
    ):
        raise ValueError(
            "normalized SMT coefficient numerator and denominator may contain at "
            f"most {_MAX_CORE_NUMERAL_DIGITS} digits"
        )


def _is_boolean_uninterpreted_fragment(assertions: tuple[Any, ...]) -> bool:
    import z3

    allowed_kinds = frozenset(
        {
            z3.Z3_OP_TRUE,
            z3.Z3_OP_FALSE,
            z3.Z3_OP_EQ,
            z3.Z3_OP_DISTINCT,
            z3.Z3_OP_ITE,
            z3.Z3_OP_AND,
            z3.Z3_OP_OR,
            z3.Z3_OP_XOR,
            z3.Z3_OP_NOT,
            z3.Z3_OP_IMPLIES,
            z3.Z3_OP_UNINTERPRETED,
        }
    )
    seen: set[int] = set()
    stack = list(assertions)
    while stack:
        expression = stack.pop()
        expression_id = expression.get_id()
        if expression_id in seen:
            continue
        seen.add(expression_id)
        if (
            not z3.is_app(expression)
            or not z3.is_bool(expression)
            or expression.decl().kind() not in allowed_kinds
        ):
            return False
        stack.extend(expression.children())
    return True


def _tracking_literals(assertion_count: int, *, context: Any) -> tuple[Any, ...]:
    import z3

    return tuple(
        z3.FreshBool(f"jacobian_unsat_core_{index}", ctx=context)
        for index in range(assertion_count)
    )


def _configured_solver(source: SmtUnsatCoreRequest, *, context: Any) -> Any:
    import z3

    solver = z3.SolverFor(source.logic.value, ctx=context)
    solver.set(
        timeout=source.timeout_ms,
        rlimit=source.rlimit,
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

    try:
        context = z3.Context()
        assertions = _parse_assertions(source.smtlib, context=context)
        solver = _configured_solver(source, context=context)
        trackers = _tracking_literals(len(assertions), context=context)
        for assertion, tracker in zip(assertions, trackers, strict=True):
            solver.assert_and_track(assertion, tracker)
        outcome, detail = _bounded_outcome(solver)
        if outcome != "UNSAT":
            return outcome, (), detail

        core_ids = {literal.get_id() for literal in solver.unsat_core()}
        tracker_ids = {tracker.get_id() for tracker in trackers}
        if not core_ids or not core_ids <= tracker_ids:
            return "UNKNOWN", (), "Z3 returned an unusable UNSAT core."
        core_indices = tuple(
            index
            for index, tracker in enumerate(trackers)
            if tracker.get_id() in core_ids
        )
        return "UNSAT", core_indices, None
    except (ValueError, z3.Z3Exception):
        return "UNKNOWN", (), "Z3 could not complete the bounded source check."


def _replay_source(
    source: SmtUnsatCoreRequest,
    selected_indices: tuple[int, ...] | None,
) -> tuple[Literal["SAT", "UNSAT", "UNKNOWN"], str | None]:
    import z3

    try:
        context = z3.Context()
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
        "Return SAT, UNKNOWN, or a deterministic backend-selected set of source-order "
        "assertion indices whose exact subsystem replays as UNSAT through the "
        "maintained Z3 Python binding. The core need not be minimal."
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
