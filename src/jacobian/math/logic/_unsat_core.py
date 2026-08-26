"""Bounded source-indexed SMT unsatisfiable cores."""

from __future__ import annotations

import re
from fractions import Fraction
from math import lcm
from typing import Any, Literal, NamedTuple, Self

from pydantic import ConfigDict, Field, StrictInt, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.logic._smt import (
    SmtLogic,
    SmtSolveRequest,
    _is_smtlib_source_diagnostic,
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
_AffineTerms = tuple[tuple[Fraction, Any], ...]
_AffineForm = tuple[_AffineTerms, Fraction]


def _logic_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("logic.unsat_core_contract", message)


class _CoefficientEnvelope(NamedTuple):
    """Numeric height plus reachable coefficient digit budgets for one expression.

    Every coefficient reachable while reading the expression linearly has
    magnitude at most ``height``, numerator digit count at most
    ``numerator_digits``, and denominator digit count at most
    ``denominator_digits``. ``pairs`` is the exact closure of reachable
    ``(numerator bound, denominator)`` pairs, or a conservative closure that
    also retains unmatched child coefficients, or ``None`` when only the
    marginal digit budgets are tracked. Sums and comparisons use it to keep
    shared denominators from compounding while still bounding merged
    numerators soundly.
    """

    height: Fraction
    numerator_digits: int
    denominator_digits: int
    pairs: tuple[tuple[int, int], ...] | None


_UNIT_COEFFICIENT_ENVELOPE = _CoefficientEnvelope(Fraction(1), 1, 1, ((1, 1),))
_MAX_ENVELOPE_PAIRS = 16


def _capped_pairs(
    pairs: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    if len(pairs) > _MAX_ENVELOPE_PAIRS:
        return None
    return tuple(sorted(pairs))


def _merged_pairs(
    *pair_lists: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    """Union reachable coefficient pairs without collapsing any of them."""

    merged: set[tuple[int, int]] = set()
    for pair_list in pair_lists:
        if pair_list is None:
            return None
        merged.update(pair_list)
    return _capped_pairs(merged)


def _pairs_common_denominator(
    pairs: tuple[tuple[int, int], ...] | None,
) -> int | None:
    if pairs is None:
        return None
    return lcm(*(denominator for _numerator, denominator in pairs))


def _negated_envelope(envelope: _CoefficientEnvelope) -> _CoefficientEnvelope:
    if envelope.pairs is None:
        return envelope
    return envelope._replace(
        pairs=tuple(
            (-numerator, denominator) for numerator, denominator in envelope.pairs
        )
    )


def _scaled_pairs(
    scalar: Fraction,
    pairs: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    if pairs is None:
        return None
    return _capped_pairs(
        {
            (
                (Fraction(numerator, denominator) * scalar).numerator,
                (Fraction(numerator, denominator) * scalar).denominator,
            )
            for numerator, denominator in pairs
        }
    )


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
    translated constant obtained by flattening nested products and signed affine
    sums through coefficient-preserving wrappers (negation, additions and
    subtractions over closed and variable-carrying operands, exact division
    by closed nonzero constants, integer division included when every divided
    coefficient remains exactly divisible, and arithmetic if-then-else
    selections, where scaling is validated against every branch's reachable
    coefficient digits, so a numerically smaller branch cannot hide reciprocal
    growth beyond the digit budget).
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
        """Reject out-of-envelope source before any backend work.

        These core-specific bounds run inside the most-derived validator, so
        a source that fits the broader ``smt.solve`` envelope but violates a
        core limit is rejected lexically, before this class's single backend
        parse below.
        """

        tokens = _tokenize_smtlib(self.smtlib)
        if len(tokens) > _MAX_CORE_TOKENS:
            raise _logic_error(
                f"SMT core source may contain at most {_MAX_CORE_TOKENS} tokens"
            )
        _validate_nesting(tokens)
        _validate_numerals(tokens)
        if self.assertion_count > _MAX_CORE_ASSERTIONS:
            raise _logic_error(
                f"SMT core source may contain at most {_MAX_CORE_ASSERTIONS} source assertions"
            )
        parsed_error = _parsed_request_error(
            self.smtlib,
            assertion_count=self.assertion_count,
            logic=self.logic,
        )
        if parsed_error is not None:
            raise _logic_error(parsed_error)
        return self

    def _complete_backend_admission(self) -> None:
        """Skip the inherited base-class parse.

        ``require_bounded_indexed_assertions`` already parses through the
        backend only after every core-specific structural bound has passed,
        so sources outside this operation's envelope never reach Z3.
        """


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
            raise _logic_error("core indices must be distinct and strictly increasing")
        if any(
            index < 0 or index >= self.source.assertion_count
            for index in self.core_indices
        ):
            raise _logic_error("core index does not refer to a source assertion")

        if self.outcome == "UNSAT":
            if not self.core_indices:
                raise _logic_error(
                    "UNSAT requires a nonempty core because the source has no untracked assertions"
                )
            replay, _detail = _replay_source(self.source, self.core_indices)
            if replay != "UNSAT":
                raise _logic_error(
                    "selected source assertions must independently replay as UNSAT"
                )
        elif self.outcome == "SAT":
            if self.core_indices:
                raise _logic_error("SAT cannot carry core indices")
            replay, _detail = _replay_source(self.source, None)
            if replay != "SAT":
                raise _logic_error(
                    "complete source assertions must independently replay as SAT"
                )
        else:
            if self.core_indices:
                raise _logic_error("UNKNOWN cannot carry core indices")
            if not self.detail:
                raise _logic_error("UNKNOWN must explain the bounded execution outcome")

        if self.outcome != "UNKNOWN" and self.detail is not None:
            raise _logic_error("only UNKNOWN may carry execution detail")
        return self


def _validate_nesting(tokens: tuple[str, ...]) -> None:
    depth = 0
    for token in tokens:
        if token == "(":
            depth += 1
            if depth > _MAX_CORE_NESTING_DEPTH:
                raise _logic_error(
                    f"SMT core source nesting may not exceed {_MAX_CORE_NESTING_DEPTH}"
                )
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise _logic_error("SMT core source parentheses must be balanced")
    if depth:
        raise _logic_error("SMT core source parentheses must be balanced")


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
            raise _logic_error(
                f"SMT numeral may contain at most {_MAX_CORE_NUMERAL_DIGITS} digits"
            )


def _parse_assertions(smtlib: str, *, context: Any | None = None) -> tuple[Any, ...]:
    import z3  # type: ignore[import-untyped]

    try:
        ctx = context if context is not None else z3.Context()
        return tuple(z3.parse_smt2_string(smtlib, ctx=ctx))
    except z3.Z3Exception as exc:
        if not _is_smtlib_source_diagnostic(exc):
            raise
        raise _logic_error("SMT core source could not be parsed as SMT-LIB") from exc


def _parsed_request_error(
    smtlib: str,
    *,
    assertion_count: int,
    logic: SmtLogic,
) -> str | None:
    """Inspect parsed terms without attaching Z3 objects to validation errors."""

    try:
        import z3
    except (ImportError, OSError):
        # Backend absence provides no information about caller-controlled
        # source. Leave this accepted request for execution to report as its
        # typed UNKNOWN outcome.
        return None

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
    except (OSError, z3.Z3Exception):
        # A deferred backend failure carries no evidence about this source;
        # execution reports it through the typed UNKNOWN outcome.
        return None
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
                raise _logic_error(
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
            raise _logic_error(
                f"SMT core terms must belong to the declared {logic.value} fragment"
            )
    except z3.Z3Exception as exc:
        raise _logic_error(
            f"SMT core terms could not be classified as {logic.value}"
        ) from exc


def _normalize_closed_coefficients(assertions: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize only bounded exact arithmetic subterms with no variables."""

    normalized_by_id: dict[int, Any] = {}
    closed_value_by_id: dict[int, Fraction | None] = {}
    form_by_id: dict[int, _AffineForm | None] = {}
    envelope_by_id: dict[int, _CoefficientEnvelope] = {}
    return tuple(
        _normalize_closed_expression(
            assertion,
            normalized_by_id=normalized_by_id,
            closed_value_by_id=closed_value_by_id,
            form_by_id=form_by_id,
            envelope_by_id=envelope_by_id,
        )
        for assertion in assertions
    )


def _affine_expression(
    terms: _AffineTerms,
    offset: Fraction,
    *,
    template: Any,
) -> Any:
    """Rebuild one bounded flat affine sum, or its constant when it has no terms."""

    if not terms:
        return _exact_arithmetic_value(offset, template=template)
    lead_coefficient, lead_core = terms[0]
    total: Any = (
        _exact_arithmetic_value(lead_coefficient, template=lead_core) * lead_core
    )
    for coefficient, core in terms[1:]:
        total = total + _exact_arithmetic_value(coefficient, template=core) * core
    if offset:
        total = total + _exact_arithmetic_value(offset, template=template)
    return total


def _fold_scaled_product(
    candidate: Any,
    children: tuple[Any, ...],
    child_values: tuple[Fraction | None, ...],
    *,
    form_by_id: dict[int, _AffineForm | None],
    envelope_by_id: dict[int, _CoefficientEnvelope],
) -> tuple[Any, _AffineForm | None, _CoefficientEnvelope | None]:
    """Flatten one product with a single variable-carrying affine child."""

    import z3

    if candidate.decl().kind() != z3.Z3_OP_MUL:
        return candidate, None, None
    open_indices = [index for index, value in enumerate(child_values) if value is None]
    if len(open_indices) != 1:
        return candidate, None, None
    dependent = children[open_indices[0]]
    coefficient = _bounded_fraction_product(
        tuple(value for value in child_values if value is not None)
    )
    scaled_envelope = None
    child_form = form_by_id.get(dependent.get_id())
    if child_form is None:
        scaled_envelope = _scaled_coefficient_envelope(
            coefficient,
            envelope_by_id.get(dependent.get_id(), _UNIT_COEFFICIENT_ENVELOPE),
        )
        _require_bounded_normalized_coefficient(scaled_envelope.height)
        child_form = (((Fraction(1), dependent),), Fraction(0))
    terms, offset = child_form
    scaled_terms: list[tuple[Fraction, Any]] = []
    for term_coefficient, core in terms:
        scaled = coefficient * term_coefficient
        _require_bounded_normalized_coefficient(scaled)
        scaled_terms.append((scaled, core))
    scaled_offset = Fraction(0)
    if offset:
        scaled_offset = coefficient * offset
        _require_bounded_normalized_coefficient(scaled_offset)
    folded_terms = tuple(scaled_terms)
    folded = _affine_expression(folded_terms, scaled_offset, template=candidate)
    return folded, (folded_terms, scaled_offset), scaled_envelope


def _normalize_closed_expression(
    expression: Any,
    *,
    normalized_by_id: dict[int, Any],
    closed_value_by_id: dict[int, Fraction | None],
    form_by_id: dict[int, _AffineForm | None],
    envelope_by_id: dict[int, _CoefficientEnvelope],
) -> Any:
    """Normalize one expression without retaining a recursive closure over its AST."""

    import z3

    expression_id = expression.get_id()
    if expression_id in normalized_by_id:
        return normalized_by_id[expression_id]
    if z3.is_int_value(expression) or z3.is_rational_value(expression):
        value = (
            Fraction(expression.as_long())
            if z3.is_int_value(expression)
            else expression.as_fraction()
        )
        normalized_by_id[expression_id] = expression
        closed_value_by_id[expression_id] = value
        form_by_id[expression_id] = ((), value)
        envelope_by_id[expression_id] = _closed_value_envelope(value)
        return expression
    if not z3.is_app(expression):
        normalized_by_id[expression_id] = expression
        closed_value_by_id[expression_id] = None
        form_by_id[expression_id] = None
        envelope_by_id[expression_id] = _UNIT_COEFFICIENT_ENVELOPE
        return expression
    if z3.is_arith(expression) and not expression.children():
        normalized_by_id[expression_id] = expression
        closed_value_by_id[expression_id] = None
        form_by_id[expression_id] = (((Fraction(1), expression),), Fraction(0))
        envelope_by_id[expression_id] = _UNIT_COEFFICIENT_ENVELOPE
        return expression

    children = expression.children()
    normalized_children = tuple(
        _normalize_closed_expression(
            child,
            normalized_by_id=normalized_by_id,
            closed_value_by_id=closed_value_by_id,
            form_by_id=form_by_id,
            envelope_by_id=envelope_by_id,
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
    candidate, form, scaled_envelope = _fold_scaled_product(
        candidate,
        children,
        child_values,
        form_by_id=form_by_id,
        envelope_by_id=envelope_by_id,
    )

    closed_value = _evaluate_closed_arithmetic(candidate.decl().kind(), child_values)
    if closed_value is not None:
        candidate = _exact_arithmetic_value(closed_value, template=candidate)
        form = ((), closed_value)
    elif form is None:
        form = _wrapped_linear_form(
            candidate, children, child_values, form_by_id=form_by_id
        )
    if closed_value is not None:
        envelope = _closed_value_envelope(closed_value)
    elif scaled_envelope is not None:
        envelope = scaled_envelope
    elif form is not None:
        envelope = _affine_form_envelope(
            form,
            height=_coefficient_height(
                candidate.decl().kind(),
                children,
                child_values,
                envelope_by_id=envelope_by_id,
                enforce_digit_budget=False,
            ).height,
        )
    else:
        envelope = _coefficient_height(
            candidate.decl().kind(),
            children,
            child_values,
            envelope_by_id=envelope_by_id,
            enforce_digit_budget=True,
        )

    normalized_by_id[expression_id] = candidate
    closed_value_by_id[expression_id] = closed_value
    form_by_id[expression_id] = form
    envelope_by_id[expression_id] = envelope
    if not candidate.eq(expression):
        form_by_id[candidate.get_id()] = form
        envelope_by_id[candidate.get_id()] = envelope
    return candidate


def _wrapped_linear_form(
    candidate: Any,
    children: tuple[Any, ...],
    child_values: tuple[Fraction | None, ...],
    *,
    form_by_id: dict[int, _AffineForm | None],
) -> _AffineForm | None:
    """Combine children's affine forms through coefficient-preserving wrappers."""

    import z3

    kind = candidate.decl().kind()
    if len(children) == 1 and kind == z3.Z3_OP_UMINUS:
        child_form = form_by_id.get(children[0].get_id())
        if child_form is None:
            return None
        terms, offset = child_form
        return (
            tuple((-coefficient, core) for coefficient, core in terms),
            -offset,
        )
    if kind == z3.Z3_OP_ADD and children:
        signs: tuple[int, ...] = (1,) * len(children)
    elif kind == z3.Z3_OP_SUB and len(children) >= 2:
        signs = (1,) + (-1,) * (len(children) - 1)
    else:
        signs = ()
    if signs:
        return _combined_signed_form(children, signs, form_by_id=form_by_id)
    if (
        len(children) == 2
        and kind in (z3.Z3_OP_DIV, z3.Z3_OP_IDIV)
        and child_values[0] is None
        and child_values[1] is not None
        and child_values[1] != 0
    ):
        dividend_form = form_by_id.get(children[0].get_id())
        if dividend_form is None:
            return None
        divisor = child_values[1]
        if kind == z3.Z3_OP_IDIV:
            return _exact_integer_division_form(dividend_form, divisor)
        terms, offset = dividend_form
        scaled_terms: list[tuple[Fraction, Any]] = []
        for coefficient, core in terms:
            scaled = coefficient / divisor
            _require_bounded_normalized_coefficient(scaled)
            scaled_terms.append((scaled, core))
        scaled_offset = offset / divisor
        _require_bounded_normalized_coefficient(scaled_offset)
        return (tuple(scaled_terms), scaled_offset)
    return None


def _coefficient_height(
    kind: int,
    children: tuple[Any, ...],
    child_values: tuple[Fraction | None, ...],
    *,
    envelope_by_id: dict[int, _CoefficientEnvelope],
    enforce_digit_budget: bool,
) -> _CoefficientEnvelope:
    """Bound every coefficient reachable in one expression's linear reading.

    Arithmetic if-then-else selects exactly one branch, so its envelope is the
    componentwise maximum of the branch envelopes: each branch keeps its own
    numerator and denominator digit budget, so a numerically smaller reciprocal
    branch cannot hide denominator growth from later scaling. Products multiply
    envelopes and signed sums merge them, matching every scalar the preserving
    wrappers can derive downstream. Divisions by closed constants and
    composed signed sums merge branch budgets into one coefficient, so they
    validate those budgets unless an exact flattened form already validates
    every coefficient of the node.
    """

    import z3

    child_envelopes = tuple(envelope_by_id[child.get_id()] for child in children)
    if kind == z3.Z3_OP_ITE and len(child_envelopes) == 3:
        left, right = child_envelopes[1], child_envelopes[2]
        return _CoefficientEnvelope(
            max(left.height, right.height),
            max(left.numerator_digits, right.numerator_digits),
            max(left.denominator_digits, right.denominator_digits),
            _merged_pairs(left.pairs, right.pairs),
        )
    if kind == z3.Z3_OP_UMINUS:
        return _negated_envelope(child_envelopes[0])
    if kind == z3.Z3_OP_MUL:
        return _multiplied_envelope(
            child_envelopes,
            validate_budget=enforce_digit_budget,
        )
    if (
        len(children) == 2
        and kind in (z3.Z3_OP_DIV, z3.Z3_OP_IDIV)
        and child_values[1] is not None
        and child_values[1] != 0
    ):
        return _divided_envelope(
            child_envelopes[0],
            child_values[1],
            exact=kind == z3.Z3_OP_DIV or abs(child_values[1]) == 1,
            validate_budget=enforce_digit_budget,
        )
    if kind in (z3.Z3_OP_ADD, z3.Z3_OP_SUB):
        return _signed_sum_envelope(
            child_envelopes,
            signs=(1,) * len(child_envelopes)
            if kind == z3.Z3_OP_ADD
            else (1,) + (-1,) * (len(child_envelopes) - 1),
            validate_budget=enforce_digit_budget,
        )
    if kind in (
        z3.Z3_OP_EQ,
        z3.Z3_OP_DISTINCT,
        z3.Z3_OP_LE,
        z3.Z3_OP_LT,
        z3.Z3_OP_GE,
        z3.Z3_OP_GT,
    ):
        return _compared_envelope(
            child_envelopes,
            validate_budget=enforce_digit_budget,
        )
    height = Fraction(0)
    for envelope in child_envelopes:
        height += envelope.height
        _require_bounded_normalized_coefficient(height)
    envelope = _maximal_digit_envelope(
        child_envelopes,
        height=height,
    )
    if enforce_digit_budget:
        _require_bounded_coefficient_digit_budget(
            envelope.numerator_digits,
            envelope.denominator_digits,
        )
    return envelope


def _multiplied_envelope(
    envelopes: tuple[_CoefficientEnvelope, ...],
    *,
    validate_budget: bool,
) -> _CoefficientEnvelope:
    """Scale one envelope by every remaining envelope, as products do."""

    numerator_digits = sum(envelope.numerator_digits for envelope in envelopes)
    denominator_digits = sum(envelope.denominator_digits for envelope in envelopes)
    pairs = _product_pairs(tuple(envelope.pairs for envelope in envelopes))
    common_denominator = _pairs_common_denominator(pairs)
    if common_denominator is not None:
        denominator_digits = min(denominator_digits, len(str(common_denominator)))
    if validate_budget:
        _require_bounded_coefficient_digit_budget(
            numerator_digits,
            denominator_digits,
        )
    return _CoefficientEnvelope(
        _bounded_fraction_product(tuple(envelope.height for envelope in envelopes)),
        numerator_digits,
        denominator_digits,
        pairs,
    )


def _product_pairs(
    pair_lists: tuple[tuple[tuple[int, int], ...] | None, ...],
) -> tuple[tuple[int, int], ...] | None:
    result: set[tuple[int, int]] = {(1, 1)}
    for pair_list in pair_lists:
        if pair_list is None:
            return None
        combined: set[tuple[int, int]] = set()
        for numerator, denominator in pair_list:
            left = Fraction(numerator, denominator)
            for shared_numerator, shared_denominator in result:
                right = Fraction(shared_numerator, shared_denominator)
                product = left * right
                combined.add((product.numerator, product.denominator))
        result = combined
        if len(result) > _MAX_ENVELOPE_PAIRS:
            return None
    return _capped_pairs(result)


def _divided_envelope(
    dividend: _CoefficientEnvelope,
    divisor: Fraction,
    *,
    exact: bool,
    validate_budget: bool,
) -> _CoefficientEnvelope:
    """Divide one envelope by a closed nonzero constant, SMT-LIB div included."""

    scaled = dividend.height / abs(divisor)
    numerator_digits = dividend.numerator_digits + len(str(divisor.denominator))
    denominator_digits = dividend.denominator_digits + len(str(abs(divisor.numerator)))
    if not exact:
        scaled += 1
        numerator_digits += 1
    pairs = _scaled_pairs(Fraction(1) / divisor, dividend.pairs) if exact else None
    common_denominator = _pairs_common_denominator(pairs)
    if common_denominator is not None:
        denominator_digits = min(denominator_digits, len(str(common_denominator)))
    _require_bounded_normalized_coefficient(scaled)
    if validate_budget:
        _require_bounded_coefficient_digit_budget(
            numerator_digits,
            denominator_digits,
        )
    return _CoefficientEnvelope(scaled, numerator_digits, denominator_digits, pairs)


def _signed_sum_envelope(
    envelopes: tuple[_CoefficientEnvelope, ...],
    *,
    signs: tuple[int, ...],
    validate_budget: bool,
) -> _CoefficientEnvelope:
    """Merge child readings the way the node's signed combination does."""

    height = Fraction(0)
    for envelope in envelopes:
        height += envelope.height
        _require_bounded_normalized_coefficient(height)
    denominator_digits = sum(envelope.denominator_digits for envelope in envelopes)
    widest_numerator = max(
        (
            envelope.numerator_digits + denominator_digits - envelope.denominator_digits
            for envelope in envelopes
        ),
        default=0,
    )
    numerator_digits = widest_numerator + len(str(len(envelopes)))
    pairs = None
    shared = _shared_denominator_of_pairs(envelopes)
    if shared is not None:
        lifted: list[set[Fraction]] = []
        for sign, envelope in zip(signs, envelopes, strict=True):
            lifted.append(
                {
                    sign * Fraction(numerator, denominator) * shared
                    for numerator, denominator in envelope.pairs or ()
                }
            )
        totals: set[Fraction] = {Fraction(0)}
        for values in lifted:
            totals = {total + value for total in totals for value in values}
            if len(totals) > _MAX_ENVELOPE_PAIRS:
                break
        else:
            reachable: set[Fraction] = set(totals)
            for values in lifted:
                reachable.update(values)
            pairs = _capped_pairs(
                {
                    ((value / shared).numerator, (value / shared).denominator)
                    for value in reachable
                }
            )
            common_denominator = _pairs_common_denominator(pairs)
            if common_denominator is not None:
                denominator_digits = min(
                    denominator_digits, len(str(common_denominator))
                )
            widest_merged = max(abs((value / shared).numerator) for value in reachable)
            numerator_digits = min(numerator_digits, len(str(widest_merged)))
    else:
        pairs = None
    if validate_budget:
        _require_bounded_coefficient_digit_budget(
            numerator_digits,
            denominator_digits,
        )
    return _CoefficientEnvelope(height, numerator_digits, denominator_digits, pairs)


def _shared_denominator_of_pairs(
    envelopes: tuple[_CoefficientEnvelope, ...],
) -> int | None:
    denominators: list[int] = []
    for envelope in envelopes:
        if envelope.pairs is None:
            return None
        denominators.extend(denominator for _numerator, denominator in envelope.pairs)
    return lcm(*denominators)


def _compared_envelope(
    envelopes: tuple[_CoefficientEnvelope, ...],
    *,
    validate_budget: bool,
) -> _CoefficientEnvelope:
    """Bound the difference an arithmetic comparison forms between its sides.

    The difference of two nonnegative quantities never exceeds the larger one,
    so each side is lifted to the shared denominator and only the widest lifted
    numerator bounds the merged coefficient.
    """

    height = Fraction(0)
    for envelope in envelopes:
        height += envelope.height
        _require_bounded_normalized_coefficient(height)
    shared = _shared_denominator_of_pairs(envelopes)
    if shared is None:
        fallback = _signed_sum_envelope(
            envelopes,
            signs=(1,) * len(envelopes),
            validate_budget=validate_budget,
        )
        return _CoefficientEnvelope(
            height,
            fallback.numerator_digits,
            fallback.denominator_digits,
            None,
        )
    denominator_digits = min(
        sum(envelope.denominator_digits for envelope in envelopes),
        len(str(shared)),
    )
    lifted_numerator = 0
    for left_index, left in enumerate(envelopes):
        for right in envelopes[left_index + 1 :]:
            for left_numerator, left_denominator in left.pairs or ():
                for right_numerator, right_denominator in right.pairs or ():
                    difference = abs(
                        Fraction(left_numerator, left_denominator)
                        - Fraction(right_numerator, right_denominator)
                    )
                    lifted_numerator = max(lifted_numerator, difference.numerator)
    numerator_digits = len(str(lifted_numerator))
    if validate_budget:
        _require_bounded_coefficient_digit_budget(
            numerator_digits,
            denominator_digits,
        )
    return _CoefficientEnvelope(
        height,
        numerator_digits,
        denominator_digits,
        ((lifted_numerator, shared),),
    )


def _maximal_digit_envelope(
    envelopes: tuple[_CoefficientEnvelope, ...],
    *,
    height: Fraction,
) -> _CoefficientEnvelope:
    """Carry the widest child digit budgets without compounding them."""

    return _CoefficientEnvelope(
        height,
        max((envelope.numerator_digits for envelope in envelopes), default=0),
        max((envelope.denominator_digits for envelope in envelopes), default=0),
        None,
    )


def _closed_value_envelope(value: Fraction) -> _CoefficientEnvelope:
    return _CoefficientEnvelope(
        abs(value),
        len(str(abs(value.numerator))),
        len(str(value.denominator)),
        ((value.numerator, value.denominator),),
    )


def _affine_form_envelope(
    form: _AffineForm, *, height: Fraction
) -> _CoefficientEnvelope:
    """Read exact digit budgets off the flattened coefficients of one form."""

    coefficients = (form[1], *(coefficient for coefficient, _core in form[0]))
    return _CoefficientEnvelope(
        height,
        max(len(str(abs(value.numerator))) for value in coefficients),
        max(len(str(value.denominator)) for value in coefficients),
        _capped_pairs({(value.numerator, value.denominator) for value in coefficients}),
    )


def _scaled_coefficient_envelope(
    coefficient: Fraction,
    envelope: _CoefficientEnvelope,
) -> _CoefficientEnvelope:
    """Validate a scalar against a dependent's budget before scaling it."""

    numerator_digits = len(str(abs(coefficient.numerator))) + envelope.numerator_digits
    denominator_digits = len(str(coefficient.denominator)) + envelope.denominator_digits
    pairs = _scaled_pairs(coefficient, envelope.pairs)
    common_denominator = _pairs_common_denominator(pairs)
    if common_denominator is not None:
        denominator_digits = min(denominator_digits, len(str(common_denominator)))
        widest_scaled_numerator = max(
            abs(numerator) for numerator, _denominator in pairs or ()
        )
        if widest_scaled_numerator:
            numerator_digits = min(numerator_digits, len(str(widest_scaled_numerator)))
    _require_bounded_coefficient_digit_budget(numerator_digits, denominator_digits)
    return _CoefficientEnvelope(
        coefficient * envelope.height,
        numerator_digits,
        denominator_digits,
        pairs,
    )


def _combined_signed_form(
    children: tuple[Any, ...],
    signs: tuple[int, ...],
    *,
    form_by_id: dict[int, _AffineForm | None],
) -> _AffineForm | None:
    """Add signed affine forms, bounding every merged coefficient and constant."""

    merged: dict[int, list[Any]] = {}
    total = Fraction(0)
    for child, sign in zip(children, signs, strict=True):
        child_form = form_by_id.get(child.get_id())
        if child_form is None:
            return None
        terms, offset = child_form
        for coefficient, core in terms:
            entry = merged.setdefault(core.get_id(), [Fraction(0), core])
            entry[0] += sign * coefficient
            _require_bounded_normalized_coefficient(entry[0])
        total += sign * offset
        _require_bounded_normalized_coefficient(total)
    return (
        tuple((entry[0], entry[1]) for entry in merged.values() if entry[0] != 0),
        total,
    )


def _exact_integer_division_form(
    form: _AffineForm,
    divisor: Fraction,
) -> _AffineForm | None:
    """Divide an affine form by an integer constant under SMT-LIB div semantics.

    Exact whenever the divisor divides every coefficient: for integers s, o, d
    with d dividing s, ``(div (+ (* s x) o) d)`` equals ``(* (/ s d) x)``
    plus the Euclidean quotient of ``o`` by ``d``, since the constant remainder
    ``o mod d`` already satisfies the Euclidean range on its own.
    """

    if divisor.denominator != 1:
        return None
    terms, offset = form
    if offset.denominator != 1:
        return None
    divided_terms: list[tuple[Fraction, Any]] = []
    for coefficient, core in terms:
        if coefficient.denominator != 1 or coefficient.numerator % divisor.numerator:
            return None
        quotient = coefficient / divisor
        _require_bounded_normalized_coefficient(quotient)
        divided_terms.append((quotient, core))
    quotient_offset = _euclidean_quotient(offset, divisor)
    _require_bounded_normalized_coefficient(quotient_offset)
    return (tuple(divided_terms), quotient_offset)


def _euclidean_quotient(value: Fraction, divisor: Fraction) -> Fraction:
    quotient = Fraction(value // divisor)
    if value - divisor * quotient < 0:
        quotient += 1
    return quotient


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
            raise _logic_error("normalized integer coefficient must remain integral")
        return z3.IntVal(value.numerator, ctx=template.ctx)
    if template.sort().kind() == z3.Z3_REAL_SORT:
        return z3.RealVal(
            f"{value.numerator}/{value.denominator}",
            ctx=template.ctx,
        )
    raise _logic_error("normalized coefficient must have an arithmetic sort")


def _require_bounded_coefficient_digit_budget(
    numerator_digits: int,
    denominator_digits: int,
) -> None:
    if (
        numerator_digits > _MAX_CORE_NUMERAL_DIGITS
        or denominator_digits > _MAX_CORE_NUMERAL_DIGITS
    ):
        raise _logic_error(
            "normalized SMT coefficient numerator and denominator may contain at "
            f"most {_MAX_CORE_NUMERAL_DIGITS} digits"
        )


def _require_bounded_normalized_coefficient(value: Fraction) -> None:
    if any(
        len(str(abs(component))) > _MAX_CORE_NUMERAL_DIGITS
        for component in (value.numerator, value.denominator)
    ):
        raise _logic_error(
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
    try:
        import z3
    except (ImportError, OSError) as exc:
        return "UNKNOWN", (), f"the Z3 backend could not initialize: {exc}"[:1_024]

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
    except (OSError, ValueError, z3.Z3Exception):
        return "UNKNOWN", (), "Z3 could not complete the bounded source check."


def _replay_source(
    source: SmtUnsatCoreRequest,
    selected_indices: tuple[int, ...] | None,
) -> tuple[Literal["SAT", "UNSAT", "UNKNOWN"], str | None]:
    try:
        import z3
    except (ImportError, OSError) as exc:
        return "UNKNOWN", f"the Z3 backend could not initialize: {exc}"[:1_024]

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
    except (OSError, ValueError, z3.Z3Exception):
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
