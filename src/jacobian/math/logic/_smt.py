"""Bounded SMT-LIB solver contracts and direct Z3 kernel."""

from __future__ import annotations

import json
import math
import sys
import time
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, NamedTuple, Self

from pydantic import (
    Field,
    StrictInt,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    bind_request_deadline,
    current_request_execution,
    remaining_timeout_ms,
    request_checkpoint,
    require_execution_deadline,
)
from jacobian._models import StrictModel
from jacobian._worker_errors import decode_worker_execution_error
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic._solver_errors import (
    _Z3_SOURCE_DIAGNOSTIC,
    _classify_exhaustion,
    _project_unknown,
    _raise_exhaustion,
    _UnknownResource,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    MAX_REAL_ALGEBRAIC_DEGREE,
    RealAlgebraicValue,
)
from jacobian.process import (
    ProcessResourceLimits,
    check_bounded_process_result,
    run_bounded_process,
    worker_environment,
)

_MAX_SMTLIB_BYTES = 128_000
# Structural SMT-LIB budgets, enforced before Z3 sees the source. Depth bounds
# the recursive-descent parser's native stack per nesting level; compound terms
# bound the assertion-DAG nodes the parser allocates and the solver preprocesses;
# declarations bound the symbol table and the width of any returned model;
# numeral digits bound the big-integer expansion of one literal spelling, the
# quantity whose coefficient work measurably outgrows Z3's own timeout and
# rlimit checks near 16k digits (4,096 keeps worst measured shapes within a
# small multiple of the declared wall time).
_MAX_SMTLIB_DEPTH = 512
_MAX_SMTLIB_TERMS = 32_768
_MAX_SMTLIB_DECLARATIONS = 4_096
_MAX_SMTLIB_NUMERAL_DIGITS = 4_096
# Parsed closed arithmetic may grow beyond one literal (a product of two
# admitted literals is roughly twice as wide), but repeated DAG multiplication
# must not be allowed to expand without an owner-level envelope.
_MAX_SMTLIB_ARITHMETIC_BITS = _MAX_SMTLIB_NUMERAL_DIGITS * 16
_MAX_SMTLIB_ARITHMETIC_WORK = _MAX_SMTLIB_NUMERAL_DIGITS**2 * 64
# Request-scoped solver budgets beyond wall time. Z3 rlimit is a deterministic
# work measure: identical requests cut off identically regardless of host load
# or speed. The ceiling is orders of magnitude above admitted easy queries
# (measured <1k units) while cutting runaway search within seconds at a
# measured ~0.2-5M units/s across QF regimes. max_memory caps Z3's own arena so
# exhaustion surfaces as typed execution failure instead of host memory pressure.
_SOLVER_RLIMIT = 20_000_000
_SOLVER_MAX_MEMORY_MB = 1024
_DEFAULT_LOGIC_TIMEOUT_MS = 10_000
_MAX_LOGIC_TIMEOUT_MS = 120_000
_LOGIC_TIMEOUT_DESCRIPTION = (
    "Full-lifecycle wall-clock limit in milliseconds, from 1 through 120000, "
    "covering worker startup, parsing, solving, result validation, and projection. "
    "This does not increase the operation's separate deterministic solver work limit."
)
_MAX_MODEL_BYTES = 64_000
_SMT_WORKER = Path(__file__).with_name("_smt_worker.py")
# JSON escaping can expand a model's UTF-8 spelling by almost twofold.  The
# process capture allowance therefore bounds the transport representation,
# while ``SmtSolveResult`` continues to own the public 64 KiB model contract.
_SMT_WORKER_OUTPUT_BYTES = (_MAX_MODEL_BYTES * 2) + 4_096
_SMT_WORKER_ERROR_BYTES = 16_384
_SMT_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
# The worker receives no legitimate filesystem output.  Retain a small cap for
# backend diagnostics while preventing an admitted query from filling the
# inherited filesystem if a native dependency misbehaves.
_SMT_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024
_SUPPORTED_SMTLIB_COMMANDS = frozenset(
    {"set-logic", "declare-const", "declare-fun", "assert", "check-sat"}
)
_SUPPORTED_SMTLIB_COMMANDS_DESCRIPTION = (
    "set-logic, declare-const, declare-fun, assert, and check-sat"
)


def _execution_deadline(timeout_ms: int, stage: str) -> float:
    now = time.monotonic()
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else now
    owner_deadline = started_at + (timeout_ms / 1_000)
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    _require_execution_deadline(deadline, stage)
    return deadline


def _require_execution_deadline(deadline: float, stage: str) -> None:
    request_checkpoint(stage)
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(f"request deadline expired {stage}")


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class SmtLogic(StrEnum):
    QF_UF = "QF_UF"
    QF_LIA = "QF_LIA"
    QF_LRA = "QF_LRA"
    QF_NRA = "QF_NRA"


# Bounded structurally enforceable quantifier-free nonlinear real-arithmetic
# grammar. The admitted terms are Boolean combinations of equalities,
# inequalities, and strict inequalities between rational polynomials over
# declared real constants. Division, integer variables, quantifiers, and
# transcendental or underspecified operators are rejected before Z3 solves.
# The variable, assertion, degree, monomial, coefficient-digit, and aggregate
# polynomial-work ceilings bound polynomial expansion independently of the
# source byte and compound-term lexicons.
_MAX_NRA_VARIABLES = 32
_MAX_NRA_ASSERTIONS = 1_024
_MAX_NRA_TOTAL_DEGREE = 16
_MAX_NRA_MONOMIALS = 4_096
_MAX_NRA_COEFFICIENT_DIGITS = _MAX_SMTLIB_NUMERAL_DIGITS
_MAX_NRA_POLYNOMIAL_WORK = 1_000_000
# A satisfying model is published only through exact carriers. Rational
# components use the canonical reduced rational; irrational components are
# admitted exactly through the real algebraic value when their minimal
# polynomial fits the shared carrier envelope. Anything else returns UNKNOWN.
_MAX_NRA_MODEL_DEGREE = MAX_REAL_ALGEBRAIC_DEGREE
_MAX_NRA_MODEL_COEFFICIENT_DIGITS = MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
_MAX_NRA_EXACT_MODEL_BYTES = 64_000


class _ArithmeticBound(NamedTuple):
    numerator_bits: int
    denominator_bits: int


def _saturating_add(*values: int) -> int:
    return min(_MAX_SMTLIB_ARITHMETIC_BITS + 1, sum(values))


def _saturating_work_add(total: int, amount: int) -> int:
    return min(_MAX_SMTLIB_ARITHMETIC_WORK + 1, total + amount)


def _check_arithmetic_bound(bound: _ArithmeticBound) -> None:
    if (
        bound.numerator_bits > _MAX_SMTLIB_ARITHMETIC_BITS
        or bound.denominator_bits > _MAX_SMTLIB_ARITHMETIC_BITS
    ):
        raise ValueError("SMT closed arithmetic exceeds the admitted growth bound")


def _literal_arithmetic_bound(expression: Any, z3: Any) -> _ArithmeticBound | None:
    if z3.is_int_value(expression):
        return _ArithmeticBound(max(1, abs(expression.as_long()).bit_length()), 1)
    if z3.is_rational_value(expression):
        value = expression.as_fraction()
        return _ArithmeticBound(
            max(1, abs(value.numerator).bit_length()),
            max(1, value.denominator.bit_length()),
        )
    return None


def _sum_arithmetic_bounds(
    bounds: tuple[_ArithmeticBound, ...], *, extra_terms: int = 0
) -> _ArithmeticBound:
    denominator_bits = _saturating_add(*(bound.denominator_bits for bound in bounds))
    numerator_bits = _saturating_add(
        max(
            bound.numerator_bits + denominator_bits - bound.denominator_bits
            for bound in bounds
        ),
        max(1, extra_terms.bit_length()),
    )
    return _ArithmeticBound(numerator_bits, denominator_bits)


def _multiply_arithmetic_bounds(
    bounds: tuple[_ArithmeticBound, ...],
) -> _ArithmeticBound:
    return _ArithmeticBound(
        _saturating_add(*(bound.numerator_bits for bound in bounds)),
        _saturating_add(*(bound.denominator_bits for bound in bounds)),
    )


def _arithmetic_node_bound(
    expression: Any, bounds: tuple[_ArithmeticBound, ...], z3: Any
) -> _ArithmeticBound:
    """Bound coefficient height, treating variables as unit formal terms.

    Addition and multiplication bound the common denominator and the sum of
    absolute numerator coefficients. The same bound therefore covers closed
    constants, affine expressions, and mixtures of the two without evaluating
    them. The fragment probe still owns linearity and theory membership.
    """

    kind = expression.decl().kind()
    if kind in (z3.Z3_OP_ADD, z3.Z3_OP_SUB):
        return _sum_arithmetic_bounds(bounds, extra_terms=len(bounds))
    if kind == z3.Z3_OP_MUL:
        return _multiply_arithmetic_bounds(bounds)
    if kind == z3.Z3_OP_DIV:
        left, right = bounds
        return _ArithmeticBound(
            _saturating_add(left.numerator_bits, right.denominator_bits),
            _saturating_add(left.denominator_bits, right.numerator_bits),
        )
    if kind in (z3.Z3_OP_IDIV, z3.Z3_OP_MOD, z3.Z3_OP_REM):
        # Integer quotient and signed remainder do not have rational quotient
        # heights. In particular, mod(-1, b) can be b-1, not a small numerator
        # over a large denominator. Retain both operand heights conservatively.
        return _ArithmeticBound(
            _saturating_add(max(bound.numerator_bits for bound in bounds), 1), 1
        )
    if kind == z3.Z3_OP_TO_INT:
        return _ArithmeticBound(_saturating_add(bounds[0].numerator_bits, 1), 1)
    if kind in (z3.Z3_OP_TO_REAL, z3.Z3_OP_UMINUS):
        return bounds[0]
    if kind == z3.Z3_OP_ITE:
        return _ArithmeticBound(
            max(bound.numerator_bits for bound in bounds),
            max(bound.denominator_bits for bound in bounds),
        )
    raise ValueError("unsupported arithmetic operator in SMT fragment")


def _require_closed_arithmetic_growth(assertions: tuple[Any, ...], z3: Any) -> None:
    """Bound parsed arithmetic before simplification, without expanding its DAG."""

    memo: dict[int, _ArithmeticBound | None] = {}
    pending: list[tuple[Any, bool]] = [(assertion, False) for assertion in assertions]
    work = 0
    while pending:
        expression, expanded = pending.pop()
        expression_id = expression.get_id()
        if expression_id in memo:
            continue
        literal = _literal_arithmetic_bound(expression, z3)
        if literal is not None:
            _check_arithmetic_bound(literal)
            memo[expression_id] = literal
            continue
        children = expression.children()
        if not expanded and children:
            pending.append((expression, True))
            pending.extend(
                (child, False) for child in children if child.get_id() not in memo
            )
            continue
        if not z3.is_arith(expression):
            memo[expression_id] = None
            continue
        if not children:
            memo[expression_id] = _ArithmeticBound(1, 1)
            continue
        # The condition of a numeric ite is Boolean, not an arithmetic operand.
        operands = children[1:] if z3.is_app_of(expression, z3.Z3_OP_ITE) else children
        bounds = tuple(memo[child.get_id()] for child in operands)
        if any(bound is None for bound in bounds):
            raise ValueError("unsupported arithmetic operand in SMT fragment")
        numeric_bounds = tuple(bound for bound in bounds if bound is not None)
        bound = _arithmetic_node_bound(expression, numeric_bounds, z3)
        _check_arithmetic_bound(bound)
        # A sum of operand widths squared bounds schoolbook cross-products,
        # products and quotient work for this parsed arithmetic node. Reused
        # children are visited once but counted at every parent occurrence.
        work = _saturating_work_add(
            work,
            sum(
                value.numerator_bits + value.denominator_bits
                for value in numeric_bounds
            )
            ** 2,
        )
        if work > _MAX_SMTLIB_ARITHMETIC_WORK:
            raise ValueError("SMT arithmetic exceeds the admitted work bound")
        memo[expression_id] = bound


def _consume_smtlib_string(source: str, position: int) -> int:
    position += 1
    while position < len(source):
        if source[position] != '"':
            position += 1
        elif position + 1 < len(source) and source[position + 1] == '"':
            position += 2
        else:
            return position + 1
    return position


def _consume_smtlib_quoted_symbol(source: str, position: int) -> int:
    r"""Scan a quoted SMT-LIB symbol, handling escaped bars (\| inside |...|)."""
    pos = position + 1
    while pos < len(source):
        if source[pos] == chr(92) and pos + 1 < len(source) and source[pos + 1] == "|":
            pos += 2
        elif source[pos] == "|":
            return pos + 1
        else:
            pos += 1
    return len(source)


def _consume_smtlib_atom(source: str, position: int) -> int:
    while (
        position < len(source)
        and not source[position].isspace()
        and source[position] not in ';()"|'
    ):
        position += 1
    return position


def _tokenize_smtlib(source: str) -> tuple[str, ...]:
    """Tokenize enough SMT-LIB syntax to distinguish real command forms."""

    tokens: list[str] = []
    position = 0
    while position < len(source):
        character = source[position]
        if character.isspace():
            position += 1
        elif character == ";":
            newline = source.find("\n", position)
            position = len(source) if newline == -1 else newline + 1
        elif character in "()":
            tokens.append(character)
            position += 1
        elif character == '"':
            position = _consume_smtlib_string(source, position)
            tokens.append("<string>")
        elif character == "|":
            position = _consume_smtlib_quoted_symbol(source, position)
            tokens.append("<quoted-symbol>")
        else:
            end = _consume_smtlib_atom(source, position)
            tokens.append(source[position:end])
            position = end
    return tuple(tokens)


def _top_level_smtlib_commands(source: str) -> tuple[tuple[str, ...], ...]:
    """Return the heads and immediate atoms of complete top-level commands.

    This deliberately does not parse SMT-LIB.  Z3 remains the parser and solver;
    the small lexical pass only prevents comments, strings, and nested terms from
    impersonating the two boundary commands whose presence this contract owns.
    """

    commands: list[tuple[str, ...]] = []
    command: list[str] = []
    depth = 0
    for token in _tokenize_smtlib(source):
        if token == "(":
            if depth == 0:
                command = []
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                commands.append(tuple(command))
        elif depth == 1:
            command.append(token)
    return tuple(commands)


class _SmtLibStructure(NamedTuple):
    """Lexical structure of one SMT-LIB source, measured without parsing it."""

    max_depth: int
    compound_terms: int
    numeral_digits: int


def _atom_numeral_weight(atom: str) -> int:
    """Return the digit width of one classified numeric-literal token.

    Numeral, decimal, and bit-vector spellings expand into big integers or
    rationals inside the solver. Digits inside simple symbols are interned
    names and carry no weight; malformed tokens remain the backend parser's
    typed rejection.
    """

    if atom.startswith("#"):
        return max(len(atom) - 2, 0)
    if atom.isdigit():
        return len(atom)
    head, separator, tail = atom.partition(".")
    if separator and head.isdigit() and (not tail or tail.isdigit()):
        return len(head) + len(tail)
    return 0


def _smtlib_structure(source: str) -> _SmtLibStructure:
    """Measure nesting depth, compound-term count, and numeral width lexically.

    This is the same deliberately non-parsing scan used for command shape; it
    bounds what ``z3.parse_smt2_string`` may build before that parser runs.
    An indexed literal such as ``(_ bvN w)`` spells its value inside a simple
    symbol, so index position decides whether ``bvN`` digits carry numeral
    weight; elsewhere digits stay interned names.
    """

    max_depth = depth = compound_terms = numeral_digits = 0
    indexed_depth: int | None = None
    previous_token = ""
    for token in _tokenize_smtlib(source):
        if token == "(":
            depth += 1
            compound_terms += 1
            if depth > max_depth:
                max_depth = depth
        elif token == ")":
            if indexed_depth == depth:
                indexed_depth = None
            depth -= 1
        elif previous_token == "(" and token == "_":
            indexed_depth = depth
        else:
            weight = _atom_numeral_weight(token)
            if (
                indexed_depth is not None
                and token.startswith("bv")
                and token[2:].isdigit()
            ):
                weight = len(token) - 2
            if weight > numeral_digits:
                numeral_digits = weight
        previous_token = token
    return _SmtLibStructure(max_depth, compound_terms, numeral_digits)


def _exception_message(exc: Exception) -> str:
    message = exc.args[0] if exc.args else ""
    if isinstance(message, bytes):
        return message.decode("ascii", errors="replace")
    return str(message)


def _is_smtlib_source_diagnostic(exc: Exception) -> bool:
    """Report whether one backend parse exception diagnoses the caller's source.

    Z3's SMT-LIB front end reports caller-correctable source problems as
    ``(error "line L column C: <diagnostic>")``. Backend conditions such as
    exhausted memory or interruption surface through Z3's fixed error-code
    message table and carry no source locator. A located diagnostic names a
    grammar defect regardless of which resource keywords its text contains,
    because the diagnostic quotes caller-controlled source spellings that may
    legitimately contain words such as ``memory``.
    """

    return _Z3_SOURCE_DIAGNOSTIC.search(_exception_message(exc)) is not None


class SmtSolveRequest(StrictModel):
    logic: SmtLogic = Field(
        description=(
            "Declared quantifier-free fragment. QF_UF admits Boolean-sorted "
            "uninterpreted constants and functions; QF_LIA and QF_LRA admit pure "
            "linear integer and real arithmetic; QF_NRA admits quantifier-free "
            "Boolean combinations of rational-polynomial equalities, "
            "inequalities, and strict inequalities over declared real variables "
            "with no division, integer variables, or transcendental operators. "
            "The nonlinear fragment is additionally bounded by "
            f"{_MAX_NRA_VARIABLES} real variables, {_MAX_NRA_ASSERTIONS} "
            f"assertions, total degree {_MAX_NRA_TOTAL_DEGREE}, "
            f"{_MAX_NRA_MONOMIALS} monomials per polynomial, "
            f"{_MAX_NRA_COEFFICIENT_DIGITS} coefficient digits, and "
            f"{_MAX_NRA_POLYNOMIAL_WORK:,} coefficient operations."
        )
    )
    smtlib: str = Field(
        min_length=1,
        max_length=_MAX_SMTLIB_BYTES,
        description=(
            "ASCII SMT-LIB that declares logic, contains exactly one check-sat command, "
            "and ends with that command. Bounded before parsing: nesting depth at most "
            f"{_MAX_SMTLIB_DEPTH}, compound terms at most {_MAX_SMTLIB_TERMS}, declared "
            f"symbols at most {_MAX_SMTLIB_DECLARATIONS}, and any one numeral, decimal, "
            f"or indexed bit-vector spelling at most {_MAX_SMTLIB_NUMERAL_DIGITS} digits. "
            "Closed arithmetic and coefficient-only linear scaling are also bounded "
            f"at {_MAX_SMTLIB_ARITHMETIC_BITS} bits per numerator or denominator and "
            f"{_MAX_SMTLIB_ARITHMETIC_WORK:,} units of aggregate arithmetic work. "
            "The source is structurally admitted before execution; Z3 parsing and "
            "solving occur within the operation's bounded execution envelope. "
            "The only accepted top-level commands are "
            f"{_SUPPORTED_SMTLIB_COMMANDS_DESCRIPTION}; definitions such as "
            "define-fun are not accepted."
        ),
        examples=[
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        ],
    )
    timeout_ms: StrictInt = Field(
        default=_DEFAULT_LOGIC_TIMEOUT_MS,
        ge=1,
        le=_MAX_LOGIC_TIMEOUT_MS,
        description=_LOGIC_TIMEOUT_DESCRIPTION,
    )

    @model_validator(mode="after")
    def require_single_smtlib_query(self) -> Self:
        try:
            encoded = self.smtlib.encode("ascii")
        except UnicodeEncodeError as exc:
            raise _validation_error(
                "logic.smtlib_ascii", "SMT-LIB input must be ASCII"
            ) from exc
        if len(encoded) > _MAX_SMTLIB_BYTES:
            raise _validation_error(
                "logic.smtlib_byte_budget", "SMT-LIB input exceeds the byte limit"
            )
        structure = _smtlib_structure(self.smtlib)
        if structure.max_depth > _MAX_SMTLIB_DEPTH:
            raise _validation_error(
                "logic.smtlib_depth_budget",
                f"SMT-LIB nesting exceeds the maximum term depth of {_MAX_SMTLIB_DEPTH}",
            )
        if structure.compound_terms > _MAX_SMTLIB_TERMS:
            raise _validation_error(
                "logic.smtlib_term_budget",
                f"SMT-LIB exceeds the maximum of {_MAX_SMTLIB_TERMS} compound terms",
            )
        if structure.numeral_digits > _MAX_SMTLIB_NUMERAL_DIGITS:
            raise _validation_error(
                "logic.smtlib_numeral_budget",
                "SMT-LIB contains a numeral wider than "
                f"{_MAX_SMTLIB_NUMERAL_DIGITS} digits",
            )
        commands = _top_level_smtlib_commands(self.smtlib)
        declarations = sum(
            1
            for command in commands
            if command[:1] in (("declare-const",), ("declare-fun",))
        )
        if declarations > _MAX_SMTLIB_DECLARATIONS:
            raise _validation_error(
                "logic.smtlib_declaration_budget",
                f"SMT-LIB declares more than {_MAX_SMTLIB_DECLARATIONS} symbols",
            )
        logic_commands = tuple(
            command for command in commands if command[:1] == ("set-logic",)
        )
        if logic_commands != (("set-logic", self.logic.value),):
            raise _validation_error(
                "logic.smtlib_logic_declaration",
                "SMT-LIB input must declare the requested logic",
            )
        if commands.count(("check-sat",)) != 1:
            raise _validation_error(
                "logic.smtlib_check_sat_count",
                "SMT-LIB input must contain exactly one check-sat command",
            )
        if commands[-1:] != (("check-sat",),):
            raise _validation_error(
                "logic.smtlib_check_sat_position",
                "SMT-LIB input must end with its check-sat command",
            )
        for command in commands:
            if command and command[0] not in _SUPPORTED_SMTLIB_COMMANDS:
                raise ValidationError.from_exception_data(
                    type(self).__name__,
                    [
                        {
                            "type": _validation_error(
                                "logic.smtlib_command",
                                f"unsupported SMT-LIB command: {command[0]}; supported commands are "
                                f"{_SUPPORTED_SMTLIB_COMMANDS_DESCRIPTION}. Inline definitions in assertions.",
                            ),
                            "loc": ("smtlib",),
                            "input": self.smtlib,
                        }
                    ],
                )
        return self


class SmtRationalModelValue(StrictModel):
    """One exact rational component of a QF_NRA satisfying model."""

    kind: Literal["RATIONAL"] = "RATIONAL"
    value: CanonicalRational = Field(
        description="Exact reduced rational value of one declared real variable."
    )


class SmtAlgebraicModelValue(StrictModel):
    """One exact real algebraic component of a QF_NRA satisfying model."""

    kind: Literal["ALGEBRAIC"] = "ALGEBRAIC"
    value: RealAlgebraicValue = Field(
        description=(
            "Exact indexed real root of an integer minimal polynomial, using the "
            "domain-owned real algebraic carrier rather than a decimal approximation."
        )
    )


SmtModelValue = Annotated[
    SmtRationalModelValue | SmtAlgebraicModelValue,
    Field(discriminator="kind"),
]


class SmtModelBinding(StrictModel):
    """One declared real variable bound to its exact model value."""

    variable: str = Field(min_length=1, max_length=256)
    value: SmtModelValue


class SmtExactModel(StrictModel):
    """Source-bound exact model of one satisfied QF_NRA query."""

    bindings: tuple[SmtModelBinding, ...] = Field(
        max_length=_MAX_NRA_VARIABLES,
        description=(
            "One binding per declared real variable, in canonical name order. "
            "Each value is an exact canonical rational or an exact real algebraic "
            "root; no floating-point component is ever published."
        ),
    )


class SmtSolveResult(StrictModel):
    """One solver outcome bound to the exact SMT-LIB request it answers."""

    source: SmtSolveRequest
    outcome: Literal["SAT", "UNSAT", "UNKNOWN"]
    model_smtlib: str | None = Field(
        default=None,
        max_length=_MAX_MODEL_BYTES,
        description=(
            "Bounded Z3 display projection of the worker's satisfying model. It is "
            "provided for inspection, not as a canonical mathematical value or an "
            "independently verifiable model encoding. For QF_NRA it renders exact "
            "rational or root-object spells, never decimal approximations."
        ),
    )
    exact_model: SmtExactModel | None = Field(
        default=None,
        description=(
            "Exact source-bound model for a satisfied QF_NRA query. Present only "
            "for QF_NRA SAT outcomes; other logics retain the display projection."
        ),
    )
    exhausted: _UnknownResource | None = Field(default=None)
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_model_to_outcome(self) -> Self:
        if (self.outcome == "SAT") != (self.model_smtlib is not None):
            raise _validation_error(
                "logic.sat_model_outcome", "only a SAT result may carry a model"
            )
        is_nra = self.source.logic == SmtLogic.QF_NRA
        if self.exact_model is not None and (self.outcome != "SAT" or not is_nra):
            raise _validation_error(
                "logic.exact_model_outcome",
                "only a QF_NRA SAT result may carry an exact model",
            )
        if self.outcome == "SAT" and is_nra and self.exact_model is None:
            raise _validation_error(
                "logic.exact_model_required",
                "a QF_NRA SAT result must carry an exact model",
            )
        if self.exhausted is not None and self.outcome != "UNKNOWN":
            raise _validation_error(
                "logic.unknown_exhaustion",
                "only an UNKNOWN result may name an exhausted budget",
            )
        return self


def _solver_settings(timeout_ms: int, *, logic: str | None = None) -> dict[str, int]:
    """Return the full request-scoped Z3 budget: wall time, work, and memory.

    Nonlinear real arithmetic may overrun a per-call timeout until Z3 reaches a
    safe interruption point, so QF_NRA reserves half of the remaining lifecycle
    allowance for the solver call.  The outer killable worker keeps the full
    request wall limit as the hard safety bound.
    """

    timeout = remaining_timeout_ms(timeout_ms)
    if logic == SmtLogic.QF_NRA.value:
        timeout = max(1, timeout // 2)
    return {
        "timeout": timeout,
        "rlimit": _SOLVER_RLIMIT,
        "max_memory": _SOLVER_MAX_MEMORY_MB,
    }


def _require_supported_fragment_nodes(
    assertions: tuple[Any, ...], logic: str, z3: Any
) -> None:
    """Reject theory sorts and applications outside the advertised fragment."""

    if logic == SmtLogic.QF_UF.value:
        allowed_sort_kinds = frozenset({z3.Z3_BOOL_SORT, z3.Z3_UNINTERPRETED_SORT})
    elif logic == SmtLogic.QF_LIA.value:
        allowed_sort_kinds = frozenset({z3.Z3_BOOL_SORT, z3.Z3_INT_SORT})
    else:
        allowed_sort_kinds = frozenset({z3.Z3_BOOL_SORT, z3.Z3_REAL_SORT})

    stack = list(assertions)
    seen: set[int] = set()
    while stack:
        expression = stack.pop()
        expression_id = expression.get_id()
        if expression_id in seen:
            continue
        seen.add(expression_id)
        if expression.sort().kind() not in allowed_sort_kinds:
            raise ValueError(f"SMT terms must belong to the declared {logic} fragment")
        if (
            logic != SmtLogic.QF_UF.value
            and z3.is_app(expression)
            and expression.decl().kind() == z3.Z3_OP_UNINTERPRETED
            and expression.decl().arity() != 0
        ):
            raise ValueError(f"SMT terms must belong to the declared {logic} fragment")
        stack.extend(expression.children())


def _nra_domain_error(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("smtlib",), code=code, message=message
    )


def _nra_resource_error(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("smtlib",), code=code, message=message
    )


def _nra_coefficient_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


class _NraExpressionState:
    """Bounded exact polynomial expansion for one admitted QF_NRA source.

    Every declared real constant becomes a formal variable; addition,
    subtraction, negation, multiplication, and nonnegative integer powers build
    the exact sparse rational polynomial of each Real-sorted subterm. Monomial
    count, total degree, coefficient digits, and aggregate multiplication work
    are charged during expansion so an admitted source cannot materialize an
    unbounded polynomial.
    """

    def __init__(self, z3: Any, index: dict[str, int]) -> None:
        self.z3 = z3
        self.index = index
        self.width = len(index)
        self.work = 0
        self.memo: dict[int, dict[tuple[int, ...], Fraction]] = {}

    def _degree_limit(self, polynomial: dict[tuple[int, ...], Fraction]) -> None:
        if len(polynomial) > _MAX_NRA_MONOMIALS:
            raise _nra_resource_error(
                "logic.smt.nra_monomial_budget",
                f"a QF_NRA polynomial exceeds the {_MAX_NRA_MONOMIALS}-monomial envelope",
            )
        for exponents in polynomial:
            if sum(exponents) > _MAX_NRA_TOTAL_DEGREE:
                raise _nra_resource_error(
                    "logic.smt.nra_degree_budget",
                    f"a QF_NRA polynomial exceeds total degree {_MAX_NRA_TOTAL_DEGREE}",
                )
        for coefficient in polynomial.values():
            if _nra_coefficient_digits(coefficient) > _MAX_NRA_COEFFICIENT_DIGITS:
                raise _nra_resource_error(
                    "logic.smt.nra_coefficient_budget",
                    "a QF_NRA coefficient exceeds the "
                    f"{_MAX_NRA_COEFFICIENT_DIGITS}-digit envelope",
                )

    def _charge_work(self, amount: int) -> None:
        self.work += amount
        if self.work > _MAX_NRA_POLYNOMIAL_WORK:
            raise _nra_resource_error(
                "logic.smt.nra_work_budget",
                f"QF_NRA polynomial expansion exceeds {_MAX_NRA_POLYNOMIAL_WORK} "
                "coefficient operations",
            )

    def constant(self, value: Fraction) -> dict[tuple[int, ...], Fraction]:
        if value == 0:
            return {}
        return {(0,) * self.width: value}

    def one(self) -> dict[tuple[int, ...], Fraction]:
        return {(0,) * self.width: Fraction(1)}

    def add(
        self,
        left: dict[tuple[int, ...], Fraction],
        right: dict[tuple[int, ...], Fraction],
    ) -> dict[tuple[int, ...], Fraction]:
        self._charge_work(len(left) + len(right))
        result = dict(left)
        for exponents, coefficient in right.items():
            total = result.get(exponents, Fraction(0)) + coefficient
            if total:
                result[exponents] = total
            else:
                result.pop(exponents, None)
        self._degree_limit(result)
        return result

    def negate(
        self, polynomial: dict[tuple[int, ...], Fraction]
    ) -> dict[tuple[int, ...], Fraction]:
        return {
            exponents: -coefficient for exponents, coefficient in polynomial.items()
        }

    def multiply(
        self,
        left: dict[tuple[int, ...], Fraction],
        right: dict[tuple[int, ...], Fraction],
    ) -> dict[tuple[int, ...], Fraction]:
        self._charge_work(max(1, len(left)) * max(1, len(right)))
        result: dict[tuple[int, ...], Fraction] = {}
        for left_exponents, left_coefficient in left.items():
            for right_exponents, right_coefficient in right.items():
                exponents = tuple(
                    a + b for a, b in zip(left_exponents, right_exponents, strict=True)
                )
                if sum(exponents) > _MAX_NRA_TOTAL_DEGREE:
                    raise _nra_resource_error(
                        "logic.smt.nra_degree_budget",
                        "a QF_NRA polynomial product exceeds total degree "
                        f"{_MAX_NRA_TOTAL_DEGREE}",
                    )
                total = result.get(exponents, Fraction(0)) + (
                    left_coefficient * right_coefficient
                )
                if total:
                    result[exponents] = total
                else:
                    result.pop(exponents, None)
                if len(result) > _MAX_NRA_MONOMIALS:
                    raise _nra_resource_error(
                        "logic.smt.nra_monomial_budget",
                        "a QF_NRA polynomial product exceeds the "
                        f"{_MAX_NRA_MONOMIALS}-monomial envelope",
                    )
        self._degree_limit(result)
        return result

    def power(
        self,
        polynomial: dict[tuple[int, ...], Fraction],
        exponent: int,
    ) -> dict[tuple[int, ...], Fraction]:
        result = self.one()
        for _ in range(exponent):
            result = self.multiply(result, polynomial)
        return result


def _nra_real_polynomial(
    expression: Any, state: _NraExpressionState
) -> dict[tuple[int, ...], Fraction]:
    z3 = state.z3
    expression_id = expression.get_id()
    if expression_id in state.memo:
        return state.memo[expression_id]
    if z3.is_rational_value(expression):
        value = expression.as_fraction()
        polynomial = state.constant(value)
        state.memo[expression_id] = polynomial
        return polynomial
    if not z3.is_app(expression):
        raise _nra_domain_error(
            "logic.smt.nra_operator",
            "QF_NRA admits only applied polynomial terms",
        )
    kind = expression.decl().kind()
    if kind == z3.Z3_OP_UNINTERPRETED:
        name = expression.decl().name()
        position = state.index.get(name)
        if position is None:
            raise _nra_domain_error(
                "logic.smt.nra_declaration",
                "QF_NRA admits only declared real constants",
            )
        vector = tuple(1 if i == position else 0 for i in range(state.width))
        polynomial = {vector: Fraction(1)}
        state.memo[expression_id] = polynomial
        return polynomial
    children = expression.children()
    if kind in (z3.Z3_OP_ADD, z3.Z3_OP_SUB):
        result = _nra_real_polynomial(children[0], state)
        for child in children[1:]:
            operand = _nra_real_polynomial(child, state)
            if kind == z3.Z3_OP_SUB:
                result = state.add(result, state.negate(operand))
            else:
                result = state.add(result, operand)
        state.memo[expression_id] = result
        return result
    if kind == z3.Z3_OP_UMINUS:
        result = state.negate(_nra_real_polynomial(children[0], state))
        state.memo[expression_id] = result
        return result
    if kind == z3.Z3_OP_MUL:
        result = state.one()
        for child in children:
            result = state.multiply(result, _nra_real_polynomial(child, state))
        state.memo[expression_id] = result
        return result
    if kind == z3.Z3_OP_POWER:
        exponent_expression = children[1]
        if not z3.is_int_value(exponent_expression):
            raise _nra_domain_error(
                "logic.smt.nra_operator",
                "QF_NRA admits only constant nonnegative integer powers",
            )
        exponent = exponent_expression.as_long()
        if exponent < 0 or exponent > _MAX_NRA_TOTAL_DEGREE:
            raise _nra_resource_error(
                "logic.smt.nra_degree_budget",
                f"QF_NRA powers are bounded by total degree {_MAX_NRA_TOTAL_DEGREE}",
            )
        result = state.power(_nra_real_polynomial(children[0], state), exponent)
        state.memo[expression_id] = result
        return result
    raise _nra_domain_error(
        "logic.smt.nra_operator",
        f"QF_NRA rejects the non-polynomial operator {expression.decl().name()}",
    )


def _nra_require_boolean(expression: Any, state: _NraExpressionState) -> None:
    z3 = state.z3
    if not z3.is_app(expression):
        raise _nra_domain_error(
            "logic.smt.nra_operator",
            "QF_NRA admits only applied Boolean terms",
        )
    kind = expression.decl().kind()
    if kind in (z3.Z3_OP_TRUE, z3.Z3_OP_FALSE):
        return
    children = expression.children()
    if kind in (
        z3.Z3_OP_AND,
        z3.Z3_OP_OR,
        z3.Z3_OP_XOR,
        z3.Z3_OP_NOT,
        z3.Z3_OP_IMPLIES,
    ):
        for child in children:
            _nra_require_boolean(child, state)
        return
    if kind in (z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT):
        sorts = {child.sort().kind() for child in children}
        if sorts <= {z3.Z3_REAL_SORT}:
            for child in children:
                _nra_real_polynomial(child, state)
            return
        if sorts <= {z3.Z3_BOOL_SORT}:
            for child in children:
                _nra_require_boolean(child, state)
            return
        raise _nra_domain_error(
            "logic.smt.nra_sort",
            "QF_NRA equalities must compare Real or Boolean terms, not mixed sorts",
        )
    if kind in (
        z3.Z3_OP_LT,
        z3.Z3_OP_LE,
        z3.Z3_OP_GT,
        z3.Z3_OP_GE,
    ):
        for child in children:
            if child.sort().kind() != z3.Z3_REAL_SORT:
                raise _nra_domain_error(
                    "logic.smt.nra_sort",
                    "QF_NRA inequalities require Real-sorted polynomial terms",
                )
            _nra_real_polynomial(child, state)
        return
    raise _nra_domain_error(
        "logic.smt.nra_operator",
        f"QF_NRA rejects the non-polynomial Boolean operator {expression.decl().name()}",
    )


def _require_qf_nra_fragment(assertions: tuple[Any, ...], z3: Any) -> None:
    """Admit one structurally bounded quantifier-free nonlinear real query."""

    context = assertions[0].ctx
    raw_goal = z3.Goal(ctx=context)
    raw_goal.add(*assertions)
    if float(z3.Probe("has-quantifiers", ctx=context)(raw_goal)) != 0.0:
        raise _nra_domain_error(
            "logic.smt.nra_quantifier",
            "QF_NRA admits no quantifiers",
        )
    if len(assertions) > _MAX_NRA_ASSERTIONS:
        raise _nra_resource_error(
            "logic.smt.nra_assertion_budget",
            f"QF_NRA admits at most {_MAX_NRA_ASSERTIONS} assertions",
        )

    variable_names: set[str] = set()
    stack = list(assertions)
    seen: set[int] = set()
    while stack:
        expression = stack.pop()
        expression_id = expression.get_id()
        if expression_id in seen:
            continue
        seen.add(expression_id)
        if not z3.is_app(expression):
            raise _nra_domain_error(
                "logic.smt.nra_operator",
                "QF_NRA admits no quantifiers or binding terms",
            )
        sort_kind = expression.sort().kind()
        if sort_kind not in (z3.Z3_BOOL_SORT, z3.Z3_REAL_SORT):
            raise _nra_domain_error(
                "logic.smt.nra_sort",
                "QF_NRA admits only Real-sorted variables and Boolean structure",
            )
        if expression.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            if expression.decl().arity() != 0 or sort_kind != z3.Z3_REAL_SORT:
                raise _nra_domain_error(
                    "logic.smt.nra_declaration",
                    "QF_NRA admits only zero-arity Real-sorted declarations",
                )
            variable_names.add(expression.decl().name())
        stack.extend(expression.children())

    variables = tuple(sorted(variable_names))
    if len(variables) > _MAX_NRA_VARIABLES:
        raise _nra_resource_error(
            "logic.smt.nra_variable_budget",
            f"QF_NRA admits at most {_MAX_NRA_VARIABLES} real variables",
        )
    index = {name: position for position, name in enumerate(variables)}
    state = _NraExpressionState(z3, index)
    for assertion in assertions:
        _nra_require_boolean(assertion, state)


def _probe_declared_logic(
    assertions: Any, logic: str, z3: Any, *, simplify_arithmetic: bool = True
) -> None:
    """Reject parsed assertions outside the advertised quantifier-free fragment."""

    assertion_tuple = tuple(assertions)
    if not assertion_tuple:
        return
    if logic == SmtLogic.QF_NRA.value:
        _require_qf_nra_fragment(assertion_tuple, z3)
        return
    _require_supported_fragment_nodes(assertion_tuple, logic, z3)
    if logic in (SmtLogic.QF_LIA.value, SmtLogic.QF_LRA.value):
        _require_closed_arithmetic_growth(assertion_tuple, z3)
    if logic == SmtLogic.QF_UF.value:
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
        stack = list(assertion_tuple)
        seen: set[int] = set()
        while stack:
            expression = stack.pop()
            if expression.get_id() in seen:
                continue
            seen.add(expression.get_id())
            if (
                not z3.is_app(expression)
                or not z3.is_bool(expression)
                or expression.decl().kind() not in allowed_kinds
            ):
                raise ValueError("SMT terms must belong to the declared QF_UF fragment")
            stack.extend(expression.children())
        return

    raw_goal = z3.Goal(ctx=assertion_tuple[0].ctx)
    raw_goal.add(*assertion_tuple)
    has_quantifiers = float(
        z3.Probe("has-quantifiers", ctx=assertion_tuple[0].ctx)(raw_goal)
    )
    goal = z3.Goal(ctx=assertion_tuple[0].ctx)
    goal.add(
        *(z3.simplify(assertion) for assertion in assertion_tuple)
        if simplify_arithmetic
        else assertion_tuple
    )
    probe_name = "is-lia" if logic == SmtLogic.QF_LIA.value else "is-lra"
    belongs_to_fragment = float(z3.Probe(probe_name, ctx=assertion_tuple[0].ctx)(goal))
    if has_quantifiers != 0.0 or belongs_to_fragment != 1.0:
        raise ValueError(f"SMT terms must belong to the declared {logic} fragment")


def _rejected_response(error: OperationDomainValidationError) -> dict[str, Any]:
    """Project one typed admission failure into the bounded worker response."""

    detail = error.errors()[0]
    return {
        "kind": "rejected",
        "resource": isinstance(error, OperationResourceAdmissionError),
        "code": str(detail["type"])[:128],
        "message": str(detail["msg"])[:1_024],
    }


def _within_digit_bound(magnitude: int, digits: int) -> bool:
    if magnitude.bit_length() <= 3 * digits:
        return True
    return bool(magnitude < 10**digits)


def _nra_exact_model_value(value: Any, z3: Any) -> dict[str, Any] | None:
    """Return one exact wire component for a Z3 model value, or ``None``.

    Rational values are reduced exactly. Irrational values are admitted only
    through the domain-owned real algebraic carrier as an integer minimal
    polynomial with an increasing real-root index; anything else is refused so
    that no lossy decimal can enter a published mathematical value.
    """

    if z3.is_rational_value(value):
        fraction = value.as_fraction()
        if not _within_digit_bound(
            max(abs(fraction.numerator), fraction.denominator),
            MAX_CANONICAL_RATIONAL_DIGITS,
        ):
            return None
        return {
            "kind": "RATIONAL",
            "value": {"num": fraction.numerator, "den": fraction.denominator},
        }
    if z3.is_algebraic_value(value):
        coefficients: list[int] = []
        for coefficient in value.poly():
            if z3.is_int_value(coefficient):
                coefficients.append(coefficient.as_long())
            elif (
                z3.is_rational_value(coefficient)
                and coefficient.denominator_as_long() == 1
            ):
                coefficients.append(coefficient.numerator_as_long())
            else:
                return None
        coefficients_tuple = tuple(coefficients)
        if len(coefficients_tuple) - 1 > _MAX_NRA_MODEL_DEGREE:
            return None
        if coefficients_tuple[-1] <= 0:
            return None
        if any(
            not _within_digit_bound(abs(coefficient), _MAX_NRA_MODEL_COEFFICIENT_DIGITS)
            for coefficient in coefficients_tuple
        ):
            return None
        root_index = int(value.index())
        if root_index < 1:
            return None
        return {
            "kind": "ALGEBRAIC",
            "value": {
                "polynomial": tuple(reversed(coefficients_tuple)),
                "real_root_index": root_index - 1,
            },
        }
    return None


def _nra_model_projection(bindings: list[tuple[str, Any]], z3: Any) -> str:
    """Render an exact, float-free S-expression projection of an NRA model."""

    lines: list[str] = []
    for name, value in bindings:
        if z3.is_rational_value(value):
            fraction = value.as_fraction()
            rendered = (
                str(fraction.numerator)
                if fraction.denominator == 1
                else f"(/ {fraction.numerator} {fraction.denominator})"
            )
        else:
            rendered = value.sexpr()
        lines.append(f"(define-fun {name} () Real {rendered})")
    return "\n".join(lines)


def _materialize_nra_model(
    model: Any, z3: Any
) -> tuple[dict[str, Any] | None, str | None]:
    """Return an exact model payload and projection, or a bounded UNKNOWN reason."""

    exact_bindings: list[dict[str, Any]] = []
    projection_bindings: list[tuple[str, Any]] = []
    for declaration in sorted(model.decls(), key=lambda decl: decl.name()):
        value = model.eval(declaration(), model_completion=True)
        component = _nra_exact_model_value(value, z3)
        if component is None:
            return (
                None,
                "the solver model has a component outside the exact algebraic envelope",
            )
        exact_bindings.append({"variable": declaration.name(), "value": component})
        projection_bindings.append((declaration.name(), value))
    exact_model: dict[str, Any] = {"bindings": exact_bindings}
    encoded = json.dumps(exact_model, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_NRA_EXACT_MODEL_BYTES:
        return None, "the exact model exceeds the admitted representation envelope"
    model_smtlib = _nra_model_projection(projection_bindings, z3)
    if len(model_smtlib.encode("utf-8")) > _MAX_MODEL_BYTES:
        return None, "the model projection exceeds the admitted display envelope"
    return exact_model, model_smtlib


def _nra_unknown_payload(reason: str | None) -> dict[str, Any]:
    """Project an inconclusive QF_NRA solver answer without raising."""

    exhausted = _classify_exhaustion(reason or "")
    if exhausted is not None:
        detail = f"the solver exhausted its {exhausted} budget"
    elif (reason or "").strip():
        detail = "the solver returned an inconclusive answer"
    else:
        detail = "the solver returned no completeness evidence"
    return {
        "outcome": "UNKNOWN",
        "model_smtlib": None,
        "exact_model": None,
        "exhausted": exhausted,
        "detail": detail,
    }


def _sat_kernel_response(
    model: Any, assertions: tuple[Any, ...], *, is_nra: bool, z3: Any
) -> dict[str, Any]:
    """Validate one Z3 model against every source assertion and project it."""

    if not all(
        z3.is_true(model.eval(assertion, model_completion=True))
        for assertion in assertions
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    if is_nra:
        exact_model, model_smtlib = _materialize_nra_model(model, z3)
        if exact_model is None:
            return {
                "outcome": "UNKNOWN",
                "model_smtlib": None,
                "exact_model": None,
                "exhausted": None,
                "detail": model_smtlib,
            }
        return {
            "outcome": "SAT",
            "model_smtlib": model_smtlib,
            "exact_model": exact_model,
            "exhausted": None,
            "detail": None,
        }
    model_smtlib = model.sexpr()
    if len(model_smtlib.encode("utf-8")) > _MAX_MODEL_BYTES:
        raise OperationResourceExhaustedError(ExecutionResource.OUTPUT)
    return {
        "outcome": "SAT",
        "model_smtlib": model_smtlib,
        "exhausted": None,
        "detail": None,
    }


def _solve_smt_kernel(*, logic: str, smtlib: str, timeout_ms: int) -> dict[str, Any]:
    """Run one complete Z3 lifecycle inside the owned worker process."""

    try:
        import z3
    except (ImportError, OSError) as exc:
        raise OperationBackendError(BackendFailureReason.INITIALIZATION) from exc

    is_nra = logic == SmtLogic.QF_NRA.value
    try:
        assertions = z3.parse_smt2_string(smtlib)
        try:
            _probe_declared_logic(assertions, logic, z3)
        except OperationDomainValidationError as exc:
            return _rejected_response(exc)
        except ValueError as exc:
            return {"kind": "invalid", "detail": str(exc)[:1_024]}
        solver = z3.SolverFor(logic)
        solver.add(assertions)
        solver.set(**_solver_settings(timeout_ms, logic=logic))
        outcome = solver.check()
        if outcome == z3.sat:
            return _sat_kernel_response(
                solver.model(), tuple(assertions), is_nra=is_nra, z3=z3
            )
        if outcome == z3.unsat:
            return {
                "outcome": "UNSAT",
                "model_smtlib": None,
                "exhausted": None,
                "detail": None,
            }
    except (OperationExecutionTimeoutError, OperationExecutionCancelledError):
        raise
    except (OSError, z3.Z3Exception) as exc:
        if isinstance(exc, z3.Z3Exception) and _is_smtlib_source_diagnostic(exc):
            return {
                "kind": "invalid",
                "detail": "SMT-LIB source could not be parsed as SMT-LIB",
            }
        exhausted = _classify_exhaustion(str(exc))
        if is_nra and exhausted is not None:
            return _nra_unknown_payload(str(exc))
        if exhausted is not None:
            _raise_exhaustion(exhausted, cause=exc)
        raise OperationBackendError(BackendFailureReason.ABNORMAL_EXIT) from exc
    if is_nra:
        return _nra_unknown_payload(solver.reason_unknown())
    exhausted, detail = _project_unknown(solver.reason_unknown())
    return {
        "outcome": "UNKNOWN",
        "model_smtlib": None,
        "exhausted": exhausted,
        "detail": detail,
    }


def _run_smt_worker(request: SmtSolveRequest) -> SmtSolveResult:
    """Project one killable worker invocation onto the public typed result."""

    deadline = _execution_deadline(request.timeout_ms, "before SMT worker")
    try:
        # The isolated worker has no reason to inherit the checkout as its
        # current directory.  Its input arrives only through stdin and its
        # only accepted output is the bounded JSON response on stdout.
        with TemporaryDirectory(prefix="jacobian-smt-") as worker_directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before SMT worker"
                )
            completed = run_bounded_process(
                [sys.executable, str(_SMT_WORKER)],
                input_bytes=json.dumps(
                    {
                        "_deadline": deadline,
                        "logic": request.logic.value,
                        "smtlib": request.smtlib,
                        "timeout_ms": request.timeout_ms,
                    },
                    separators=(",", ":"),
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_SMT_WORKER_OUTPUT_BYTES,
                stderr_limit=_SMT_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(request.timeout_ms / 1_000)),
                    address_space_bytes=_SMT_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_SMT_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    check_bounded_process_result(completed)
    require_execution_deadline(deadline)
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        require_execution_deadline(deadline)
        decode_worker_execution_error(response)
        if not isinstance(response, dict):
            raise TypeError("worker response must be an object")
        if "source" in response:
            raise TypeError("worker response must not replace the retained source")
        if (
            set(response) == {"kind", "detail"}
            and response.get("kind") == "invalid"
            and isinstance(response.get("detail"), str)
            and 0 < len(response["detail"]) <= 1_024
        ):
            raise OperationDomainValidationError(
                location=("smtlib",),
                code="logic.smtlib_source",
                message=response["detail"],
            )
        if (
            set(response) == {"kind", "resource", "code", "message"}
            and response.get("kind") == "rejected"
            and isinstance(response.get("resource"), bool)
            and isinstance(response.get("code"), str)
            and 0 < len(response["code"]) <= 128
            and isinstance(response.get("message"), str)
            and 0 < len(response["message"]) <= 1_024
        ):
            error_type = (
                OperationResourceAdmissionError
                if response["resource"]
                else OperationDomainValidationError
            )
            raise error_type(
                location=("smtlib",),
                code=response["code"],
                message=response["message"],
            )
        result = SmtSolveResult.model_validate(
            {"source": request.model_dump(mode="json"), **response}
        )
        if result.exhausted is not None and result.source.logic != SmtLogic.QF_NRA:
            _raise_exhaustion(result.exhausted)
        _require_execution_deadline(deadline, "after SMT result projection")
        return result
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc


def solve_smt(request: SmtSolveRequest) -> SmtSolveResult:
    """Solve one query in a killable owner-local Z3 worker process."""

    return _run_smt_worker(request)


__all__ = [
    "SmtLogic",
    "SmtSolveRequest",
    "SmtSolveResult",
    "_UnknownResource",
    "_classify_exhaustion",
    "_execution_deadline",
    "_is_smtlib_source_diagnostic",
    "_project_unknown",
    "_require_execution_deadline",
    "_run_smt_worker",
    "_solve_smt_kernel",
    "_solver_settings",
    "_tokenize_smtlib",
    "_top_level_smtlib_commands",
    "solve_smt",
]
