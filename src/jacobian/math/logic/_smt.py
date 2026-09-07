"""Bounded SMT-LIB solver contracts and direct Z3 kernel."""

from __future__ import annotations

import json
import math
import re
import sys
import time
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, NamedTuple, Self

from pydantic import (
    Field,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.process import (
    ProcessResourceLimits,
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
# exhaustion surfaces as typed UNKNOWN instead of host memory pressure.
_SOLVER_RLIMIT = 20_000_000
_SOLVER_MAX_MEMORY_MB = 1024
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
_UnknownResource = Literal["time", "work", "memory"]


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


_Z3_SOURCE_DIAGNOSTIC = re.compile(r'\(error "line \d+ column \d+: ')


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
    logic: SmtLogic
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
    timeout_ms: StrictInt = Field(default=1_000, ge=1, le=10_000)

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
                raise _validation_error(
                    "logic.smtlib_command",
                    f"unsupported SMT-LIB command: {command[0]}; supported commands are "
                    f"{_SUPPORTED_SMTLIB_COMMANDS_DESCRIPTION}",
                )
        return self


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
            "independently verifiable model encoding."
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
        if self.exhausted is not None and self.outcome != "UNKNOWN":
            raise _validation_error(
                "logic.unknown_exhaustion",
                "only an UNKNOWN result may name an exhausted budget",
            )
        return self


_EXHAUSTION_DETAILS: dict[_UnknownResource, str] = {
    "work": "the bounded solver work budget was exhausted",
    "memory": "the bounded solver memory budget was exhausted",
    "time": "the bounded solver time budget was exhausted",
}


def _classify_exhaustion(message: str) -> _UnknownResource | None:
    """Classify one Z3 reason or exception message onto the exhausted budgets.

    Exhaustion keywords classify only backend conditions, which carry no
    source locator. A message containing a located ``(error "line ...
    column ...: ...")`` diagnostic is never classified as exhaustion, even
    when its text mentions a resource keyword: the diagnostic quotes
    caller-controlled source spellings, so an undeclared identifier named
    ``memory`` or a comment mentioning ``timeout`` must not report an
    exhausted budget.
    """

    if _Z3_SOURCE_DIAGNOSTIC.search(message) is not None:
        return None
    lowered = message.strip().lower()
    if "resource limit" in lowered or "canceled" in lowered:
        return "work"
    if "memory" in lowered:
        return "memory"
    if "timeout" in lowered or "time limit" in lowered:
        return "time"
    return None


def _project_unknown(reason: str | None) -> tuple[_UnknownResource | None, str]:
    """Project one Z3 unknown reason onto the typed exhausted-budget taxonomy."""

    text = (reason or "").strip()
    classified = _classify_exhaustion(text)
    if classified is not None:
        return classified, _EXHAUSTION_DETAILS[classified]
    if not text:
        return None, "the solver returned no completeness evidence"
    return None, text[:1_024]


def _solver_settings(timeout_ms: int) -> dict[str, int]:
    """Return the full request-scoped Z3 budget: wall time, work, and memory."""

    return {
        "timeout": timeout_ms,
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


def _probe_declared_logic(
    assertions: Any, logic: str, z3: Any, *, simplify_arithmetic: bool = True
) -> None:
    """Reject parsed assertions outside the advertised quantifier-free fragment."""

    assertion_tuple = tuple(assertions)
    if not assertion_tuple:
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


def _result(request: SmtSolveRequest, **values: object) -> SmtSolveResult:
    """Bind every projected worker outcome to its exact admitted source."""

    return SmtSolveResult.model_validate(
        {"source": request.model_dump(mode="json"), **values}
    )


def _time_exhausted_result(request: SmtSolveRequest) -> SmtSolveResult:
    """Project expiry of the admitted owner envelope as a non-conclusion."""

    return _result(
        request,
        outcome="UNKNOWN",
        exhausted="time",
        detail=_EXHAUSTION_DETAILS["time"],
    )


def _solve_smt_kernel(
    *, logic: str, smtlib: str, timeout_ms: int
) -> dict[str, str | None]:
    """Run one complete Z3 lifecycle inside the owned worker process."""

    try:
        import z3
    except (ImportError, OSError) as exc:
        return {
            "outcome": "UNKNOWN",
            "model_smtlib": None,
            "exhausted": None,
            "detail": f"the Z3 backend could not initialize: {exc}"[:1_024],
        }

    try:
        assertions = z3.parse_smt2_string(smtlib)
        try:
            _probe_declared_logic(assertions, logic, z3)
        except ValueError as exc:
            return {"kind": "invalid", "detail": str(exc)[:1_024]}
        solver = z3.SolverFor(logic)
        solver.add(assertions)
        solver.set(**_solver_settings(timeout_ms))
        outcome = solver.check()
        if outcome == z3.sat:
            model = solver.model()
            if not all(
                z3.is_true(model.eval(assertion, model_completion=True))
                for assertion in assertions
            ):
                return {
                    "outcome": "UNKNOWN",
                    "model_smtlib": None,
                    "exhausted": None,
                    "detail": (
                        "the Z3 backend returned a model that does not satisfy the "
                        "admitted SMT-LIB assertions"
                    ),
                }
            model_smtlib = model.sexpr()
            if len(model_smtlib.encode("utf-8")) > _MAX_MODEL_BYTES:
                return {
                    "outcome": "UNKNOWN",
                    "model_smtlib": None,
                    "exhausted": None,
                    "detail": "the satisfying model exceeds the bounded result limit",
                }
            return {
                "outcome": "SAT",
                "model_smtlib": model_smtlib,
                "exhausted": None,
                "detail": None,
            }
        if outcome == z3.unsat:
            return {
                "outcome": "UNSAT",
                "model_smtlib": None,
                "exhausted": None,
                "detail": None,
            }
    except (OSError, z3.Z3Exception) as exc:
        if isinstance(exc, z3.Z3Exception) and _is_smtlib_source_diagnostic(exc):
            return {
                "kind": "invalid",
                "detail": "SMT-LIB source could not be parsed as SMT-LIB",
            }
        exhausted = _classify_exhaustion(str(exc))
        if exhausted is not None:
            return {
                "outcome": "UNKNOWN",
                "model_smtlib": None,
                "exhausted": exhausted,
                "detail": _EXHAUSTION_DETAILS[exhausted],
            }
        detail = f"the Z3 backend failed during the bounded solve: {exc}"
        return {
            "outcome": "UNKNOWN",
            "model_smtlib": None,
            "exhausted": None,
            "detail": detail[:1_024],
        }
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
    except OSError:
        _require_execution_deadline(deadline, "after SMT worker startup")
        return _result(
            request,
            outcome="UNKNOWN",
            detail="the bounded Z3 worker could not be started",
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "request deadline expired during SMT worker"
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError("request cancelled during SMT worker")
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _result(
            request,
            outcome="UNKNOWN",
            detail="the bounded Z3 worker exceeded its output limit",
        )
    if completed.returncode != 0:
        return _result(
            request,
            outcome="UNKNOWN",
            detail="the bounded Z3 worker failed before returning a result",
        )
    _require_execution_deadline(deadline, "after SMT worker")
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
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
        result = SmtSolveResult.model_validate(
            {"source": request.model_dump(mode="json"), **response}
        )
        _require_execution_deadline(deadline, "after SMT result projection")
        return result
    except OperationDomainValidationError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return _result(
            request,
            outcome="UNKNOWN",
            detail="the bounded Z3 worker returned malformed output",
        )


def solve_smt(request: SmtSolveRequest) -> SmtSolveResult:
    """Solve one query in a killable owner-local Z3 worker process."""

    return _run_smt_worker(request)


__all__ = [
    "_EXHAUSTION_DETAILS",
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
