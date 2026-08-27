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
from typing import Literal, NamedTuple, Self

from pydantic import (
    Field,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
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
_UnknownResource = Literal["time", "work", "memory"]


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class SmtLogic(StrEnum):
    QF_UF = "QF_UF"
    QF_LIA = "QF_LIA"
    QF_LRA = "QF_LRA"


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
            "The source is structurally admitted before execution; Z3 parsing and "
            "solving occur within the operation's bounded execution envelope."
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
                    "logic.smtlib_command", f"unsupported SMT-LIB command: {command[0]}"
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
        import z3  # type: ignore[import-untyped]
    except (ImportError, OSError) as exc:
        return {
            "outcome": "UNKNOWN",
            "model_smtlib": None,
            "exhausted": None,
            "detail": f"the Z3 backend could not initialize: {exc}"[:1_024],
        }

    try:
        assertions = z3.parse_smt2_string(smtlib)
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

    deadline = time.monotonic() + (request.timeout_ms / 1_000)
    try:
        # The isolated worker has no reason to inherit the checkout as its
        # current directory.  Its input arrives only through stdin and its
        # only accepted output is the bounded JSON response on stdout.
        with TemporaryDirectory(prefix="jacobian-smt-") as worker_directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _time_exhausted_result(request)
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
    except OSError:
        return _result(
            request,
            outcome="UNKNOWN",
            detail="the bounded Z3 worker could not be started",
        )
    if completed.timed_out:
        return _time_exhausted_result(request)
    if completed.cancelled:
        return _result(
            request, outcome="UNKNOWN", detail="the bounded Z3 worker was cancelled"
        )
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
    if time.monotonic() >= deadline:
        return _time_exhausted_result(request)
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        result = SmtSolveResult.model_validate(
            {"source": request.model_dump(mode="json"), **response}
        )
        return (
            result if time.monotonic() < deadline else _time_exhausted_result(request)
        )
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
    "_is_smtlib_source_diagnostic",
    "_project_unknown",
    "_run_smt_worker",
    "_solve_smt_kernel",
    "_solver_settings",
    "_tokenize_smtlib",
    "_top_level_smtlib_commands",
    "solve_smt",
]
