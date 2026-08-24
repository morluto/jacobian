"""Stateless logic operations with domain-owned contracts and kernels."""

from __future__ import annotations

import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_MAX_VARIABLES = 1_024
_MAX_CLAUSES = 8_192
_MAX_LITERALS = 32_768
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
_MAX_LPR_STEPS = 2_048
_MAX_LPR_CLAUSE_WIDTH = 128
_MAX_LPR_HINT_IDS = 8_192
_MAX_LPR_REPLAY_WORK = 2_000_000
_LPR_WALL_SECONDS = 10
_LPR_HEAP_MEBIBYTES = 64
_LPR_STACK_MEBIBYTES = 16
_LPR_ADDRESS_SPACE_BYTES = 128 * 1024 * 1024
_LPR_PROCESS_OUTPUT_BYTES = 16_384
_MAX_LPR_RESULT_BYTES = 9 * 1024 * 1024
_CAKE_LPR_MANIFEST = Path("/usr/local/share/jacobian/cake-lpr.manifest")
_CAKE_LPR_MANIFEST_CONTENT = (
    "format=jacobian.cake-lpr/v1\n"
    "upstream_commit=a36874a8b750b43fe4b385b8ddbf5b033e46a3fa\n"
    "basis_ffi.c=8e30d84fdcb2177aa5571d7fa6661a2fae5ecfd56baa0ce49c65f9233a9f87cb\n"
    "cake_lpr.S=2f3af32d55083839b3fa0e693afd817679c0b8944bef41def05a8b0ec72b7d4a\n"
)
_LEAN_TOOLCHAIN = "leanprover/lean4:v4.31.0"
_SUPPORTED_SMTLIB_COMMANDS = frozenset(
    {"set-logic", "declare-const", "declare-fun", "assert", "check-sat"}
)
_UnknownResource = Literal["time", "work", "memory"]


class CanonicalCnf(StrictModel):
    """A bounded canonical propositional formula value."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"variables": ["a", "b"], "clauses": [[-1, 2], [1]]}]
        }
    )

    variables: tuple[str, ...] = Field(
        max_length=_MAX_VARIABLES,
        description="Distinct variable names in ascending lexicographic order.",
        examples=[["a", "b"]],
    )
    clauses: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=_MAX_CLAUSES,
        description=(
            "Unique non-tautological clauses in canonical order. Literals are signed "
            "one-based indexes into variables and each clause is ordered by variable."
        ),
        examples=[[[-1, 2], [1]]],
    )

    @field_validator("variables")
    @classmethod
    def require_canonical_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name or len(name) > 128 for name in value):
            raise ValueError(
                "CNF variables must be nonempty names of at most 128 characters"
            )
        if len(set(value)) != len(value) or value != tuple(sorted(value)):
            raise ValueError("CNF variables must be distinct and sorted")
        return value

    @model_validator(mode="after")
    def require_canonical_clauses(self) -> Self:
        if sum(len(clause) for clause in self.clauses) > _MAX_LITERALS:
            raise ValueError(f"CNF may contain at most {_MAX_LITERALS} literals")
        try:
            normalized = tuple(
                _canonical_clause(clause, len(self.variables))
                for clause in self.clauses
            )
        except _TautologicalClauseError as exc:
            raise ValueError("CNF clauses must be non-tautological") from exc
        if self.clauses != tuple(sorted(set(normalized), key=_clause_sort_key)):
            raise ValueError("CNF clauses must be unique, non-tautological, and sorted")
        return self


class CnfCanonicalizeRequest(StrictModel):
    """Named clauses whose literals refer to the supplied variable order."""

    variable_names: tuple[str, ...] = Field(max_length=_MAX_VARIABLES)
    clauses: tuple[tuple[StrictInt, ...], ...] = Field(max_length=_MAX_CLAUSES)

    @model_validator(mode="after")
    def require_bounded_input(self) -> Self:
        if any(not name or len(name) > 128 for name in self.variable_names):
            raise ValueError(
                "CNF variable names must be nonempty and at most 128 characters"
            )
        if len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError("CNF variable names must be unique")
        if sum(len(clause) for clause in self.clauses) > _MAX_LITERALS:
            raise ValueError(f"CNF may contain at most {_MAX_LITERALS} literals")
        for clause in self.clauses:
            for literal in clause:
                if literal == 0 or abs(literal) > len(self.variable_names):
                    raise ValueError("CNF literal references an undeclared variable")
        return self


class CnfCanonicalizeResult(StrictModel):
    cnf: CanonicalCnf


class SatAssignmentCheckRequest(StrictModel):
    cnf: CanonicalCnf
    assignment: tuple[StrictBool, ...] = Field(max_length=_MAX_VARIABLES)

    @model_validator(mode="after")
    def require_total_assignment(self) -> Self:
        if len(self.assignment) != len(self.cnf.variables):
            raise ValueError("assignment must contain one Boolean per CNF variable")
        return self


class SatAssignmentCheckResult(StrictModel):
    satisfies: StrictBool
    first_unsatisfied_clause: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bind_failure_index(self) -> Self:
        if self.satisfies != (self.first_unsatisfied_clause is None):
            raise ValueError(
                "an assignment result must carry an index exactly when it fails"
            )
        return self


class SatSolveRequest(StrictModel):
    cnf: CanonicalCnf
    timeout_ms: StrictInt = Field(default=1_000, ge=1, le=10_000)


class SatSolveResult(StrictModel):
    outcome: Literal["SAT", "UNSAT", "UNKNOWN"]
    assignment: tuple[StrictBool, ...] | None = Field(
        default=None, max_length=_MAX_VARIABLES
    )
    exhausted: _UnknownResource | None = Field(default=None)
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_assignment_to_outcome(self) -> Self:
        if (self.outcome == "SAT") != (self.assignment is not None):
            raise ValueError("only a SAT result may carry an assignment")
        if self.exhausted is not None and self.outcome != "UNKNOWN":
            raise ValueError("only an UNKNOWN result may name an exhausted budget")
        return self


class LprPropagationHint(StrictModel):
    """One named LPR propagation check and its ordered unit hints."""

    clause_id: StrictInt = Field(
        ge=1,
        description=(
            "A currently live clause-ID label whose propagation is checked. "
            "Labels may be sparse solver-assigned values."
        ),
    )
    at_hint_clause_ids: tuple[StrictInt, ...] = Field(
        max_length=_MAX_LPR_HINT_IDS,
        description=(
            "Ordered positive clause IDs used for the propagation check of this "
            "named live clause."
        ),
    )

    @field_validator("at_hint_clause_ids")
    @classmethod
    def require_positive_hint_ids(
        cls, value: tuple[StrictInt, ...]
    ) -> tuple[StrictInt, ...]:
        if any(clause_id <= 0 for clause_id in value):
            raise ValueError("LPR hint clause IDs must be positive")
        return value


class LprAddition(StrictModel):
    """One source-bound ASCII LPR PR/RAT addition."""

    kind: Literal["addition"] = "addition"
    clause_id: StrictInt = Field(
        ge=1,
        description=(
            "A fresh positive clause-ID label above all canonical source clause "
            "numbers; it may be a sparse solver-assigned value and may not "
            "overwrite a live clause."
        ),
    )
    clause: tuple[StrictInt, ...] = Field(
        max_length=_MAX_LPR_CLAUSE_WIDTH,
        description="Ordered nonzero literals on the exact CNF variable axis.",
    )
    witness: tuple[StrictInt, ...] | None = Field(
        default=None,
        max_length=_MAX_LPR_CLAUSE_WIDTH,
        description=(
            "Optional ordered PR witness. When present, its first literal equals "
            "the added clause's first literal."
        ),
    )
    at_hint_clause_ids: tuple[StrictInt, ...] = Field(
        max_length=_MAX_LPR_HINT_IDS,
        description=(
            "Ordered currently live clause IDs for the addition's "
            "asymmetric-tautology check."
        ),
    )
    propagation_hints: tuple[LprPropagationHint, ...] = Field(
        max_length=_MAX_LPR_HINT_IDS,
        description=(
            "At most one propagation hint per currently live clause ID, in the "
            "order written to the LPR proof."
        ),
    )

    @field_validator("at_hint_clause_ids")
    @classmethod
    def require_positive_at_hint_ids(
        cls, value: tuple[StrictInt, ...]
    ) -> tuple[StrictInt, ...]:
        if any(clause_id <= 0 for clause_id in value):
            raise ValueError("LPR hint clause IDs must be positive")
        return value

    @model_validator(mode="after")
    def bind_witness_to_the_clause_pivot(self) -> Self:
        if self.witness is not None:
            if not self.clause:
                raise ValueError("an empty LPR clause may not carry a witness")
            if not self.witness or self.witness[0] != self.clause[0]:
                raise ValueError(
                    "an LPR witness must start with the added clause's pivot literal"
                )
        return self


class LprDeletion(StrictModel):
    """One ASCII LPR deletion of currently live clause IDs."""

    kind: Literal["deletion"] = "deletion"
    clause_ids: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=_MAX_LPR_HINT_IDS,
        description="Distinct currently live clause IDs to remove before the next step.",
    )

    @field_validator("clause_ids")
    @classmethod
    def require_positive_distinct_clause_ids(
        cls, value: tuple[StrictInt, ...]
    ) -> tuple[StrictInt, ...]:
        if any(clause_id <= 0 for clause_id in value):
            raise ValueError("deleted LPR clause IDs must be positive")
        if len(set(value)) != len(value):
            raise ValueError("one LPR deletion may not name a clause more than once")
        return value


LprStep = Annotated[LprAddition | LprDeletion, Field(discriminator="kind")]


class SatLprRefutation(StrictModel):
    """A bounded typed LPR/ASCII-v1 refutation, without checker syntax or flags."""

    profile: Literal["LPR_ASCII_V1"] = "LPR_ASCII_V1"
    steps: tuple[LprStep, ...] = Field(
        max_length=_MAX_LPR_STEPS,
        description=(
            "Ordered LPR additions and deletions. The checker uses canonical source "
            "clause IDs 1..m; every hint and deletion must name a currently live ID. "
            "The derived literal-inspection work must not exceed 2,000,000."
        ),
    )


class SatRefutationCheckRequest(StrictModel):
    cnf: CanonicalCnf
    refutation: SatLprRefutation = Field(
        description=(
            "One source-bound LPR/ASCII-v1 derivation. It uses the CNF's exact "
            "one-based canonical clause order and variable axis."
        )
    )

    @model_validator(mode="after")
    def require_source_bound_lpr_profile(self) -> Self:
        _validate_lpr_refutation(self.cnf, self.refutation)
        return self


class SatRefutationCheckResult(StrictModel):
    """A source-bound LPR replay outcome; only VALID_REFUTATION proves UNSAT."""

    outcome: Literal[
        "VALID_REFUTATION", "INVALID_REFUTATION", "UNAVAILABLE", "TIMEOUT", "ERROR"
    ]
    cnf: CanonicalCnf
    refutation: SatLprRefutation
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_execution_detail(self) -> Self:
        if (self.outcome == "VALID_REFUTATION") != (self.detail is None):
            raise ValueError("only a valid refutation may omit its outcome detail")
        return self


def _require_live_lpr_ids(
    clause_ids: tuple[int, ...],
    live_clause_widths: dict[int, int],
    label: str,
) -> None:
    missing = next(
        (clause_id for clause_id in clause_ids if clause_id not in live_clause_widths),
        None,
    )
    if missing is not None:
        raise ValueError(f"{label} references non-live clause ID {missing}")


def _require_lpr_literal_axis(
    literals: tuple[int, ...], variable_count: int, label: str
) -> None:
    if any(literal == 0 or abs(literal) > variable_count for literal in literals):
        raise ValueError(f"{label} literal is outside the CNF variable axis")


def _lpr_addition_work(
    step: LprAddition,
    live_clause_widths: dict[int, int],
) -> int:
    candidate_width = len(step.clause) + len(step.witness or ())
    inspection_factor = candidate_width + 1
    total = sum(width + 1 for width in live_clause_widths.values()) * inspection_factor
    total += sum(
        (live_clause_widths[clause_id] + 1) * inspection_factor
        for clause_id in step.at_hint_clause_ids
    )
    return total + sum(
        (live_clause_widths[hint.clause_id] + 1) * inspection_factor
        + sum(
            (live_clause_widths[at_clause_id] + 1) * inspection_factor
            for at_clause_id in hint.at_hint_clause_ids
        )
        for hint in step.propagation_hints
    )


def _validate_lpr_addition(
    step: LprAddition,
    *,
    variable_count: int,
    source_clause_count: int,
    live_clause_widths: dict[int, int],
) -> None:
    if step.clause_id <= source_clause_count:
        raise ValueError(
            "LPR additions must use IDs after the canonical source clauses"
        )
    if step.clause_id in live_clause_widths:
        raise ValueError("LPR additions may not overwrite a live clause ID")
    _require_lpr_literal_axis(step.clause, variable_count, "LPR clause")
    if step.witness is not None:
        _require_lpr_literal_axis(step.witness, variable_count, "LPR witness")
    _require_live_lpr_ids(
        step.at_hint_clause_ids, live_clause_widths, "LPR asymmetric-tautology hint"
    )
    if len({hint.clause_id for hint in step.propagation_hints}) != len(
        step.propagation_hints
    ):
        raise ValueError("LPR propagation hint clause IDs must be unique")
    for hint in step.propagation_hints:
        _require_live_lpr_ids(
            (hint.clause_id,), live_clause_widths, "LPR propagation hint"
        )
        _require_live_lpr_ids(
            hint.at_hint_clause_ids,
            live_clause_widths,
            "LPR propagation asymmetric-tautology hint",
        )


def _validate_lpr_refutation(cnf: CanonicalCnf, refutation: SatLprRefutation) -> None:
    live_clause_widths = {
        index: len(clause) for index, clause in enumerate(cnf.clauses, 1)
    }
    source_clause_count = len(live_clause_widths)
    total_work = 0
    for step in refutation.steps:
        if isinstance(step, LprDeletion):
            _require_live_lpr_ids(step.clause_ids, live_clause_widths, "LPR deletion")
            for clause_id in step.clause_ids:
                del live_clause_widths[clause_id]
            continue
        _validate_lpr_addition(
            step,
            variable_count=len(cnf.variables),
            source_clause_count=source_clause_count,
            live_clause_widths=live_clause_widths,
        )
        total_work += _lpr_addition_work(step, live_clause_widths)
        if total_work > _MAX_LPR_REPLAY_WORK:
            raise ValueError(
                "LPR replay exceeds the declared literal-inspection work bound"
            )
        live_clause_widths[step.clause_id] = len(step.clause)
    echoed_result = {
        "outcome": "INVALID_REFUTATION",
        "cnf": cnf.model_dump(mode="json"),
        "refutation": refutation.model_dump(mode="json"),
        "detail": "x" * 1_024,
    }
    if len(canonicalize_json(echoed_result)) > _MAX_LPR_RESULT_BYTES:
        raise ValueError("LPR refutation exceeds the source-bound result limit")


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
            f"or indexed bit-vector spelling at most {_MAX_SMTLIB_NUMERAL_DIGITS} digits."
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
            raise ValueError("SMT-LIB input must be ASCII") from exc
        if len(encoded) > _MAX_SMTLIB_BYTES:
            raise ValueError("SMT-LIB input exceeds the byte limit")
        structure = _smtlib_structure(self.smtlib)
        if structure.max_depth > _MAX_SMTLIB_DEPTH:
            raise ValueError(
                f"SMT-LIB nesting exceeds the maximum term depth of {_MAX_SMTLIB_DEPTH}"
            )
        if structure.compound_terms > _MAX_SMTLIB_TERMS:
            raise ValueError(
                f"SMT-LIB exceeds the maximum of {_MAX_SMTLIB_TERMS} compound terms"
            )
        if structure.numeral_digits > _MAX_SMTLIB_NUMERAL_DIGITS:
            raise ValueError(
                "SMT-LIB contains a numeral wider than "
                f"{_MAX_SMTLIB_NUMERAL_DIGITS} digits"
            )
        commands = _top_level_smtlib_commands(self.smtlib)
        declarations = sum(
            1
            for command in commands
            if command[:1] in (("declare-const",), ("declare-fun",))
        )
        if declarations > _MAX_SMTLIB_DECLARATIONS:
            raise ValueError(
                f"SMT-LIB declares more than {_MAX_SMTLIB_DECLARATIONS} symbols"
            )
        logic_commands = tuple(
            command for command in commands if command[:1] == ("set-logic",)
        )
        if logic_commands != (("set-logic", self.logic.value),):
            raise ValueError("SMT-LIB input must declare the requested logic")
        if commands.count(("check-sat",)) != 1:
            raise ValueError("SMT-LIB input must contain exactly one check-sat command")
        if commands[-1:] != (("check-sat",),):
            raise ValueError("SMT-LIB input must end with its check-sat command")
        for command in commands:
            if command and command[0] not in _SUPPORTED_SMTLIB_COMMANDS:
                raise ValueError(f"unsupported SMT-LIB command: {command[0]}")
        return self


class SmtSolveResult(StrictModel):
    outcome: Literal["SAT", "UNSAT", "UNKNOWN"]
    model_smtlib: str | None = Field(default=None, max_length=_MAX_MODEL_BYTES)
    exhausted: _UnknownResource | None = Field(default=None)
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_model_to_outcome(self) -> Self:
        if (self.outcome == "SAT") != (self.model_smtlib is not None):
            raise ValueError("only a SAT result may carry a model")
        if self.exhausted is not None and self.outcome != "UNKNOWN":
            raise ValueError("only an UNKNOWN result may name an exhausted budget")
        return self


class LeanDiagnostic(StrictModel):
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str = Field(min_length=1, max_length=4_000)


class LeanCheckRequest(StrictModel):
    """A source snippet checked in the service image's fixed Lean environment."""

    source: str = Field(
        min_length=1,
        max_length=32_000,
        description="A self-contained Lean source snippet for the fixed service toolchain.",
        examples=["example : True := by trivial"],
    )
    timeout_seconds: StrictInt = Field(default=10, ge=1, le=30)


class LeanCheckResult(StrictModel):
    outcome: Literal["ELABORATED", "REJECTED", "UNAVAILABLE", "TIMEOUT", "ERROR"]
    diagnostics: tuple[LeanDiagnostic, ...] = Field(max_length=64)
    detail: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def bind_diagnostics_to_outcome(self) -> Self:
        if self.outcome == "ELABORATED" and any(
            diagnostic.severity == "ERROR" for diagnostic in self.diagnostics
        ):
            raise ValueError("an elaborated source result cannot contain an error")
        return self


def canonicalize_cnf(request: CnfCanonicalizeRequest) -> CnfCanonicalizeResult:
    """Return the unique canonical CNF for one named clause collection."""

    indexed_names = tuple(enumerate(request.variable_names, start=1))
    sorted_names = tuple(sorted(indexed_names, key=lambda item: item[1]))
    old_to_new = {old: new for new, (old, _name) in enumerate(sorted_names, start=1)}
    clauses: set[tuple[int, ...]] = set()
    for clause in request.clauses:
        remapped = tuple(
            old_to_new[abs(literal)] if literal > 0 else -old_to_new[abs(literal)]
            for literal in clause
        )
        try:
            clauses.add(_canonical_clause(remapped, len(sorted_names)))
        except _TautologicalClauseError:
            continue
    return CnfCanonicalizeResult(
        cnf=CanonicalCnf(
            variables=tuple(name for _old, name in sorted_names),
            clauses=tuple(sorted(clauses, key=_clause_sort_key)),
        )
    )


def check_sat_assignment(
    request: SatAssignmentCheckRequest,
) -> SatAssignmentCheckResult:
    """Evaluate a total Boolean assignment directly against one canonical CNF."""

    for index, clause in enumerate(request.cnf.clauses):
        if not any(
            request.assignment[abs(literal) - 1]
            if literal > 0
            else not request.assignment[abs(literal) - 1]
            for literal in clause
        ):
            return SatAssignmentCheckResult(
                satisfies=False, first_unsatisfied_clause=index
            )
    return SatAssignmentCheckResult(satisfies=True)


_EXHAUSTION_DETAILS: dict[_UnknownResource, str] = {
    "work": "the bounded solver work budget was exhausted",
    "memory": "the bounded solver memory budget was exhausted",
    "time": "the bounded solver time budget was exhausted",
}


def _classify_exhaustion(message: str) -> _UnknownResource | None:
    """Classify one Z3 reason or exception message onto the exhausted budgets."""

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


def solve_sat(request: SatSolveRequest) -> SatSolveResult:
    """Solve one bounded canonical CNF through the maintained Z3 Python binding."""

    import z3  # type: ignore[import-untyped]

    variables = tuple(z3.Bool(name) for name in request.cnf.variables)
    solver = z3.Solver()
    solver.set(**_solver_settings(request.timeout_ms))
    try:
        for clause in request.cnf.clauses:
            terms = tuple(
                variables[abs(literal) - 1]
                if literal > 0
                else z3.Not(variables[abs(literal) - 1])
                for literal in clause
            )
            solver.add(z3.Or(*terms))
        outcome = solver.check()
        if outcome == z3.sat:
            model = solver.model()
            assignment = tuple(
                z3.is_true(model.eval(variable, model_completion=True))
                for variable in variables
            )
            return SatSolveResult(outcome="SAT", assignment=assignment)
        if outcome == z3.unsat:
            return SatSolveResult(outcome="UNSAT")
    except z3.Z3Exception as exc:
        exhausted = _classify_exhaustion(str(exc))
        if exhausted is not None:
            return SatSolveResult(
                outcome="UNKNOWN",
                exhausted=exhausted,
                detail=_EXHAUSTION_DETAILS[exhausted],
            )
        detail = f"the Z3 backend failed during the bounded solve: {exc}"
        return SatSolveResult(outcome="UNKNOWN", detail=detail[:1_024])
    exhausted, detail = _project_unknown(solver.reason_unknown())
    return SatSolveResult(outcome="UNKNOWN", exhausted=exhausted, detail=detail)


def _dimacs_cnf(cnf: CanonicalCnf) -> bytes:
    lines = [f"p cnf {len(cnf.variables)} {len(cnf.clauses)}"]
    lines.extend(
        " ".join(str(literal) for literal in clause) + " 0" for clause in cnf.clauses
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def _ascii_lpr(refutation: SatLprRefutation) -> bytes:
    lines: list[str] = []
    for step in refutation.steps:
        if isinstance(step, LprDeletion):
            lines.append("0 d " + " ".join(map(str, step.clause_ids)) + " 0")
            continue
        fields = [str(step.clause_id), *(str(literal) for literal in step.clause)]
        if step.witness is not None:
            fields.extend(str(literal) for literal in step.witness)
        fields.append("0")
        fields.extend(str(clause_id) for clause_id in step.at_hint_clause_ids)
        if not step.propagation_hints:
            fields.append("0")
        else:
            fields.append(str(-step.propagation_hints[0].clause_id))
            for index, hint in enumerate(step.propagation_hints):
                fields.extend(str(clause_id) for clause_id in hint.at_hint_clause_ids)
                fields.append(
                    str(
                        -step.propagation_hints[index + 1].clause_id
                        if index + 1 < len(step.propagation_hints)
                        else 0
                    )
                )
        lines.append(" ".join(fields))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("ascii")


def _cake_lpr_is_supported(executable: str) -> bool:
    """Require the source-pinned Cake LPR provider installed by our OCI image."""

    try:
        return (
            Path(executable).resolve() == Path("/usr/local/bin/cake_lpr")
            and _CAKE_LPR_MANIFEST.read_text(encoding="ascii")
            == _CAKE_LPR_MANIFEST_CONTENT
        )
    except OSError:
        return False


def check_sat_refutation(
    request: SatRefutationCheckRequest,
) -> SatRefutationCheckResult:
    """Replay one typed LPR refutation through the pinned CakeML checker."""

    def unavailable(detail: str) -> SatRefutationCheckResult:
        return SatRefutationCheckResult(
            outcome="UNAVAILABLE",
            cnf=request.cnf,
            refutation=request.refutation,
            detail=detail,
        )

    executable = shutil.which("cake_lpr")
    if executable is None or not _cake_lpr_is_supported(executable):
        return unavailable(
            "The source-pinned Cake LPR backend is available only in the Linux service image."
        )
    resolved = str(Path(executable).resolve())
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-lpr-") as directory:
            formula_path = Path(directory) / "formula.cnf"
            proof_path = Path(directory) / "proof.lpr"
            formula_path.write_bytes(_dimacs_cnf(request.cnf))
            proof_path.write_bytes(_ascii_lpr(request.refutation))
            completed = run_bounded_process(
                [
                    resolved,
                    f"--CML_HEAP_SIZE={_LPR_HEAP_MEBIBYTES}",
                    f"--CML_STACK_SIZE={_LPR_STACK_MEBIBYTES}",
                    str(formula_path),
                    str(proof_path),
                ],
                input_bytes=b"",
                timeout_seconds=float(_LPR_WALL_SECONDS),
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_LPR_PROCESS_OUTPUT_BYTES,
                stderr_limit=_LPR_PROCESS_OUTPUT_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=_LPR_WALL_SECONDS,
                    address_space_bytes=_LPR_ADDRESS_SPACE_BYTES,
                    file_size_bytes=1_024 * 1_024,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError:
        return unavailable("The fixed Cake LPR backend could not be started.")
    outcome: Literal["INVALID_REFUTATION", "UNAVAILABLE", "TIMEOUT", "ERROR"]
    if completed.timed_out:
        outcome, detail = "TIMEOUT", "Cake LPR exceeded the declared wall-time limit."
    elif completed.cancelled:
        outcome, detail = "ERROR", "Cake LPR execution was cancelled."
    elif completed.stdout_exceeded or completed.stderr_exceeded:
        outcome, detail = "ERROR", "Cake LPR exceeded the diagnostic-output limit."
    elif (
        completed.returncode == 0
        and completed.stdout == b"s VERIFIED UNSAT\n"
        and not completed.stderr
    ):
        return SatRefutationCheckResult(
            outcome="VALID_REFUTATION",
            cnf=request.cnf,
            refutation=request.refutation,
        )
    elif (
        completed.returncode == 0
        and not completed.stdout
        and completed.stderr.startswith(b"c ")
        and completed.stderr.endswith(b"\n")
    ):
        outcome, detail = (
            "INVALID_REFUTATION",
            "The typed LPR refutation does not derive contradiction from this CNF.",
        )
    else:
        # Unexpected process output, including resource failures, is never a
        # mathematical negative verdict.
        outcome, detail = (
            "ERROR",
            "Cake LPR did not produce its exact verified-UNSAT verdict.",
        )
    return SatRefutationCheckResult(
        outcome=outcome,
        cnf=request.cnf,
        refutation=request.refutation,
        detail=detail,
    )


def solve_smt(request: SmtSolveRequest) -> SmtSolveResult:
    """Solve one bounded SMT-LIB query through the maintained Z3 Python binding."""

    import z3

    try:
        assertions = z3.parse_smt2_string(request.smtlib)
    except z3.Z3Exception as exc:
        raise ValueError(
            "SMT-LIB input could not be parsed by the declared logic"
        ) from exc
    solver = z3.SolverFor(request.logic.value)
    solver.set(**_solver_settings(request.timeout_ms))
    try:
        solver.add(assertions)
        outcome = solver.check()
        if outcome == z3.sat:
            model = solver.model().sexpr()
            if len(model.encode("utf-8")) > _MAX_MODEL_BYTES:
                return SmtSolveResult(
                    outcome="UNKNOWN",
                    detail="the satisfying model exceeds the bounded result limit",
                )
            return SmtSolveResult(outcome="SAT", model_smtlib=model)
        if outcome == z3.unsat:
            return SmtSolveResult(outcome="UNSAT")
    except z3.Z3Exception as exc:
        exhausted = _classify_exhaustion(str(exc))
        if exhausted is not None:
            return SmtSolveResult(
                outcome="UNKNOWN",
                exhausted=exhausted,
                detail=_EXHAUSTION_DETAILS[exhausted],
            )
        detail = f"the Z3 backend failed during the bounded solve: {exc}"
        return SmtSolveResult(outcome="UNKNOWN", detail=detail[:1_024])
    exhausted, detail = _project_unknown(solver.reason_unknown())
    return SmtSolveResult(outcome="UNKNOWN", exhausted=exhausted, detail=detail)


def check_lean_source(request: LeanCheckRequest) -> LeanCheckResult:
    """Elaborate one source snippet in a temporary request-scoped directory."""

    executable = shutil.which("lean")
    if executable is None:
        return LeanCheckResult(
            outcome="UNAVAILABLE",
            diagnostics=(),
            detail="The fixed Lean environment is not installed.",
        )
    with tempfile.TemporaryDirectory(prefix="jacobian-lean-") as directory:
        source_path = Path(directory) / "Snippet.lean"
        source_path.write_text(request.source, encoding="utf-8")
        try:
            completed = run_bounded_process(
                [executable, str(source_path)],
                input_bytes=b"",
                timeout_seconds=float(request.timeout_seconds),
                environment=worker_environment(
                    extra_variables=("PATH", "ELAN_HOME"),
                    overrides={
                        "ELAN_TOOLCHAIN": _LEAN_TOOLCHAIN,
                        "ELAN_HOME": os.environ.get(
                            "ELAN_HOME", str(Path.home() / ".elan")
                        ),
                    },
                    locale="C.UTF-8",
                ),
                stdout_limit=64_000,
                stderr_limit=64_000,
                cwd=directory,
            )
        except OSError:
            return LeanCheckResult(
                outcome="UNAVAILABLE",
                diagnostics=(),
                detail="The fixed Lean environment could not be started.",
            )
    if completed.timed_out:
        return LeanCheckResult(
            outcome="TIMEOUT",
            diagnostics=(),
            detail="Lean exceeded the declared time limit.",
        )
    if completed.cancelled or completed.stdout_exceeded or completed.stderr_exceeded:
        return LeanCheckResult(
            outcome="ERROR",
            diagnostics=(),
            detail="Lean exceeded a process resource limit.",
        )
    diagnostics = _lean_diagnostics(completed.stdout + completed.stderr)
    if completed.returncode == 0:
        return LeanCheckResult(outcome="ELABORATED", diagnostics=diagnostics)
    return LeanCheckResult(outcome="REJECTED", diagnostics=diagnostics)


def _lean_diagnostics(output: bytes) -> tuple[LeanDiagnostic, ...]:
    text = output.decode("utf-8", errors="replace")
    if not text.strip():
        return ()
    diagnostics: list[LeanDiagnostic] = []
    for line in text.splitlines():
        message = line.strip()
        if not message:
            continue
        lowered = message.lower()
        severity: Literal["ERROR", "WARNING", "INFO"] = (
            "ERROR"
            if "error:" in lowered
            else "WARNING"
            if "warning:" in lowered
            else "INFO"
        )
        diagnostics.append(LeanDiagnostic(severity=severity, message=message[:4_000]))
        if len(diagnostics) == 64:
            break
    return tuple(diagnostics)


class _TautologicalClauseError(Exception):
    pass


def _canonical_clause(clause: tuple[int, ...], variable_count: int) -> tuple[int, ...]:
    literals: set[int] = set()
    for literal in clause:
        if literal == 0 or abs(literal) > variable_count:
            raise ValueError("CNF literal references an undeclared variable")
        if -literal in literals:
            raise _TautologicalClauseError
        literals.add(literal)
    return tuple(sorted(literals, key=lambda literal: (abs(literal), literal > 0)))


def _clause_sort_key(clause: tuple[int, ...]) -> tuple[tuple[int, bool], ...]:
    return tuple((abs(literal), literal > 0) for literal in clause)


LOGIC_OPERATIONS = (
    MathTool(
        operation_id="sat.cnf.canonicalize",
        version="1",
        title="Canonicalize a bounded named CNF",
        description="Return one canonical CNF; no source, identifier, or artifact is retained.",
        request_type=CnfCanonicalizeRequest,
        result_type=CnfCanonicalizeResult,
        run=canonicalize_cnf,
        tags=("sat", "cnf", "canonical"),
        examples=(
            example(
                "two_variables",
                "Normalize a small named CNF.",
                {"variable_names": ["b", "a"], "clauses": [[1, -2], [2]]},
            ),
        ),
    ),
    MathTool(
        operation_id="sat.assignment.check",
        version="1",
        title="Check a total SAT assignment",
        description="Evaluate one complete Boolean assignment against one canonical CNF.",
        request_type=SatAssignmentCheckRequest,
        result_type=SatAssignmentCheckResult,
        run=check_sat_assignment,
        tags=("sat", "cnf", "assignment", "predicate"),
        examples=(
            example(
                "satisfying_assignment",
                "Check a total assignment against a canonical CNF.",
                {
                    "cnf": {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]},
                    "assignment": [True, True],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="sat.solve",
        version="2",
        title="Solve a bounded CNF",
        description="Run the maintained Z3 Python binding on one canonical CNF.",
        request_type=SatSolveRequest,
        result_type=SatSolveResult,
        run=solve_sat,
        tags=("sat", "cnf", "solve", "z3"),
        examples=(
            example(
                "two_variable_cnf",
                "Solve a small canonical CNF.",
                {"cnf": {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="sat.refutation.check",
        version="1",
        title="Check a bounded LPR SAT refutation",
        description=(
            "Replay one typed LPR/ASCII-v1 refutation against its exact canonical "
            "CNF through the source-pinned CakeML checker. Only VALID_REFUTATION "
            "establishes UNSAT; unavailable or failed replay is a non-conclusion."
        ),
        request_type=SatRefutationCheckRequest,
        result_type=SatRefutationCheckResult,
        run=check_sat_refutation,
        tags=("sat", "cnf", "lpr", "refutation", "certificate"),
        examples=(
            example(
                "unit_contradiction",
                "Check an LPR empty-clause derivation from two contradictory units.",
                {
                    "cnf": {"variables": ["x"], "clauses": [[-1], [1]]},
                    "refutation": {
                        "profile": "LPR_ASCII_V1",
                        "steps": [
                            {
                                "kind": "addition",
                                "clause_id": 3,
                                "clause": [],
                                "at_hint_clause_ids": [1, 2],
                                "propagation_hints": [],
                            }
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="smt.solve",
        version="3",
        title="Solve a bounded SMT-LIB query",
        description="Run the maintained Z3 Python binding on one QF SMT-LIB query.",
        request_type=SmtSolveRequest,
        result_type=SmtSolveResult,
        run=solve_smt,
        tags=("smt", "solve", "smtlib", "z3"),
        examples=(
            example(
                "positive_integer",
                "Solve a bounded quantifier-free linear-integer query.",
                {
                    "logic": "QF_LIA",
                    "smtlib": "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lean.check",
        version="1",
        title="Check a bounded Lean source snippet",
        description="Elaborate one source snippet in the fixed Lean service environment and return typed diagnostics.",
        request_type=LeanCheckRequest,
        result_type=LeanCheckResult,
        run=check_lean_source,
        tags=("lean", "elaboration", "source", "bounded"),
        examples=(
            example(
                "trivial_proposition",
                "Elaborate a self-contained proof of True.",
                {"source": "example : True := by trivial"},
            ),
        ),
    ),
)

__all__ = [
    "LOGIC_OPERATIONS",
    "CanonicalCnf",
    "CnfCanonicalizeRequest",
    "CnfCanonicalizeResult",
    "LeanCheckRequest",
    "LeanCheckResult",
    "LprAddition",
    "LprDeletion",
    "LprPropagationHint",
    "LprStep",
    "SatAssignmentCheckRequest",
    "SatAssignmentCheckResult",
    "SatLprRefutation",
    "SatRefutationCheckRequest",
    "SatRefutationCheckResult",
    "SatSolveRequest",
    "SatSolveResult",
    "SmtLogic",
    "SmtSolveRequest",
    "SmtSolveResult",
    "canonicalize_cnf",
    "check_lean_source",
    "check_sat_assignment",
    "check_sat_refutation",
    "solve_sat",
    "solve_smt",
]
