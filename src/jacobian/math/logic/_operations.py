"""Stateless logic operations with domain-owned contracts and kernels."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.process import run_bounded_process, worker_environment

_MAX_VARIABLES = 1_024
_MAX_CLAUSES = 8_192
_MAX_LITERALS = 32_768
_MAX_SMTLIB_BYTES = 128_000
_MAX_MODEL_BYTES = 64_000
_LEAN_TOOLCHAIN = "leanprover/lean4:v4.31.0"
_MATHLIB_REVISION: Literal["fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"] = (
    "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
)
_MATHLIB_MANIFEST_SHA256 = (
    "2e3e4f23e695c64bd3eac9d210a7e0aa6ce9a270495aaa10442a019ea303d679"
)
_LEAN_DECLARATION_SEARCH_SOURCE = Path(__file__).with_name(
    "_lean_declaration_search.lean"
)
_SUPPORTED_SMTLIB_COMMANDS = frozenset(
    {"set-logic", "declare-const", "declare-fun", "assert", "check-sat"}
)


def _is_ascii_lean_declaration_name(value: str) -> bool:
    """Recognize a conservative non-evaluating Lean dotted-name grammar."""

    if not value or not value.isascii():
        return False
    for component in value.split("."):
        if not component or not (component[0].isalpha() or component[0] == "_"):
            return False
        if any(
            not (character.isalnum() or character in "_'") for character in component
        ):
            return False
    return True


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
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_assignment_to_outcome(self) -> Self:
        if (self.outcome == "SAT") != (self.assignment is not None):
            raise ValueError("only a SAT result may carry an assignment")
        return self


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


class SmtSolveRequest(StrictModel):
    logic: SmtLogic
    smtlib: str = Field(
        min_length=1,
        max_length=_MAX_SMTLIB_BYTES,
        description=(
            "ASCII SMT-LIB that declares logic, contains exactly one check-sat command, "
            "and ends with that command."
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
        commands = _top_level_smtlib_commands(self.smtlib)
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
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_model_to_outcome(self) -> Self:
        if (self.outcome == "SAT") != (self.model_smtlib is not None):
            raise ValueError("only a SAT result may carry a model")
        return self


class LeanDiagnostic(StrictModel):
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str = Field(min_length=1, max_length=4_000)


class LeanCheckRequest(StrictModel):
    """A source snippet checked in the fixed Lean and Mathlib environment."""

    source: str = Field(
        min_length=1,
        max_length=32_000,
        description="A self-contained Lean source snippet for the fixed service toolchain.",
        examples=["example : True := by trivial"],
    )
    timeout_seconds: StrictInt = Field(default=30, ge=1, le=30)


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


class LeanDeclarationKind(StrEnum):
    """Public declaration kinds reported by the fixed Lean environment."""

    AXIOM = "AXIOM"
    DEFINITION = "DEFINITION"
    THEOREM = "THEOREM"
    OPAQUE = "OPAQUE"
    QUOTIENT = "QUOTIENT"
    INDUCTIVE = "INDUCTIVE"
    CONSTRUCTOR = "CONSTRUCTOR"
    RECURSOR = "RECURSOR"


class LeanDeclarationSearchRequest(StrictModel):
    """Search the fixed Mathlib environment by literal declaration metadata."""

    name_contains: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="A literal, case-sensitive substring of the declaration name.",
    )
    type_constants: tuple[str, ...] = Field(
        default=(),
        max_length=4,
        description=(
            "ASCII dotted Lean declaration names that must all occur in the "
            "declaration type; values are compared as names and never evaluated."
        ),
    )
    namespace_prefixes: tuple[str, ...] = Field(
        default=(),
        max_length=8,
        description="ASCII dotted Lean namespace names used as literal prefixes.",
    )
    kinds: tuple[LeanDeclarationKind, ...] = Field(default=(), max_length=8)
    result_limit: StrictInt = Field(default=10, ge=1, le=20)
    timeout_seconds: StrictInt = Field(default=30, ge=1, le=60)

    @field_validator("type_constants", "namespace_prefixes")
    @classmethod
    def require_lean_declaration_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(names)) != len(names):
            raise ValueError("Lean declaration names must be unique")
        if any(not _is_ascii_lean_declaration_name(name) for name in names):
            raise ValueError(
                "Lean declaration names must use the ASCII dotted-name grammar"
            )
        return names

    @field_validator("kinds")
    @classmethod
    def require_distinct_kinds(
        cls, kinds: tuple[LeanDeclarationKind, ...]
    ) -> tuple[LeanDeclarationKind, ...]:
        if len(set(kinds)) != len(kinds):
            raise ValueError("declaration kinds must be unique")
        return kinds

    @model_validator(mode="after")
    def require_search_term(self) -> Self:
        if self.name_contains is None and not self.type_constants:
            raise ValueError("name_contains or type_constants is required")
        return self


class LeanDeclaration(StrictModel):
    """One bounded declaration display from the fixed Mathlib environment."""

    name: str = Field(min_length=1, max_length=512)
    type: str = Field(min_length=1, max_length=8_000)
    type_truncated: StrictBool
    kind: LeanDeclarationKind
    module: str | None = Field(default=None, min_length=1, max_length=512)


class LeanDeclarationSearchResult(StrictModel):
    """A bounded display projection of fixed-environment declaration matches."""

    outcome: Literal["COMPLETED", "UNAVAILABLE", "TIMEOUT", "ERROR"]
    query: LeanDeclarationSearchRequest
    mathlib_revision: Literal["fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"]
    declarations: tuple[LeanDeclaration, ...] = Field(max_length=20)
    scanned_declarations: StrictInt = Field(ge=0, le=1_000_000)
    stop_reason: Literal["EXHAUSTED", "RESULT_LIMIT"] | None = None
    detail: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def bind_projection_to_query_and_outcome(self) -> Self:
        if len(self.declarations) > self.query.result_limit:
            raise ValueError("declaration results exceed the requested result limit")
        if self.outcome == "COMPLETED":
            if self.stop_reason is None or self.detail is not None:
                raise ValueError(
                    "a completed declaration search requires a stop reason only"
                )
            if (
                self.stop_reason == "RESULT_LIMIT"
                and len(self.declarations) != self.query.result_limit
            ):
                raise ValueError("a result-limited search must fill the result limit")
            for declaration in self.declarations:
                if (
                    self.query.name_contains is not None
                    and self.query.name_contains not in declaration.name
                ):
                    raise ValueError("a declaration name does not match the query")
                if self.query.kinds and declaration.kind not in self.query.kinds:
                    raise ValueError("a declaration kind does not match the query")
                if self.query.namespace_prefixes and not any(
                    declaration.name == prefix
                    or declaration.name.startswith(f"{prefix}.")
                    for prefix in self.query.namespace_prefixes
                ):
                    raise ValueError("a declaration namespace does not match the query")
        elif (
            self.declarations
            or self.scanned_declarations
            or self.stop_reason is not None
            or self.detail is None
        ):
            raise ValueError(
                "an unsuccessful declaration search carries only failure detail"
            )
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


def solve_sat(request: SatSolveRequest) -> SatSolveResult:
    """Solve one bounded canonical CNF through the maintained Z3 Python binding."""

    import z3  # type: ignore[import-untyped]

    variables = tuple(z3.Bool(name) for name in request.cnf.variables)
    solver = z3.Solver()
    solver.set(timeout=request.timeout_ms)
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
        return SatSolveResult(
            outcome="SAT",
            assignment=tuple(
                z3.is_true(model.eval(variable, model_completion=True))
                for variable in variables
            ),
        )
    if outcome == z3.unsat:
        return SatSolveResult(outcome="UNSAT")
    return SatSolveResult(outcome="UNKNOWN", detail=solver.reason_unknown())


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
    solver.set(timeout=request.timeout_ms)
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
    return SmtSolveResult(outcome="UNKNOWN", detail=solver.reason_unknown())


def check_lean_source(request: LeanCheckRequest) -> LeanCheckResult:
    """Elaborate one source snippet in the fixed request-scoped Mathlib environment."""

    mathlib_root = _mathlib_runtime_root()
    lake = shutil.which("lake")
    if mathlib_root is None or lake is None:
        return LeanCheckResult(
            outcome="UNAVAILABLE",
            diagnostics=(),
            detail="The fixed Lean and Mathlib environment is not installed.",
        )
    with tempfile.TemporaryDirectory(prefix="jacobian-lean-") as directory:
        source_path = Path(directory) / "Snippet.lean"
        source_path.write_text(request.source, encoding="utf-8")
        try:
            completed = run_bounded_process(
                [
                    lake,
                    "env",
                    "lean",
                    str(source_path),
                    "-T",
                    "1000000",
                    "-M",
                    "10240",
                    "-j",
                    "1",
                    "--trust=0",
                ],
                input_bytes=b"",
                timeout_seconds=float(request.timeout_seconds),
                environment=worker_environment(
                    extra_variables=("PATH", "ELAN_HOME", "HOME"),
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
                cwd=str(mathlib_root),
            )
        except OSError:
            return LeanCheckResult(
                outcome="UNAVAILABLE",
                diagnostics=(),
                detail="The fixed Lean and Mathlib environment could not be started.",
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


def search_mathlib_declarations(
    request: LeanDeclarationSearchRequest,
) -> LeanDeclarationSearchResult:
    """Search one fixed local Mathlib environment in a bounded child process."""

    mathlib_root = _mathlib_runtime_root()
    lake = shutil.which("lake")
    if mathlib_root is None or lake is None:
        return _mathlib_search_failure(
            request,
            "UNAVAILABLE",
            "The fixed local Mathlib environment is not installed.",
        )
    with tempfile.TemporaryDirectory(prefix="jacobian-mathlib-search-") as directory:
        query_path = Path(directory) / "query.json"
        query_path.write_text(
            json.dumps(
                {
                    "name_contains": request.name_contains,
                    "type_constants": list(request.type_constants),
                    "namespace_prefixes": list(request.namespace_prefixes),
                    "kinds": [kind.value for kind in request.kinds],
                    "limit": request.result_limit,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            completed = run_bounded_process(
                [
                    lake,
                    "env",
                    "lean",
                    str(_LEAN_DECLARATION_SEARCH_SOURCE),
                    "-T",
                    "1000000000",
                    "-M",
                    "10240",
                    "-j",
                    "1",
                    "--trust=0",
                ],
                input_bytes=b"",
                timeout_seconds=float(request.timeout_seconds),
                environment=worker_environment(
                    extra_variables=("PATH", "ELAN_HOME", "HOME"),
                    overrides={
                        "ELAN_TOOLCHAIN": _LEAN_TOOLCHAIN,
                        "JACOBIAN_LEAN_QUERY_FILE": str(query_path),
                    },
                    locale="C.UTF-8",
                ),
                stdout_limit=1_000_000,
                stderr_limit=64_000,
                cwd=str(mathlib_root),
            )
        except OSError:
            return _mathlib_search_failure(
                request,
                "UNAVAILABLE",
                "The fixed local Mathlib environment could not be started.",
            )
    if completed.timed_out:
        return _mathlib_search_failure(
            request,
            "TIMEOUT",
            "Mathlib declaration search exceeded the declared time limit.",
        )
    if (
        completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return _mathlib_search_failure(
            request,
            "ERROR",
            "Mathlib declaration search failed within a process resource limit.",
        )
    return _parse_mathlib_search_result(request, completed.stdout)


def _mathlib_runtime_root() -> Path | None:
    configured = os.environ.get("JACOBIAN_MATHLIB_ROOT")
    candidates = (
        (Path(configured),)
        if configured is not None
        else (Path(__file__).resolve().parents[4] / "lean",)
    )
    for candidate in candidates:
        manifest_path = candidate / "lake-manifest.json"
        mathlib_olean = (
            candidate
            / ".lake"
            / "packages"
            / "mathlib"
            / ".lake"
            / "build"
            / "lib"
            / "lean"
            / "Mathlib.olean"
        )
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            packages = manifest["packages"]
            revision = next(
                package["rev"]
                for package in packages
                if package.get("name") == "mathlib"
            )
            toolchain = (
                (candidate / "lean-toolchain").read_text(encoding="utf-8").strip()
            )
        except (KeyError, OSError, StopIteration, TypeError, ValueError):
            continue
        if (
            revision == _MATHLIB_REVISION
            and toolchain == _LEAN_TOOLCHAIN
            and hashlib.sha256(manifest_bytes).hexdigest() == _MATHLIB_MANIFEST_SHA256
            and mathlib_olean.is_file()
        ):
            return candidate
    return None


def _mathlib_search_failure(
    request: LeanDeclarationSearchRequest,
    outcome: Literal["UNAVAILABLE", "TIMEOUT", "ERROR"],
    detail: str,
) -> LeanDeclarationSearchResult:
    return LeanDeclarationSearchResult(
        outcome=outcome,
        query=request,
        mathlib_revision=_MATHLIB_REVISION,
        declarations=(),
        scanned_declarations=0,
        detail=detail,
    )


def _parse_mathlib_search_result(
    request: LeanDeclarationSearchRequest, output: bytes
) -> LeanDeclarationSearchResult:
    marker = "JACOBIAN_LEAN_SEARCH_RESULT "
    lines = output.decode("utf-8", errors="replace").splitlines()
    payload_lines = [
        line.removeprefix(marker) for line in lines if line.startswith(marker)
    ]
    if len(payload_lines) != 1:
        return _mathlib_search_failure(
            request,
            "ERROR",
            "Mathlib declaration search returned an invalid response.",
        )
    try:
        payload = json.loads(payload_lines[0])
        declarations = tuple(
            LeanDeclaration.model_validate(value) for value in payload["declarations"]
        )
        return LeanDeclarationSearchResult(
            outcome="COMPLETED",
            query=request,
            mathlib_revision=_MATHLIB_REVISION,
            declarations=declarations,
            scanned_declarations=payload["scanned_declarations"],
            stop_reason=payload["stop_reason"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _mathlib_search_failure(
            request,
            "ERROR",
            "Mathlib declaration search returned an invalid response.",
        )


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
        version="1",
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
        operation_id="smt.solve",
        version="1",
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
        description="Elaborate one source snippet in the fixed Lean and Mathlib service environment and return typed diagnostics.",
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
    MathTool(
        operation_id="lean.declarations.search",
        version="1",
        title="Search the fixed Mathlib declaration environment",
        description=(
            "Return bounded declaration-name and complete-type matches with "
            "typed previews from the pinned local Mathlib environment."
        ),
        request_type=LeanDeclarationSearchRequest,
        result_type=LeanDeclarationSearchResult,
        run=search_mathlib_declarations,
        tags=("lean", "mathlib", "declarations", "search", "bounded"),
        examples=(
            example(
                "natural_number_types",
                "Find declarations whose types mention Nat.",
                {"type_constants": ["Nat"]},
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
    "LeanDeclaration",
    "LeanDeclarationKind",
    "LeanDeclarationSearchRequest",
    "LeanDeclarationSearchResult",
    "SatAssignmentCheckRequest",
    "SatAssignmentCheckResult",
    "SatSolveRequest",
    "SatSolveResult",
    "SmtLogic",
    "SmtSolveRequest",
    "SmtSolveResult",
    "canonicalize_cnf",
    "check_lean_source",
    "check_sat_assignment",
    "search_mathlib_declarations",
    "solve_sat",
    "solve_smt",
]
