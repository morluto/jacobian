"""Bounded SAT solver and LPR-refutation contracts."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json
from jacobian.math.logic._cnf import (
    _MAX_VARIABLES,
    CanonicalCnf,
)
from jacobian.math.logic._smt import (
    _EXHAUSTION_DETAILS,
    _classify_exhaustion,
    _project_unknown,
    _solver_settings,
    _UnknownResource,
)
from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

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


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


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
            raise _validation_error(
                "logic.sat_assignment_outcome",
                "only a SAT result may carry an assignment",
            )
        if self.exhausted is not None and self.outcome != "UNKNOWN":
            raise _validation_error(
                "logic.unknown_exhaustion",
                "only an UNKNOWN result may name an exhausted budget",
            )
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
            raise _validation_error(
                "logic.lpr_hint_id", "LPR hint clause IDs must be positive"
            )
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
            raise _validation_error(
                "logic.lpr_hint_id", "LPR hint clause IDs must be positive"
            )
        return value

    @model_validator(mode="after")
    def bind_witness_to_the_clause_pivot(self) -> Self:
        if self.witness is not None:
            if not self.clause:
                raise _validation_error(
                    "logic.lpr_witness_clause",
                    "an empty LPR clause may not carry a witness",
                )
            if not self.witness or self.witness[0] != self.clause[0]:
                raise _validation_error(
                    "logic.lpr_witness_pivot",
                    "an LPR witness must start with the added clause's pivot literal",
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
            raise _validation_error(
                "logic.lpr_deletion_id", "deleted LPR clause IDs must be positive"
            )
        if len(set(value)) != len(value):
            raise _validation_error(
                "logic.lpr_duplicate_deletion",
                "one LPR deletion may not name a clause more than once",
            )
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
        "VALID_REFUTATION",
        "INVALID_REFUTATION",
        "UNAVAILABLE",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    cnf: CanonicalCnf
    refutation: SatLprRefutation
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_execution_detail(self) -> Self:
        if (self.outcome == "VALID_REFUTATION") != (self.detail is None):
            raise _validation_error(
                "logic.refutation_detail",
                "only a valid refutation may omit its outcome detail",
            )
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
        raise _validation_error(
            "logic.lpr_live_clause", f"{label} references non-live clause ID {missing}"
        )


def _require_lpr_literal_axis(
    literals: tuple[int, ...], variable_count: int, label: str
) -> None:
    if any(literal == 0 or abs(literal) > variable_count for literal in literals):
        raise _validation_error(
            "logic.lpr_literal_axis",
            f"{label} literal is outside the CNF variable axis",
        )


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
        raise _validation_error(
            "logic.lpr_clause_id",
            "LPR additions must use IDs after the canonical source clauses",
        )
    if step.clause_id in live_clause_widths:
        raise _validation_error(
            "logic.lpr_duplicate_clause_id",
            "LPR additions may not overwrite a live clause ID",
        )
    _require_lpr_literal_axis(step.clause, variable_count, "LPR clause")
    if step.witness is not None:
        _require_lpr_literal_axis(step.witness, variable_count, "LPR witness")
    _require_live_lpr_ids(
        step.at_hint_clause_ids, live_clause_widths, "LPR asymmetric-tautology hint"
    )
    if len({hint.clause_id for hint in step.propagation_hints}) != len(
        step.propagation_hints
    ):
        raise _validation_error(
            "logic.lpr_duplicate_hint_id",
            "LPR propagation hint clause IDs must be unique",
        )
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
            raise _validation_error(
                "logic.lpr_work_budget",
                "LPR replay exceeds the declared literal-inspection work bound",
            )
        live_clause_widths[step.clause_id] = len(step.clause)
    echoed_result = {
        "outcome": "INVALID_REFUTATION",
        "cnf": cnf.model_dump(mode="json"),
        "refutation": refutation.model_dump(mode="json"),
        "detail": "x" * 1_024,
    }
    if len(canonicalize_json(echoed_result)) > _MAX_LPR_RESULT_BYTES:
        raise _validation_error(
            "logic.lpr_result_budget",
            "LPR refutation exceeds the source-bound result limit",
        )


def solve_sat(request: SatSolveRequest) -> SatSolveResult:
    """Solve one bounded canonical CNF through the maintained Z3 Python binding."""

    try:
        import z3  # type: ignore[import-untyped]
    except (ImportError, OSError) as exc:
        return SatSolveResult(
            outcome="UNKNOWN",
            detail=f"the Z3 backend could not initialize: {exc}"[:1_024],
        )

    try:
        variables = tuple(z3.Bool(name) for name in request.cnf.variables)
        solver = z3.Solver()
        solver.set(**_solver_settings(request.timeout_ms))
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
    except (OSError, z3.Z3Exception) as exc:
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
    outcome: Literal[
        "INVALID_REFUTATION", "UNAVAILABLE", "TIMEOUT", "CANCELLED", "ERROR"
    ]
    if completed.timed_out:
        outcome, detail = "TIMEOUT", "Cake LPR exceeded the declared wall-time limit."
    elif completed.cancelled:
        outcome, detail = "CANCELLED", "Cake LPR execution was cancelled."
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


__all__ = [
    "LprAddition",
    "LprDeletion",
    "LprPropagationHint",
    "LprStep",
    "SatLprRefutation",
    "SatRefutationCheckRequest",
    "SatRefutationCheckResult",
    "SatSolveRequest",
    "SatSolveResult",
    "check_sat_refutation",
    "solve_sat",
]
