"""Bounded SAT solver contracts."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic._cnf import (
    _MAX_VARIABLES,
    CanonicalCnf,
    SatAssignmentCheckRequest,
    check_sat_assignment,
)
from jacobian.math.logic._smt import (
    _EXHAUSTION_DETAILS,
    _classify_exhaustion,
    _project_unknown,
    _solver_settings,
    _UnknownResource,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_SAT_WORKER = Path(__file__).with_name("_sat_worker.py")
_SAT_WORKER_OUTPUT_BYTES = 64 * 1024
_SAT_WORKER_ERROR_BYTES = 16_384
_SAT_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_SAT_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class SatSolveRequest(StrictModel):
    cnf: CanonicalCnf
    timeout_ms: StrictInt = Field(default=1_000, ge=1, le=10_000)


class SatSolveResult(StrictModel):
    """One solver outcome bound to the exact canonical CNF it answers."""

    source: SatSolveRequest
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
        if self.assignment is not None:
            if len(self.assignment) != len(self.source.cnf.variables):
                raise _validation_error(
                    "logic.sat_assignment_length",
                    "a SAT assignment must cover the source CNF variable axis",
                )
            if not all(
                any(
                    self.assignment[abs(literal) - 1] == (literal > 0)
                    for literal in clause
                )
                for clause in self.source.cnf.clauses
            ):
                raise _validation_error(
                    "logic.sat_assignment_unsatisfied",
                    "a SAT assignment must satisfy every source CNF clause",
                )
        if self.exhausted is not None and self.outcome != "UNKNOWN":
            raise _validation_error(
                "logic.unknown_exhaustion",
                "only an UNKNOWN result may name an exhausted budget",
            )
        return self


def _solve_sat_kernel(*, cnf: CanonicalCnf, timeout_ms: int) -> dict[str, object]:
    """Run one complete Z3 SAT lifecycle inside the owned worker process."""

    try:
        import z3
    except (ImportError, OSError) as exc:
        return {
            "outcome": "UNKNOWN",
            "assignment": None,
            "exhausted": None,
            "detail": f"the Z3 backend could not initialize: {exc}"[:1_024],
        }

    try:
        variables = tuple(z3.Bool(name) for name in cnf.variables)
        solver = z3.Solver()
        solver.set(**_solver_settings(timeout_ms))
        for clause in cnf.clauses:
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
            checked_assignment = check_sat_assignment(
                SatAssignmentCheckRequest(cnf=cnf, assignment=assignment)
            )
            if not checked_assignment.satisfies:
                return {
                    "outcome": "UNKNOWN",
                    "assignment": None,
                    "exhausted": None,
                    "detail": (
                        "the Z3 backend returned a model that does not satisfy the "
                        "canonical CNF"
                    ),
                }
            return {
                "outcome": "SAT",
                "assignment": list(assignment),
                "exhausted": None,
                "detail": None,
            }
        if outcome == z3.unsat:
            return {
                "outcome": "UNSAT",
                "assignment": None,
                "exhausted": None,
                "detail": None,
            }
    except (OSError, z3.Z3Exception) as exc:
        exhausted = _classify_exhaustion(str(exc))
        if exhausted is not None:
            return {
                "outcome": "UNKNOWN",
                "assignment": None,
                "exhausted": exhausted,
                "detail": _EXHAUSTION_DETAILS[exhausted],
            }
        detail = f"the Z3 backend failed during the bounded solve: {exc}"
        return {
            "outcome": "UNKNOWN",
            "assignment": None,
            "exhausted": None,
            "detail": detail[:1_024],
        }
    exhausted, detail = _project_unknown(solver.reason_unknown())
    return {
        "outcome": "UNKNOWN",
        "assignment": None,
        "exhausted": exhausted,
        "detail": detail,
    }


def _result(request: SatSolveRequest, **values: object) -> SatSolveResult:
    """Bind every projected worker outcome to its exact admitted source."""

    return SatSolveResult.model_validate(
        {"source": request.model_dump(mode="json"), **values}
    )


def _time_exhausted_result(request: SatSolveRequest) -> SatSolveResult:
    return _result(
        request,
        outcome="UNKNOWN",
        exhausted="time",
        detail=_EXHAUSTION_DETAILS["time"],
    )


def _run_sat_worker(request: SatSolveRequest) -> SatSolveResult:
    """Project one killable SAT worker invocation onto the public result."""

    deadline = time.monotonic() + (request.timeout_ms / 1_000)
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-sat-") as worker_directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _time_exhausted_result(request)
            completed = run_bounded_process(
                [sys.executable, str(_SAT_WORKER)],
                input_bytes=json.dumps(
                    request.model_dump(mode="json"), separators=(",", ":")
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_SAT_WORKER_OUTPUT_BYTES,
                stderr_limit=_SAT_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(request.timeout_ms / 1_000)),
                    address_space_bytes=_SAT_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_SAT_WORKER_FILE_SIZE_BYTES,
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
        result = SatSolveResult.model_validate(
            {
                "source": request.model_dump(mode="json"),
                **json.loads(completed.stdout.decode("utf-8")),
            }
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


def solve_sat(request: SatSolveRequest) -> SatSolveResult:
    """Solve one canonical CNF in a killable owner-local Z3 worker."""

    return _run_sat_worker(request)


__all__ = [
    "SatSolveRequest",
    "SatSolveResult",
    "_run_sat_worker",
    "_solve_sat_kernel",
    "solve_sat",
]
