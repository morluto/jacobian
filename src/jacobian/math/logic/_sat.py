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

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    require_execution_deadline,
)
from jacobian._models import StrictModel
from jacobian._worker_errors import decode_worker_execution_error
from jacobian.math.logic._cnf import (
    _MAX_VARIABLES,
    CanonicalCnf,
    SatAssignmentCheckRequest,
    check_sat_assignment,
)
from jacobian.math.logic._smt import (
    _execution_deadline,
    _require_execution_deadline,
    _solver_settings,
)
from jacobian.math.logic._solver_errors import (
    _classify_exhaustion,
    _project_unknown,
    _raise_exhaustion,
    _UnknownResource,
)
from jacobian.process import (
    ProcessResourceLimits,
    check_bounded_process_result,
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
        if self.assignment is not None and len(self.assignment) != len(
            self.source.cnf.variables
        ):
            raise _validation_error(
                "logic.sat_assignment_length",
                "a SAT assignment must cover the source CNF variable axis",
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
        raise OperationBackendError(BackendFailureReason.INITIALIZATION) from exc

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
    except (OperationExecutionTimeoutError, OperationExecutionCancelledError):
        raise
    except (OSError, z3.Z3Exception) as exc:
        exhausted = _classify_exhaustion(str(exc))
        if exhausted is not None:
            _raise_exhaustion(exhausted, cause=exc)
        raise OperationBackendError(BackendFailureReason.ABNORMAL_EXIT) from exc
    exhausted, detail = _project_unknown(solver.reason_unknown())
    return {
        "outcome": "UNKNOWN",
        "assignment": None,
        "exhausted": exhausted,
        "detail": detail,
    }


def _run_sat_worker(request: SatSolveRequest) -> SatSolveResult:
    """Project one killable SAT worker invocation onto the public result."""

    deadline = _execution_deadline(request.timeout_ms, "before SAT worker")
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-sat-") as worker_directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before SAT worker"
                )
            completed = run_bounded_process(
                [sys.executable, str(_SAT_WORKER)],
                input_bytes=json.dumps(
                    {"_deadline": deadline, **request.model_dump(mode="json")},
                    separators=(",", ":"),
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
        if not isinstance(response, dict) or "source" in response:
            raise TypeError("worker response must not replace the retained source")
        result = SatSolveResult.model_validate(
            {"source": request.model_dump(mode="json"), **response}
        )
        if (
            result.assignment is not None
            and not check_sat_assignment(
                SatAssignmentCheckRequest(cnf=request.cnf, assignment=result.assignment)
            ).satisfies
        ):
            # The parent-side scan may have consumed the remaining budget.
            _require_execution_deadline(deadline, "after SAT assignment check")
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        if result.exhausted is not None:
            _raise_exhaustion(result.exhausted)
        _require_execution_deadline(deadline, "after SAT result projection")
        return result
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc


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
