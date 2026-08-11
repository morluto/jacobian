"""Bounded rational optimization operations backed by SymPy."""

from __future__ import annotations

import sys
from pathlib import Path

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import (
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.validated_analysis import (
    RationalLinearProgramObligation,
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)
from jacobian.domains._examples import example
from jacobian.domains.optimization.protocol import (
    PROTOCOL,
    RationalOptimizationWorkerRequest,
    parse_optimization_worker_response,
)
from jacobian.operations import (
    BoundedSearchIncomplete,
    BoundedSearchInterrupted,
    BoundedSearchOperation,
    BoundedSearchOutcome,
    BoundedSearchWitness,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment

_WORKER_MODULE = "jacobian.domains.optimization.worker"


def _run_worker(request: RationalLinearProgramRequest) -> RationalLinearProgramResult:
    worker_request = RationalOptimizationWorkerRequest(
        protocol=PROTOCOL,
        request=request,
    )
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=("-I", "-m", _WORKER_MODULE),
            stdin_bytes=canonicalize_json(worker_request.model_dump(mode="json")),
            timeout_seconds=float(request.wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=2_000_000,
            stderr_limit_bytes=64_000,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=request.wall_seconds + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.TIMED_OUT:
        raise TimeoutError
    if (
        completed.returncode != 0
        or completed.termination is not ProcessTermination.EXITED
    ):
        raise RuntimeError("rational optimization worker failed")
    value = loads_strict_json(completed.stdout)
    return parse_optimization_worker_response(value).result


def _linear_program(
    request: RationalLinearProgramRequest,
) -> BoundedSearchOutcome[RationalLinearProgramResult]:
    try:
        result = _run_worker(request)
    except TimeoutError:
        detail = (
            "The exact rational LP worker exceeded the declared wall-clock "
            "budget; no feasibility or optimality conclusion is available."
        )
        return BoundedSearchInterrupted(
            value=RationalLinearProgramResult(status="TIMEOUT", detail=detail),
            status=ExecutionStatus.TIMEOUT,
            diagnostic=CapabilityDiagnostic(
                code="RATIONAL_LINEAR_PROGRAM_TIMEOUT",
                stage="rational_optimization_backend",
                message=detail,
            ),
        )
    except (OSError, RuntimeError, ValueError):
        detail = (
            "The exact rational LP worker failed or returned malformed "
            "output; no feasibility or optimality conclusion is available."
        )
        return BoundedSearchInterrupted(
            value=RationalLinearProgramResult(
                status="BACKEND_ERROR",
                detail=detail,
            ),
            status=ExecutionStatus.ERROR,
            diagnostic=CapabilityDiagnostic(
                code="RATIONAL_LINEAR_PROGRAM_BACKEND_ERROR",
                stage="rational_optimization_backend",
                message=detail,
            ),
        )
    if result.status == "CERTIFICATE_PRODUCED":
        return BoundedSearchWitness(result)
    return BoundedSearchIncomplete(result)


def _scope(
    request: RationalLinearProgramRequest,
    _result: RationalLinearProgramResult,
) -> dict[str, object]:
    return {
        "variables": len(request.program.variables),
        "constraints": len(request.program.coefficients),
        "wall_seconds": request.wall_seconds,
        "standard_form": "MIN_CX; AX_EQUALS_B; X_NONNEGATIVE",
    }


def _obligation(
    request: RationalLinearProgramRequest,
    result: RationalLinearProgramResult,
) -> RationalLinearProgramObligation:
    return RationalLinearProgramObligation(
        program=request.program,
        status=result.status,
        primal_candidate=result.primal_candidate,
        dual_candidate=result.dual_candidate,
    )


RATIONAL_LINEAR_CAPABILITIES = (
    BoundedSearchOperation(
        capability_id="optimization.linear.rational_optimum.compute",
        title="Produce a rational linear-program optimum certificate",
        description=(
            "Use bounded exact SymPy simplex calls to produce primal and dual "
            "candidates for a standard-form rational linear program."
        ),
        request_model=RationalLinearProgramRequest,
        result_model=RationalLinearProgramResult,
        implementation=_linear_program,
        relation_id="optimization.linear.rational_optimum.relation",
        scope_parameters=_scope,
        is_complete=lambda result: result.status == "CERTIFICATE_PRODUCED",
        obligation_model=RationalLinearProgramObligation,
        obligation=_obligation,
        incomplete_basis=(
            "bounded exact optimization did not produce primal and dual "
            "candidates with equal exact objective values"
        ),
        tags=(
            "optimization",
            "linear-program",
            "rational",
            "certificate",
            "bounded",
        ),
        invocation_examples=(
            example(
                "one_variable_unit_lp",
                "Optimize x subject to x=1 and x>=0.",
                {
                    "program": {
                        "variables": ["x"],
                        "objective": [{"num": "1", "den": "1"}],
                        "coefficients": [[{"num": "1", "den": "1"}]],
                        "rhs": [{"num": "1", "den": "1"}],
                    },
                    "wall_seconds": 5,
                },
            ),
        ),
    ),
)

__all__ = ["RATIONAL_LINEAR_CAPABILITIES"]
