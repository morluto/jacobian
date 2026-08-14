"""Bounded rational optimization operations backed by SymPy."""

from __future__ import annotations

import sys
from pathlib import Path

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import (
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.validated_analysis import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)
from jacobian.domains._examples import example
from jacobian.domains.optimization.protocol import (
    PROTOCOL,
    RationalOptimizationWorkerRequest,
    parse_optimization_worker_response,
)
from jacobian.operation_declarations import OperationDeclaration, inline_operation
from jacobian.operations import OperationAbortError
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
) -> RationalLinearProgramResult:
    try:
        result = _run_worker(request)
    except TimeoutError:
        detail = (
            "The exact rational LP worker exceeded the declared wall-clock "
            "budget; no feasibility or optimality conclusion is available."
        )
        raise OperationAbortError(
            ExecutionStatus.TIMEOUT,
            OperationDiagnostic(
                code="RATIONAL_LINEAR_PROGRAM_TIMEOUT",
                stage="rational_optimization_backend",
                message=detail,
            ),
        ) from None
    except (OSError, RuntimeError, ValueError):
        detail = (
            "The exact rational LP worker failed or returned malformed "
            "output; no feasibility or optimality conclusion is available."
        )
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            OperationDiagnostic(
                code="RATIONAL_LINEAR_PROGRAM_BACKEND_ERROR",
                stage="rational_optimization_backend",
                message=detail,
            ),
        ) from None
    return result


RATIONAL_LINEAR_OPERATIONS = (
    inline_operation(
        OperationDeclaration(
            operation_id="optimization.linear.rational_optimum.compute",
            version="1",
            title="Produce a rational linear-program optimum certificate",
            description=(
                "Use bounded exact SymPy simplex calls to produce primal and dual "
                "candidates for a standard-form rational linear program."
            ),
            request_type=RationalLinearProgramRequest,
            result_type=RationalLinearProgramResult,
            execute=_linear_program,
            tags=(
                "optimization",
                "linear-program",
                "rational",
                "certificate",
                "bounded",
            ),
            examples=(
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
        )
    ),
)

__all__ = ["RATIONAL_LINEAR_OPERATIONS"]
