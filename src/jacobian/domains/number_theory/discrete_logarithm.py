"""Wall-bounded isolated SymPy discrete-logarithm operation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Never

from pydantic import ValidationError

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.number_theory import (
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains._examples import example
from jacobian.domains.number_theory._support import number_theory_operation
from jacobian.domains.number_theory.discrete_logarithm_protocol import (
    DiscreteLogarithmWorkerRequest,
    DiscreteLogarithmWorkerResult,
)
from jacobian.operation_declarations import (
    OperationAbortError,
)
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.worker_environment import worker_environment

_STDOUT_LIMIT = 64_000
_STDERR_LIMIT = 64_000


def _failure(
    status: ExecutionStatus,
    code: str,
    message: str,
) -> Never:
    raise OperationAbortError(
        status,
        OperationDiagnostic(
            code=code,
            stage="discrete_logarithm_computation",
            message=message,
            hint="Reduce the modulus or increase the bounded wall time.",
        ),
    )


def _compute(
    request: DiscreteLogarithmRequest,
) -> DiscreteLogarithmResult:
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=(
                "-I",
                "-m",
                "jacobian.domains.number_theory.discrete_logarithm_worker",
            ),
            stdin_bytes=canonicalize_json(
                DiscreteLogarithmWorkerRequest(
                    protocol="jacobian.number-theory.discrete-logarithm.sympy.v1",
                    request=request,
                ).model_dump(mode="json")
            ),
            timeout_seconds=float(request.resource_budget.wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=_STDOUT_LIMIT,
            stderr_limit_bytes=_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=request.resource_budget.wall_seconds + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.START_FAILED:
        return _failure(
            ExecutionStatus.ERROR,
            "DISCRETE_LOGARITHM_WORKER_START_FAILED",
            "The isolated SymPy discrete-logarithm worker could not be started.",
        )
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            ExecutionStatus.TIMEOUT,
            "DISCRETE_LOGARITHM_TIMEOUT",
            "The discrete-logarithm wall-clock budget expired; no conclusion is available.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _failure(
            ExecutionStatus.ERROR,
            "DISCRETE_LOGARITHM_OUTPUT_LIMIT_EXCEEDED",
            "The isolated worker exceeded its bounded output protocol.",
        )
    if completed.returncode != 0:
        return _failure(
            ExecutionStatus.ERROR,
            "DISCRETE_LOGARITHM_WORKER_FAILED",
            "The isolated SymPy discrete-logarithm computation failed.",
        )
    try:
        result = DiscreteLogarithmWorkerResult.model_validate(
            loads_strict_json(completed.stdout)
        ).result
    except (TypeError, ValueError, ValidationError):
        return _failure(
            ExecutionStatus.ERROR,
            "DISCRETE_LOGARITHM_PROTOCOL_INVALID",
            "The worker returned a result outside the bounded exact contract.",
        )
    return result


DISCRETE_LOGARITHM_OPERATION = number_theory_operation(
    "modular.compute.discrete_logarithm",
    "Compute a bounded discrete logarithm",
    (
        "Compute a modular discrete logarithm with SymPy in an isolated "
        "wall-bounded worker. Timeout and worker failure are non-conclusions."
    ),
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
    _compute,
    "number-theory",
    "modular",
    "discrete-logarithm",
    "bounded",
    "sympy",
    version="1",
    examples=(
        example(
            "two_to_one_mod_three",
            "Solve 2^x = 1 modulo 3.",
            {
                "base": 2,
                "target": 1,
                "modulus": 3,
                "resource_budget": {"wall_seconds": 5},
            },
        ),
    ),
)
