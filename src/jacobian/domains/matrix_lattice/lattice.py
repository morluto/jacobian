"""Bounded exact lattice-basis reduction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityProviderAvailability,
)
from jacobian.contracts.matrices import IntegerMatrix
from jacobian.contracts.matrix_operations import (
    LatticeReductionRequest,
    LatticeReductionResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains._examples import example
from jacobian.operations import (
    ComputedOutcome,
    ComputedSuccess,
    MaterializedOperation,
    OperationExecutionFailure,
)
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.providers.flint_runtime import python_flint_lll_provider_runtime
from jacobian.worker_environment import worker_environment

from .lll_worker import PROTOCOL

STDOUT_LIMIT = 1_000_000
STDERR_LIMIT = 64_000
LATTICE_RUNTIME = python_flint_lll_provider_runtime()


def _failure(
    status: ExecutionStatus,
    code: str,
    message: str,
) -> OperationExecutionFailure:
    return OperationExecutionFailure(
        status=status,
        diagnostic=CapabilityDiagnostic(
            code=code,
            stage="lattice_reduction",
            message=message,
            hint=(
                "Install the pinned Python-FLINT LLL provider or reduce the "
                "matrix size, scalar size, or requested wall budget."
            ),
        ),
    )


def reduce_lattice_basis(
    request: LatticeReductionRequest,
) -> ComputedOutcome[LatticeReductionResult]:
    if (
        LATTICE_RUNTIME.availability is not CapabilityProviderAvailability.AVAILABLE
        or python_flint_lll_provider_runtime(refresh=True) != LATTICE_RUNTIME
    ):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_PROVIDER_UNAVAILABLE",
            "The pinned Python-FLINT exact-gram LLL provider is unavailable.",
        )
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=(
                "-I",
                "-m",
                "jacobian.domains.matrix_lattice.lll_worker",
            ),
            stdin_bytes=canonicalize_json(
                {
                    "protocol": PROTOCOL,
                    "basis": request.basis.model_dump(mode="json"),
                }
            ),
            timeout_seconds=float(request.resource_budget.wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=STDOUT_LIMIT,
            stderr_limit_bytes=STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=request.resource_budget.wall_seconds + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.START_FAILED:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_WORKER_START_FAILED",
            "The isolated Python-FLINT LLL worker could not be started.",
        )
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            ExecutionStatus.TIMEOUT,
            "FLINT_LLL_TIMEOUT",
            "The LLL wall-clock budget expired; no reduced basis was retained.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_OUTPUT_LIMIT_EXCEEDED",
            "The isolated LLL worker exceeded its bounded output protocol.",
        )
    if completed.returncode != 0:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_WORKER_FAILED",
            "The isolated Python-FLINT LLL worker did not complete successfully.",
        )
    try:
        output = loads_strict_json(completed.stdout)
        if not isinstance(output, dict) or set(output) != {
            "protocol",
            "rank",
            "reduced_basis",
            "transformation",
        }:
            raise ValueError("LLL worker response has unexpected fields")
        if output["protocol"] != PROTOCOL:
            raise ValueError("LLL worker protocol does not match")
        result = LatticeReductionResult(
            reduced_basis=IntegerMatrix(entries=output["reduced_basis"]),
            transformation=IntegerMatrix(entries=output["transformation"]),
            rank=output["rank"],
        )
    except (TypeError, ValueError, ValidationError):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_PROTOCOL_INVALID",
            "The LLL worker returned a result outside the bounded exact contract.",
        )
    if python_flint_lll_provider_runtime(refresh=True) != LATTICE_RUNTIME:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_LLL_PROVIDER_CHANGED",
            "The Python-FLINT runtime changed during LLL execution.",
        )
    return ComputedSuccess(result)


LATTICE_CAPABILITIES: tuple[MaterializedOperation[Any, Any, Any, Any], ...] = (
    MaterializedOperation(
        capability_id="lattice.basis.reduce",
        title="Reduce an exact integer lattice basis",
        description=(
            "Run bounded Python-FLINT exact-gram LLL and return the reduced row "
            "basis with its exact left transformation."
        ),
        request_model=LatticeReductionRequest,
        result_model=LatticeReductionResult,
        implementation=reduce_lattice_basis,
        relation_id="lattice.relation.reduced-basis-of",
        tags=("lattice", "lll", "exact-integer", "bounded", "python-flint"),
        invocation_examples=(
            example(
                "unit_basis",
                "Reduce the one-dimensional unit basis.",
                {"basis": {"entries": [["1"]]}},
            ),
        ),
        version="3",
    ),
)
