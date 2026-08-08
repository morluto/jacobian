"""FLINT conversion boundary for ordinary rational-linear values."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityProviderAvailability,
)
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.operations import (
    ComputedOutcome,
    ComputedSuccess,
    OperationExecutionFailure,
)
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.providers.flint_runtime import python_flint_provider_runtime
from jacobian.worker_environment import worker_environment

RUNTIME = python_flint_provider_runtime()
_SOLUTION_PROTOCOL = "jacobian.rational-linear-solution-worker/v1"
_INCONSISTENCY_PROTOCOL = "jacobian.rational-linear-inconsistency-worker/v1"


def _failure(
    code: str, status: ExecutionStatus, message: str
) -> OperationExecutionFailure:
    return OperationExecutionFailure(
        status=status,
        diagnostic=CapabilityDiagnostic(
            code=code,
            stage="rational_linear_provider",
            message=message,
            hint="Install the pinned Python-FLINT rational-linear provider and retry.",
        ),
    )


def _run(
    request: LinearRationalSolutionFindRequest | LinearRationalInconsistencyFindRequest,
    protocol: str,
) -> dict[str, Any] | None:
    runtime = python_flint_provider_runtime(refresh=True)
    if (
        RUNTIME.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime != RUNTIME
    ):
        return None
    budget = request.resource_budget.wall_seconds
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=("-I", "-m", "jacobian.domains.rational_linear.worker"),
            stdin_bytes=canonicalize_json(
                {"protocol": protocol, "system": request.system.model_dump(mode="json")}
            ),
            timeout_seconds=float(budget),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=1_000_000,
            stderr_limit_bytes=64_000,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=budget + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.TIMED_OUT:
        raise TimeoutError
    if (
        completed.termination is ProcessTermination.START_FAILED
        or completed.returncode != 0
    ):
        raise RuntimeError("rational-linear worker failed")
    payload = loads_strict_json(completed.stdout)
    if not isinstance(payload, dict) or payload.get("protocol") != protocol:
        raise ValueError("rational-linear worker protocol is invalid")
    return payload


def compute_rational_solution(
    request: LinearRationalSolutionFindRequest,
) -> ComputedOutcome[LinearRationalSolutionResult]:
    try:
        payload = _run(request, _SOLUTION_PROTOCOL)
        if payload is None:
            return _failure(
                "FLINT_LINEAR_PROVIDER_UNAVAILABLE",
                ExecutionStatus.ERROR,
                "The pinned Python-FLINT rational-linear provider is unavailable.",
            )
        if payload.get("status") == "NO_SOLUTION_PRODUCED":
            return _failure(
                "LINEAR_SYSTEM_NOT_UNIQUELY_SOLVABLE",
                ExecutionStatus.ERROR,
                "The system has no unique solution candidate.",
            )
        if payload.get("status") != "SOLUTION_PRODUCED":
            raise ValueError("worker did not produce a solution")
        return ComputedSuccess(LinearRationalSolutionResult(values=payload["values"]))
    except TimeoutError:
        return _failure(
            "FLINT_LINEAR_TIMEOUT",
            ExecutionStatus.TIMEOUT,
            "The bounded rational-linear computation timed out.",
        )
    except (RuntimeError, TypeError, ValueError):
        return _failure(
            "FLINT_LINEAR_WORKER_FAILED",
            ExecutionStatus.ERROR,
            "The rational-linear worker returned no usable result.",
        )


def compute_rational_inconsistency(
    request: LinearRationalInconsistencyFindRequest,
) -> ComputedOutcome[LinearRationalInconsistencyResult]:
    try:
        payload = _run(request, _INCONSISTENCY_PROTOCOL)
        if payload is None:
            return _failure(
                "FLINT_LINEAR_PROVIDER_UNAVAILABLE",
                ExecutionStatus.ERROR,
                "The pinned Python-FLINT rational-linear provider is unavailable.",
            )
        if payload.get("status") == "NO_CERTIFICATE_PRODUCED":
            return _failure(
                "LINEAR_SYSTEM_NOT_INCONSISTENT",
                ExecutionStatus.ERROR,
                "The system has no inconsistency certificate candidate.",
            )
        if payload.get("status") != "CERTIFICATE_PRODUCED":
            raise ValueError("worker did not produce an inconsistency witness")
        return ComputedSuccess(
            LinearRationalInconsistencyResult(
                left_witness=payload["left_witness"],
                rhs_pairing=payload["rhs_pairing"],
            )
        )
    except TimeoutError:
        return _failure(
            "FLINT_LINEAR_TIMEOUT",
            ExecutionStatus.TIMEOUT,
            "The bounded rational-linear computation timed out.",
        )
    except (RuntimeError, TypeError, ValueError):
        return _failure(
            "FLINT_LINEAR_WORKER_FAILED",
            ExecutionStatus.ERROR,
            "The rational-linear worker returned no usable result.",
        )
