"""FLINT conversion boundary for ordinary rational-linear values."""

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


class _RuntimeChangedError(RuntimeError):
    """The pinned FLINT runtime changed while the worker was running."""


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


def _validate_worker_payload(
    payload: object,
    request: LinearRationalSolutionFindRequest | LinearRationalInconsistencyFindRequest,
    protocol: str,
) -> dict[str, Any]:
    """Validate worker identity, status, fields, and source-bound dimensions."""

    if not isinstance(payload, dict) or payload.get("protocol") != protocol:
        raise ValueError("rational-linear worker protocol is invalid")
    status = payload.get("status")
    if protocol == _SOLUTION_PROTOCOL:
        if status == "NO_SOLUTION_PRODUCED":
            expected_keys = {"protocol", "status"}
        elif status == "SOLUTION_PRODUCED":
            expected_keys = {"protocol", "status", "values"}
            solution_result = LinearRationalSolutionResult.model_validate(
                {"values": payload.get("values")}
            )
            if solution_result.values is None or len(solution_result.values) != len(
                request.system.variables
            ):
                raise ValueError("solution dimensions do not match the source system")
        else:
            raise ValueError("rational-linear solution status is invalid")
    else:
        if status == "NO_CERTIFICATE_PRODUCED":
            expected_keys = {"protocol", "status"}
        elif status == "CERTIFICATE_PRODUCED":
            expected_keys = {"protocol", "status", "left_witness", "rhs_pairing"}
            inconsistency_result = LinearRationalInconsistencyResult.model_validate(
                {
                    "left_witness": payload.get("left_witness"),
                    "rhs_pairing": payload.get("rhs_pairing"),
                }
            )
            if inconsistency_result.left_witness is None or len(
                inconsistency_result.left_witness
            ) != len(request.system.rhs):
                raise ValueError(
                    "inconsistency witness dimensions do not match the source system"
                )
        else:
            raise ValueError("rational-linear inconsistency status is invalid")
    if set(payload) != expected_keys:
        raise ValueError("rational-linear worker response shape is invalid")
    return payload


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
    if python_flint_provider_runtime(refresh=True) != RUNTIME:
        raise _RuntimeChangedError
    return _validate_worker_payload(
        loads_strict_json(completed.stdout), request, protocol
    )


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
            return ComputedSuccess(
                LinearRationalSolutionResult(status="NO_SOLUTION_PRODUCED")
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
    except _RuntimeChangedError:
        return _failure(
            "FLINT_LINEAR_RUNTIME_CHANGED",
            ExecutionStatus.ERROR,
            "The Python-FLINT rational-linear runtime changed during the bounded computation.",
        )
    except (RuntimeError, TypeError, ValueError, ValidationError):
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
            return ComputedSuccess(
                LinearRationalInconsistencyResult(status="NO_CERTIFICATE_PRODUCED")
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
    except _RuntimeChangedError:
        return _failure(
            "FLINT_LINEAR_RUNTIME_CHANGED",
            ExecutionStatus.ERROR,
            "The Python-FLINT rational-linear runtime changed during the bounded computation.",
        )
    except (RuntimeError, TypeError, ValueError, ValidationError):
        return _failure(
            "FLINT_LINEAR_WORKER_FAILED",
            ExecutionStatus.ERROR,
            "The rational-linear worker returned no usable result.",
        )
