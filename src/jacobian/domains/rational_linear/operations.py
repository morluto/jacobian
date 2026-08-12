"""FLINT conversion boundary for ordinary rational-linear values."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Never

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
from jacobian.domains.rational_linear.protocol import (
    RationalLinearCertificateProduced,
    RationalLinearInconsistencyWorkerRequest,
    RationalLinearInconsistencyWorkerResponse,
    RationalLinearNoCertificateProduced,
    RationalLinearNoSolutionProduced,
    RationalLinearSolutionProduced,
    RationalLinearSolutionWorkerRequest,
    RationalLinearSolutionWorkerResponse,
    RationalLinearWorkerRequest,
    parse_inconsistency_worker_response,
    parse_solution_worker_response,
)
from jacobian.operations import (
    OperationAbortError,
)
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.providers.flint_runtime import python_flint_provider_runtime
from jacobian.worker_environment import worker_environment

RUNTIME = python_flint_provider_runtime()


class _RuntimeChangedError(RuntimeError):
    """The pinned FLINT runtime changed while the worker was running."""


def _failure(code: str, status: ExecutionStatus, message: str) -> Never:
    raise OperationAbortError(
        status,
        CapabilityDiagnostic(
            code=code,
            stage="rational_linear_provider",
            message=message,
            hint="Install the pinned Python-FLINT rational-linear provider and retry.",
        ),
    )


def _run(
    worker_request: RationalLinearWorkerRequest,
) -> (
    RationalLinearSolutionWorkerResponse
    | RationalLinearInconsistencyWorkerResponse
    | None
):
    runtime = python_flint_provider_runtime(refresh=True)
    if (
        RUNTIME.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime != RUNTIME
    ):
        return None
    budget = worker_request.request.resource_budget.wall_seconds
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=("-I", "-m", "jacobian.domains.rational_linear.worker"),
            stdin_bytes=canonicalize_json(worker_request.model_dump(mode="json")),
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
    payload = loads_strict_json(completed.stdout)
    if isinstance(worker_request, RationalLinearSolutionWorkerRequest):
        return parse_solution_worker_response(
            payload,
            expected_value_count=len(worker_request.request.system.variables),
        )
    return parse_inconsistency_worker_response(
        payload,
        expected_witness_count=len(worker_request.request.system.rhs),
    )


def compute_rational_solution(
    request: LinearRationalSolutionFindRequest,
) -> LinearRationalSolutionResult:
    try:
        response = _run(
            RationalLinearSolutionWorkerRequest(
                protocol="jacobian.rational-linear-solution-worker/v1",
                request=request,
            )
        )
        if response is None:
            return _failure(
                "FLINT_LINEAR_PROVIDER_UNAVAILABLE",
                ExecutionStatus.ERROR,
                "The pinned Python-FLINT rational-linear provider is unavailable.",
            )
        if isinstance(response, RationalLinearNoSolutionProduced):
            return LinearRationalSolutionResult(status="NO_SOLUTION_PRODUCED")
        if not isinstance(response, RationalLinearSolutionProduced):
            raise ValueError("worker did not produce a solution")
        return LinearRationalSolutionResult(values=response.values)
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
) -> LinearRationalInconsistencyResult:
    try:
        response = _run(
            RationalLinearInconsistencyWorkerRequest(
                protocol="jacobian.rational-linear-inconsistency-worker/v1",
                request=request,
            )
        )
        if response is None:
            return _failure(
                "FLINT_LINEAR_PROVIDER_UNAVAILABLE",
                ExecutionStatus.ERROR,
                "The pinned Python-FLINT rational-linear provider is unavailable.",
            )
        if isinstance(response, RationalLinearNoCertificateProduced):
            return LinearRationalInconsistencyResult(status="NO_CERTIFICATE_PRODUCED")
        if not isinstance(response, RationalLinearCertificateProduced):
            raise ValueError("worker did not produce an inconsistency witness")
        return LinearRationalInconsistencyResult(
            left_witness=response.left_witness,
            rhs_pairing=response.rhs_pairing,
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
