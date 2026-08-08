"""Artifact-backed row-HNF producer owned by the matrix-lattice domain."""

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
from jacobian.contracts.matrix_lattice import (
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
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
from jacobian.providers.flint_runtime import python_flint_hnf_provider_runtime
from jacobian.worker_environment import worker_environment

HNF_RUNTIME = python_flint_hnf_provider_runtime()
HNF_WORKER_PROTOCOL = "jacobian.matrix-lattice-hnf-worker/v1"
HNF_STDOUT_LIMIT = 80_000_000
HNF_STDERR_LIMIT = 64_000


def _failure(
    status: ExecutionStatus, code: str, message: str
) -> OperationExecutionFailure:
    return OperationExecutionFailure(
        status=status,
        diagnostic=CapabilityDiagnostic(
            code=code,
            stage="matrix_hnf_provider",
            message=message,
            hint="Install the pinned Python-FLINT HNF provider and retry.",
        ),
    )


def _parse_hnf_worker_result(
    output: object,
    source: IntegerMatrix,
) -> HermiteNormalFormResult:
    """Validate the complete worker envelope before exposing its certificate."""

    expected_keys = {
        "protocol",
        "status",
        "backend_version",
        "flint_library_version",
        "normal_form",
        "transformation",
    }
    if not isinstance(output, dict) or set(output) != expected_keys:
        raise ValueError("invalid HNF worker response shape")
    if (
        output["protocol"] != HNF_WORKER_PROTOCOL
        or output["status"] != "NORMAL_FORM_PRODUCED"
        or output["backend_version"] != "0.9.0"
        or output["flint_library_version"] != "3.6.0"
    ):
        raise ValueError("invalid HNF worker response identity")

    normal_form = IntegerMatrix.model_validate({"entries": output["normal_form"]})
    transformation = IntegerMatrix.model_validate({"entries": output["transformation"]})
    source_rows = len(source.entries)
    source_columns = len(source.entries[0])
    if len(normal_form.entries) != source_rows or any(
        len(row) != source_columns for row in normal_form.entries
    ):
        raise ValueError("HNF normal form dimensions do not match the source")
    if len(transformation.entries) != source_rows or any(
        len(row) != source_rows for row in transformation.entries
    ):
        raise ValueError("HNF transformation dimensions do not match the source")
    return HermiteNormalFormResult(
        normal_form=normal_form,
        transformation=transformation,
    )


def compute_hermite_normal_form(
    request: HermiteNormalFormRequest,
) -> ComputedOutcome[HermiteNormalFormResult]:
    runtime = python_flint_hnf_provider_runtime(refresh=True)
    if (
        HNF_RUNTIME.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime != HNF_RUNTIME
    ):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_PROVIDER_UNAVAILABLE",
            "The pinned Python-FLINT HNF provider is unavailable.",
        )
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=("-I", "-m", "jacobian.domains.matrix_lattice.hnf_worker"),
            stdin_bytes=canonicalize_json(
                {
                    "protocol": HNF_WORKER_PROTOCOL,
                    "matrix": request.matrix.model_dump(mode="json"),
                }
            ),
            timeout_seconds=float(request.resource_budget.wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=HNF_STDOUT_LIMIT,
            stderr_limit_bytes=HNF_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=request.resource_budget.wall_seconds + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            ExecutionStatus.TIMEOUT,
            "FLINT_HNF_TIMEOUT",
            "The bounded HNF computation timed out; no result artifact was retained.",
        )
    if completed.termination is ProcessTermination.START_FAILED:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_WORKER_START_FAILED",
            "The isolated Python-FLINT HNF worker could not be started.",
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_WORKER_FAILED",
            "The isolated Python-FLINT HNF worker did not complete successfully.",
        )
    if python_flint_hnf_provider_runtime(refresh=True) != HNF_RUNTIME:
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_RUNTIME_CHANGED",
            "The Python-FLINT HNF runtime changed during the bounded computation.",
        )
    try:
        result = _parse_hnf_worker_result(
            loads_strict_json(completed.stdout), request.matrix
        )
    except (TypeError, ValueError, ValidationError):
        return _failure(
            ExecutionStatus.ERROR,
            "FLINT_HNF_PROTOCOL_INVALID",
            "The HNF worker returned an invalid exact result.",
        )
    return ComputedSuccess(result)


HERMITE_NORMAL_FORM_CAPABILITY: MaterializedOperation[
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
    HermiteNormalFormResult,
    Any,
] = MaterializedOperation(
    capability_id="matrix.normal_form.hermite.materialize",
    title="Materialize an exact row Hermite normal form",
    description=(
        "Use pinned Python-FLINT to retain H and U for one bounded integer matrix, "
        "with the proposed relation H = U A."
    ),
    request_model=HermiteNormalFormRequest,
    result_model=HermiteNormalFormResult,
    implementation=compute_hermite_normal_form,
    relation_id="matrix.normal_form.hermite.relation",
    tags=("matrix", "integer", "hermite-normal-form", "certificate", "python-flint"),
    resource_reason=(
        "the complete H and U basis-transformation certificate exceeds reliable "
        "inline transport and is retained for independent replay"
    ),
    provider_runtime=HNF_RUNTIME,
    invocation_examples=(
        example(
            "unit_matrix",
            "Materialize the row HNF of the one-by-one unit matrix.",
            {"matrix": {"entries": [["1"]]}},
        ),
    ),
    version="1",
)

__all__ = ["HERMITE_NORMAL_FORM_CAPABILITY", "compute_hermite_normal_form"]
