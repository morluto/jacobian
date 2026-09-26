"""Killable process owner for symbolic hypergeometric action normalization."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_cancellation,
    execution_deadline,
    request_checkpoint,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import ShiftOreOperator
from jacobian.math.ore_algebras.proper_hypergeometric_terms._models import (
    ProperHypergeometricTerm,
)
from jacobian.math.polynomials.values import RationalFunction
from jacobian.process import (
    ProcessResourceLimits,
    run_checked_worker_process,
    worker_environment,
)

_WORKER_PATH = Path(__file__).resolve().with_name("_action_worker.py")
_WALL_SECONDS = 120.0
# The mathematical admission runs in the parent before serialization. Keep a
# separate, finite transport ceiling for admitted payloads and worker output.
_STREAM_LIMIT = 64 * 1024 * 1024
_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024


def _decode_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("hypergeometric action worker result must be an object")
    return value


def run_action_worker(
    operator: ShiftOreOperator, term: ProperHypergeometricTerm
) -> RationalFunction:
    """Run cancellation and polynomial normalization in a bounded child."""
    request_checkpoint("before hypergeometric action worker")
    deadline = execution_deadline(_WALL_SECONDS)
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise OperationExecutionTimeoutError(
            "hypergeometric action deadline expired before normalization"
        )
    payload = encode_strict_json(
        {
            "operator": operator.model_dump_json(),
            "term": term.model_dump_json(),
        }
    )
    try:
        response = run_checked_worker_process(
            [sys.executable, str(_WORKER_PATH)],
            input_bytes=payload,
            timeout_seconds=timeout,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=_STREAM_LIMIT,
            stderr_limit=64 * 1024,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, math.ceil(timeout)),
                address_space_bytes=_ADDRESS_SPACE_BYTES,
                file_size_bytes=_STREAM_LIMIT,
            ),
            cancellation_event=current_request_cancellation(),
            decode_result=_decode_result,
        )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        raise RuntimeError("hypergeometric action worker could not be started") from exc

    if response.get("ok") is True and isinstance(
        response.get("relative_multiplier"), str
    ):
        try:
            return RationalFunction.model_validate_json(response["relative_multiplier"])
        except (TypeError, ValueError) as exc:
            from jacobian._execution import BackendFailureReason, OperationBackendError

            raise OperationBackendError(
                BackendFailureReason.MALFORMED_RESPONSE
            ) from exc
    kind = response.get("kind")
    location = response.get("location")
    code = response.get("code")
    message = response.get("message")
    if (
        isinstance(location, list)
        and all(isinstance(item, (str, int)) for item in location)
        and isinstance(code, str)
        and isinstance(message, str)
    ):
        error_type = (
            OperationResourceAdmissionError
            if kind == "resource"
            else OperationDomainValidationError
        )
        raise error_type(location=tuple(location), code=code, message=message)
    from jacobian._execution import BackendFailureReason, OperationBackendError

    raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE)
