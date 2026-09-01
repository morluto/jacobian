"""Strict parsing and stateless execution for one ``math.run`` call."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

from pydantic import BaseModel, ValidationError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    RequestCancellationSignal,
    request_cancellation,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationId,
    OperationResult,
)


class OperationRequestValidationError(ValueError):
    """A selected operation rejected its caller-supplied request payload."""

    def __init__(
        self,
        cause: ValidationError | CanonicalizationError,
    ) -> None:
        self.cause = cause
        super().__init__("operation payload failed validation")

    def errors(self) -> Sequence[Mapping[str, Any]]:
        if isinstance(self.cause, ValidationError):
            return self.cause.errors(
                include_url=False,
                include_context=False,
                include_input=True,
            )
        return [
            {
                "loc": (),
                "type": "canonicalization_error",
                "msg": str(self.cause),
                "input": None,
            }
        ]


class _OperationResolutionError(ValueError):
    """The immutable catalog has no binding for the requested operation."""


def parse_operation_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request once into its owning strict model."""

    # This is the encoded request boundary, so the canonical input ceiling is
    # explicit. Result projection does not inherit this transport policy.
    encoded = encode_strict_json(payload, limits=CanonicalLimits())
    return model.model_validate_json(encoded, strict=True)


def invoke_operation(
    operation_id: OperationId,
    payload: dict[str, Any],
    catalog: Catalog,
) -> OperationResult:
    """Select, parse, call, and project one typed mathematical operation."""

    return execute_operation(
        operation_id,
        payload,
        catalog,
        projector=_operation_result_projector,
    )


def execute_operation[ProjectedT](
    operation_id: OperationId,
    payload: dict[str, Any],
    catalog: Catalog,
    *,
    projector: Callable[[OperationId, StrictModel, float], ProjectedT],
    cancellation_signal: RequestCancellationSignal | None = None,
) -> ProjectedT:
    """Own one complete parse, invocation, and projection execution envelope."""

    started = time.monotonic()
    cancellation_context = (
        request_cancellation(cancellation_signal)
        if cancellation_signal is not None
        else nullcontext()
    )
    with request_execution(started), cancellation_context:
        request_checkpoint("before parsing")
        binding = catalog._binding(operation_id)
        if binding is None:
            raise _OperationResolutionError(f"unknown operation: {operation_id}")
        try:
            request = parse_operation_input(binding.request_type, payload)
        except (CanonicalizationError, ValidationError) as exc:
            raise OperationRequestValidationError(exc) from exc
        request_checkpoint("after parsing")
        result = binding.run(request)
        request_checkpoint("after operation execution")
        projected = projector(operation_id, result, started)
        request_checkpoint("after result projection")
        return projected


def _operation_result_projector(
    operation_id: OperationId,
    result: StrictModel,
    started: float,
) -> OperationResult:
    output = result.model_dump(mode="json")
    return OperationResult(
        operation_id=operation_id,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        output=output,
    )


__all__ = [
    "OperationDomainValidationError",
    "OperationExecutionTimeoutError",
    "OperationRequestValidationError",
    "execute_operation",
    "invoke_operation",
    "parse_operation_input",
]
