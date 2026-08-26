"""Strict parsing and stateless execution for one ``math.run`` call."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalizationError, encode_strict_json
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


@dataclass(frozen=True, slots=True)
class _PreparedOperation:
    """One request parsed against its selected immutable operation binding."""

    operation_id: OperationId
    run: Callable[[StrictModel], StrictModel]
    request: StrictModel


def parse_operation_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request once into its owning strict model."""

    encoded = encode_strict_json(payload)
    return model.model_validate_json(encoded, strict=True)


def invoke_operation(
    operation_id: OperationId,
    payload: dict[str, Any],
    catalog: Catalog,
) -> OperationResult:
    """Select, parse, call, and project one typed mathematical operation."""

    started = time.monotonic()
    return _invoke_prepared_operation(
        _prepare_operation(operation_id, payload, catalog),
        started=started,
    )


def _prepare_operation(
    operation_id: OperationId,
    payload: dict[str, Any],
    catalog: Catalog,
) -> _PreparedOperation:
    """Select and parse one request before its kernel execution is scheduled."""

    binding = catalog._binding(operation_id)
    if binding is None:
        raise ValueError(f"unknown operation: {operation_id}")
    try:
        parsed = parse_operation_input(binding.request_type, payload)
    except (CanonicalizationError, ValidationError) as exc:
        raise OperationRequestValidationError(exc) from exc
    return _PreparedOperation(
        operation_id=operation_id,
        run=binding.run,
        request=parsed,
    )


def _invoke_prepared_operation(
    prepared: _PreparedOperation,
    *,
    started: float | None = None,
) -> OperationResult:
    """Run one already-admitted request and project its typed result."""

    if started is None:
        started = time.monotonic()
    result = prepared.run(prepared.request)

    return OperationResult(
        operation_id=prepared.operation_id,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        output=result.model_dump(mode="json"),
    )


__all__ = [
    "OperationDomainValidationError",
    "OperationRequestValidationError",
    "invoke_operation",
    "parse_operation_input",
]
