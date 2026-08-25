"""Strict parsing and stateless execution for one ``math.run`` call."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationId, OperationResult


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
    operation = catalog.operation(operation_id)
    if operation is None:
        raise ValueError(f"unknown operation: {operation_id}")
    try:
        parsed = cast(
            StrictModel,
            parse_operation_input(operation.request_type, payload),
        )
    except (CanonicalizationError, ValidationError) as exc:
        raise OperationRequestValidationError(exc) from exc
    result = operation.run(parsed)

    return OperationResult(
        operation_id=operation.operation_id,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        output=result.model_dump(mode="json"),
    )


__all__ = [
    "OperationRequestValidationError",
    "invoke_operation",
    "parse_operation_input",
]
