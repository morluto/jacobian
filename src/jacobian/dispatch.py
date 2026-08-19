"""Strict parsing and stateless execution for one ``math.run`` call."""

from __future__ import annotations

import time
from typing import Any, cast

from pydantic import BaseModel

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationId, OperationResult


def parse_operation_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request once into its owning strict model."""

    try:
        encoded = encode_strict_json(payload)
    except CanonicalizationError as exc:
        raise ValueError("operation request is not valid bounded JSON") from exc
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
    parsed = cast(
        StrictModel,
        parse_operation_input(operation.request_type, payload),
    )
    result = operation.run(parsed)

    return OperationResult(
        operation_id=operation.operation_id,
        operation_version=operation.version,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        output=result.model_dump(mode="json"),
    )


__all__ = [
    "invoke_operation",
    "parse_operation_input",
]
