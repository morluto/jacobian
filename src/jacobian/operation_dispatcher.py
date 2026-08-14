"""Stateless final boundary for one ``math.run`` call."""

from __future__ import annotations

import time
from typing import Any, cast

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import (
    OperationId,
    OperationResult,
)
from jacobian.operation_adapters import parse_operation_input
from jacobian.serving_catalog import ServingCatalog


def invoke_operation(
    operation_id: OperationId,
    payload: dict[str, Any],
    catalog: ServingCatalog,
) -> OperationResult:
    """Select, parse, call, and project one typed mathematical operation."""

    started = time.monotonic()
    descriptor = catalog.inspect(operation_id)
    if descriptor is None:
        raise ValueError(f"unknown operation: {operation_id}")
    operation = catalog.operation(operation_id)
    if operation is None:
        raise RuntimeError(f"installed catalog has no declaration: {operation_id}")
    parsed = cast(
        ContractModel,
        parse_operation_input(operation.request_type, payload),
    )
    result = operation.run(parsed)

    return OperationResult(
        operation_id=operation.operation_id,
        operation_version=operation.version,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        output=result.model_dump(mode="json"),
    )


__all__ = ["invoke_operation"]
