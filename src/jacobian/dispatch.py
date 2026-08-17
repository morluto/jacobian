"""Strict parsing and stateless execution for one ``math.run`` call."""

from __future__ import annotations

import time
from collections.abc import Collection, Mapping
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationId, OperationResult

MAX_PUBLIC_VALIDATION_ERRORS = 8
MAX_PUBLIC_VALIDATION_MESSAGE_LENGTH = 1024


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


def _safe_validation_message(
    item: Mapping[str, Any],
    *,
    safe_messages: Collection[str] = (),
) -> str:
    """Describe a Pydantic failure without interpolating rejected values."""

    message = str(item.get("msg", ""))
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    if message in safe_messages:
        return message
    error_type = str(item.get("type", "validation_error"))
    return f"Request value violates validation rule {error_type}"


def project_validation_errors(
    error: ValidationError,
    *,
    safe_messages: Collection[str] = (),
) -> tuple[list[dict[str, Any]], int]:
    """Return bounded field errors without retaining rejected input values."""

    projected = []
    for raw_item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:MAX_PUBLIC_VALIDATION_ERRORS]:
        item = dict(raw_item)
        item["msg"] = _safe_validation_message(item, safe_messages=safe_messages)
        projected.append(item)
    return projected, error.error_count()


def validation_error_message(
    error: ValidationError,
    *,
    safe_messages: Collection[str] = (),
) -> str:
    """Summarize one validation failure without formatting its input value."""

    first = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[0]
    path = ".".join(str(part) for part in first["loc"]) or "request"
    count = error.error_count()
    noun = "error" if count == 1 else "errors"
    return (
        f"{count} validation {noun}; first at {path}: "
        f"{_safe_validation_message(first, safe_messages=safe_messages)}"
    )


def bounded_validation_exception_message(error: Exception) -> str:
    """Project validation failures without echoing rejected values."""

    if isinstance(error, ValidationError):
        return validation_error_message(error)
    message = str(error)
    if len(message) <= MAX_PUBLIC_VALIDATION_MESSAGE_LENGTH:
        return message
    suffix = "... [validation detail truncated]"
    return f"{message[: MAX_PUBLIC_VALIDATION_MESSAGE_LENGTH - len(suffix)]}{suffix}"


__all__ = [
    "MAX_PUBLIC_VALIDATION_ERRORS",
    "MAX_PUBLIC_VALIDATION_MESSAGE_LENGTH",
    "bounded_validation_exception_message",
    "invoke_operation",
    "parse_operation_input",
    "project_validation_errors",
    "validation_error_message",
]
