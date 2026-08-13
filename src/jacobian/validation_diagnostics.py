"""Bounded public projections for model-validation failures."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from pydantic import ValidationError

MAX_PUBLIC_VALIDATION_ERRORS = 8
MAX_PUBLIC_VALIDATION_MESSAGE_LENGTH = 1024


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
