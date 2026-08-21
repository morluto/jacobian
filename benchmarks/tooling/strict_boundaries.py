"""Shared strict-model primitives for benchmark control-plane loaders.

This module provides the closed ``_StrictModel`` base and diagnostic helpers
used by owner-local strict configuration models.  Each model family lives
beside the loader that interprets its cross-field semantics.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
)

from benchmarks.tooling.errors import HarborSuiteError


class _StrictModel(BaseModel):
    """Closed base: forbid extras and reject loose scalar coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


def format_strict_errors(exc: ValidationError, *, label: str) -> list[str]:
    """Render a Pydantic ``ValidationError`` as field-path/code diagnostics."""

    failures: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        path = f"{label}.{loc}" if loc else label
        failures.append(f"{path}: {error['msg']} ({error['type']})")
    return failures


def strict_model_failures(
    model: type[BaseModel],
    payload: Any,
    *,
    label: str,
) -> list[str]:
    """Validate *payload* against *model* and return field-path diagnostics."""

    try:
        model.model_validate(payload)
    except ValidationError as exc:
        return format_strict_errors(exc, label=label)
    return []


def raise_strict_model(
    model: type[BaseModel],
    payload: Any,
    *,
    label: str,
) -> BaseModel:
    """Validate *payload* and raise ``HarborSuiteError`` on structural failure."""

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        failures = format_strict_errors(exc, label=label)
        raise HarborSuiteError(
            f"{label}: strict configuration validation failed: " + "; ".join(failures)
        ) from exc


__all__ = [
    "format_strict_errors",
    "raise_strict_model",
    "strict_model_failures",
]
