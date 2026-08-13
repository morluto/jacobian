"""Errors shared by capability registration and invocation boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.validation_diagnostics import validation_error_message


class CapabilityError(RuntimeError):
    """A capability descriptor, request, or verification boundary is invalid."""


class PayloadValidationError(CapabilityError):
    """Structured descriptor-schema failure safe for public diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        actual_type: str,
        expected: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.actual_type = actual_type
        self.expected = expected
        self.details = details or {}


class CapabilityDiscoveryCursorError(ValueError):
    """A continuation cursor does not belong to the filtered result."""


class CapabilityInvocationError(RuntimeError):
    """An expected adapter failure that is safe to return to a model."""

    def __init__(self, diagnostic: CapabilityDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def enriched_invalid_request(
    base: CapabilityDiagnostic,
    exc: ValidationError,
) -> CapabilityDiagnostic:
    """Add the first Pydantic error location to an invocation diagnostic."""

    errors = exc.errors()
    if not errors:
        return base
    first = errors[0]
    loc = first.get("loc", ())
    path = "/".join(str(part) for part in loc) if loc else None
    return base.model_copy(
        update={
            "path": path,
            "hint": validation_error_message(exc),
        }
    )


__all__ = [
    "CapabilityDiscoveryCursorError",
    "CapabilityError",
    "CapabilityInvocationError",
    "PayloadValidationError",
    "enriched_invalid_request",
]
