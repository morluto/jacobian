"""Narrow runtime adapter protocols shared by registration and dispatch."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from jacobian.canonical import CanonicalizationError, canonicalize_json
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.operation_projection import OperationProjection

PreparedT = TypeVar("PreparedT")


def parse_capability_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one JSON capability payload strictly into its owning model."""

    try:
        encoded = canonicalize_json(payload)
    except CanonicalizationError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INVALID_REQUEST",
                stage="capability_input_validation",
                message="The capability request is not valid bounded JSON.",
                hint=(
                    "Use only JSON objects, arrays, strings, booleans, null, "
                    "and supported finite numbers within the configured limits."
                ),
            )
        ) from exc
    return model.model_validate_json(encoded, strict=True)


class CapabilityAdapter(Protocol[PreparedT]):
    """Installed typed adapter; registration requires no MCP changes."""

    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def prepare(self, request: CapabilityRequest) -> PreparedT: ...

    def invoke(self, prepared: PreparedT) -> OperationProjection: ...


__all__ = ["CapabilityAdapter", "parse_capability_input"]
