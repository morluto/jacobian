"""Narrow runtime adapter protocols shared by registration and dispatch."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.base import ContractModel
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.operation_projection import OperationProjection

PreparedT = TypeVar("PreparedT")


def _contains_typed_value(value: Any) -> bool:
    if isinstance(value, ContractModel):
        return True
    if isinstance(value, list):
        return any(_contains_typed_value(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_typed_value(item) for item in value.values())
    return False


def parse_capability_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request into its owning model.

    Ordinary caller input crosses the strict JSON parser exactly once. Requests
    containing values resolved from typed input ports bind those already-
    validated objects directly in strict Python mode; the operation adapter has
    separately accounted for their canonical JSON projection before this call.
    """

    if _contains_typed_value(payload):
        return model.model_validate(payload, strict=True)
    try:
        encoded = encode_strict_json(payload)
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
