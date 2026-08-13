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


def parse_capability_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request into its owning model.

    Ordinary caller input crosses the strict JSON parser exactly once. Requests
    containing values resolved from typed input ports bind those already-
    validated objects directly in strict Python mode; the operation adapter has
    separately accounted for their canonical JSON projection before this call.
    """

    try:
        typed_values = {
            key: value
            for key, value in payload.items()
            if isinstance(value, ContractModel)
        }
        wire_payload = {
            key: (
                value.model_dump(mode="json")
                if isinstance(value, ContractModel)
                else value
            )
            for key, value in payload.items()
        }
        encoded = encode_strict_json(wire_payload)
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
    parsed = model.model_validate_json(encoded, strict=True)
    if typed_values:
        # Parse all ordinary fields through the strict JSON boundary, then
        # restore the already-validated port values. This preserves identity
        # without asking strict Python-mode validation to accept JSON lists as
        # tuple/dataclass fields.
        return parsed.model_copy(update=typed_values)
    return parsed


class CapabilityAdapter(Protocol[PreparedT]):
    """Installed typed adapter; registration requires no MCP changes."""

    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def prepare(self, request: CapabilityRequest) -> PreparedT: ...

    def invoke(self, prepared: PreparedT) -> OperationProjection: ...


__all__ = ["CapabilityAdapter", "parse_capability_input"]
