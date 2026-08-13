"""Narrow runtime adapter protocols shared by registration and dispatch."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.operation_ports import _BoundTypedValue
from jacobian.operation_projection import OperationProjection

PreparedT = TypeVar("PreparedT")


def _restore_bound_typed_values(value: Any) -> tuple[Any, bool]:
    if isinstance(value, _BoundTypedValue):
        return value.typed_value, True
    if isinstance(value, list):
        restored_items: list[Any] = []
        found = False
        for item in value:
            restored, nested_found = _restore_bound_typed_values(item)
            restored_items.append(restored)
            found = found or nested_found
        return restored_items, found
    if isinstance(value, dict):
        restored_object: dict[str, Any] = {}
        found = False
        for key, item in value.items():
            restored, nested_found = _restore_bound_typed_values(item)
            restored_object[key] = restored
            found = found or nested_found
        return restored_object, found
    return value, False


def parse_capability_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request into its owning model.

    Ordinary caller input crosses the strict JSON parser exactly once. Typed
    composition values retain their already-validated object identity while
    their attached JSON views still participate in canonical resource
    accounting before this function is called.
    """

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
    restored, contains_typed_values = _restore_bound_typed_values(payload)
    if contains_typed_values:
        return model.model_validate(restored, strict=True)
    return model.model_validate_json(encoded, strict=True)


class CapabilityAdapter(Protocol[PreparedT]):
    """Installed typed adapter; registration requires no MCP changes."""

    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def prepare(self, request: CapabilityRequest) -> PreparedT: ...

    def invoke(self, prepared: PreparedT) -> OperationProjection: ...


__all__ = ["CapabilityAdapter", "parse_capability_input"]
