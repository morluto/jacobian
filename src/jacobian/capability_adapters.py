"""Narrow runtime adapter protocols shared by registration and dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

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


def _restore_json_tuple_paths(
    value: Any,
    paths: tuple[tuple[Any, ...], ...],
) -> Any:
    """Restore tuple-shaped model fields after JSON parsing.

    JSON arrays are the wire representation for both lists and tuples. Pydantic
    strict JSON validation accepts the former, but rejects some constrained
    tuple annotations even though their JSON representation is valid. Only the
    paths identified by that precise validation error are adapted; all scalar
    values remain subject to strict Python-mode validation below.
    """

    if any(not path for path in paths):
        return tuple(value) if isinstance(value, list) else value

    if isinstance(value, Mapping):
        restored_mapping = dict(value)
        for key in restored_mapping:
            child_paths = tuple(path[1:] for path in paths if path and path[0] == key)
            if child_paths:
                restored_mapping[key] = _restore_json_tuple_paths(
                    restored_mapping[key], child_paths
                )
        return restored_mapping

    if isinstance(value, (list, tuple)):
        restored_sequence = list(value)
        for index in range(len(restored_sequence)):
            child_paths = tuple(path[1:] for path in paths if path and path[0] == index)
            if child_paths:
                restored_sequence[index] = _restore_json_tuple_paths(
                    restored_sequence[index], child_paths
                )
        return (
            tuple(restored_sequence) if isinstance(value, tuple) else restored_sequence
        )

    return value


def _validate_strict_json_model[ModelT: BaseModel](
    model: type[ModelT], encoded: bytes, wire_payload: object
) -> ModelT:
    """Validate JSON strictly while honoring tuple fields' array wire form."""

    try:
        parsed = model.model_validate_json(encoded, strict=True)
    except ValidationError as exc:
        # Pydantic's strict JSON path treats some constrained tuple fields as
        # Python-only tuples, although JSON arrays are their valid wire form.
        # Normalize only paths reported as tuple mismatches and retry the whole
        # model in strict Python mode. Nested tuple fields are reported only
        # after their containing tuple has been normalized, so repeat until the
        # model validates or a non-tuple error remains.
        normalized_payload = wire_payload
        for _ in range(64):
            errors = exc.errors()
            tuple_paths = tuple(
                tuple(error["loc"]) for error in errors if error["type"] == "tuple_type"
            )
            if not tuple_paths:
                raise exc
            updated_payload = _restore_json_tuple_paths(
                normalized_payload,
                tuple_paths,
            )
            if updated_payload == normalized_payload:
                raise exc
            normalized_payload = updated_payload
            try:
                parsed = model.model_validate(normalized_payload, strict=True)
            except ValidationError as retry_exc:
                exc = retry_exc
            else:
                return parsed
        raise exc
    return parsed


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
    parsed = _validate_strict_json_model(model, encoded, wire_payload)
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
