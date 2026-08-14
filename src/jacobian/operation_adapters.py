"""Narrow runtime adapter protocols shared by registration and dispatch."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from jacobian.canonical import (
    CanonicalizationError,
    encode_strict_json,
    loads_strict_json,
)
from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
)
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection

PreparedT = TypeVar("PreparedT")


def parse_operation_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request into its owning model.

    Canonical JSON is enforced once for size and wire shape. Typed port values
    stay as already-validated objects. JSON arrays, enums, and nested objects
    are decoded by the owning model; advertised integers remain strict.
    """

    try:
        jsonable = {
            key: (
                value.model_dump(mode="json")
                if isinstance(value, ContractModel)
                else value
            )
            for key, value in payload.items()
        }
        encoded = encode_strict_json(jsonable)
    except CanonicalizationError as exc:
        raise OperationInvocationError(
            OperationDiagnostic(
                code="INVALID_REQUEST",
                stage="operation_input_validation",
                message="The operation request is not valid bounded JSON.",
                hint=(
                    "Use only JSON objects, arrays, strings, booleans, null, "
                    "and supported finite numbers within the configured limits."
                ),
            )
        ) from exc
    if not any(isinstance(value, ContractModel) for value in payload.values()):
        return model.model_validate_json(encoded, strict=True)
    decoded = loads_strict_json(encoded)
    if not isinstance(decoded, dict):
        raise OperationInvocationError(
            OperationDiagnostic(
                code="INVALID_REQUEST",
                stage="operation_input_validation",
                message="The operation request is not valid bounded JSON.",
                hint="Provide a JSON object request payload.",
            )
        )
    assembled = {
        key: (
            payload[key]
            if isinstance(payload.get(key), ContractModel)
            else decoded[key]
        )
        for key in decoded
    }
    return model.model_validate(assembled, strict=True)


class OperationAdapter(Protocol[PreparedT]):
    """Installed typed adapter; registration requires no MCP changes."""

    @property
    def descriptor(self) -> OperationDescriptor: ...

    def prepare(self, request: OperationRequest) -> PreparedT: ...

    def invoke(self, prepared: PreparedT) -> OperationProjection: ...


__all__ = ["OperationAdapter", "parse_operation_input"]
