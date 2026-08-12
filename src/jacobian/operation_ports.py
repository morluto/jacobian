"""Minimal supported composition ports for typed operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.contracts.base import ContractModel


@dataclass(frozen=True, slots=True)
class InputPort[
    ValueT: ContractModel,
]:
    """Bind one exact semantic value to one request-model field."""

    name: str
    value_type: type[ValueT]
    request_field: str

    def __post_init__(self) -> None:
        if not self.name or not self.request_field:
            raise ValueError("input port name and request field must be nonempty")

    def bind_to_request(
        self,
        payload: dict[str, Any],
        value: ValueT,
    ) -> dict[str, Any]:
        if type(value) is not self.value_type:
            raise TypeError(
                f"input port {self.name!r} requires {self.value_type.__name__}"
            )
        if self.request_field in payload:
            raise ValueError(
                f"input port {self.name!r} conflicts with payload field "
                f"{self.request_field!r}"
            )
        return {**payload, self.request_field: value}


@dataclass(frozen=True, slots=True)
class OutputPort[
    ValueT: ContractModel,
]:
    """Expose one exact semantic result value for typed composition."""

    name: str
    value_type: type[ValueT]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("output port name must be nonempty")

    def extract_from_result(self, result: ContractModel) -> ValueT:
        if type(result) is not self.value_type:
            raise TypeError(
                f"output port {self.name!r} produced {type(result).__name__}; "
                f"expected {self.value_type.__name__}"
            )
        return result


def validate_ports(
    request_type: type[ContractModel],
    result_type: type[ContractModel],
    input_ports: tuple[InputPort[Any], ...],
    output_ports: tuple[OutputPort[Any], ...],
) -> None:
    """Reject port declarations that disagree with their Pydantic fields."""

    _require_unique_names(input_ports, "input")
    _require_unique_names(output_ports, "output")
    if len(output_ports) > 1:
        raise ValueError("an operation result may expose at most one output port")
    bound_request_fields: set[str] = set()
    for input_port in input_ports:
        if input_port.request_field in bound_request_fields:
            raise ValueError(
                f"request field {input_port.request_field!r} has multiple ports"
            )
        bound_request_fields.add(input_port.request_field)
        field = request_type.model_fields.get(input_port.request_field)
        if field is None or field.annotation is not input_port.value_type:
            raise ValueError(
                f"input port {input_port.name!r} does not match "
                f"{request_type.__name__}.{input_port.request_field}"
            )
    for output_port in output_ports:
        if result_type is not output_port.value_type:
            raise ValueError(
                f"output port {output_port.name!r} does not match its result value type"
            )


def _require_unique_names(ports: tuple[Any, ...], label: str) -> None:
    names = tuple(port.name for port in ports)
    if len(names) != len(set(names)):
        raise ValueError(f"{label} port names must be unique")


__all__ = ["InputPort", "OutputPort"]
