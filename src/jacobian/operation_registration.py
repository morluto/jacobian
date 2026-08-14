"""Shared validation for one explicitly bound operation adapter."""

from __future__ import annotations

from typing import Any

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_errors import OperationError
from jacobian.operation_validation import validate_payload, validator


def register_operation(
    adapter: OperationAdapter[Any],
    adapters: dict[str, OperationAdapter[Any]],
    descriptors: dict[str, OperationDescriptor],
) -> None:
    descriptor = adapter.descriptor
    if descriptor.operation_id in adapters:
        raise OperationError(f"duplicate operation ID: {descriptor.operation_id}")
    validator(descriptor.input_schema)
    validator(descriptor.output_schema)
    for example in descriptor.examples:
        try:
            validate_payload(descriptor.input_schema, example.input)
        except OperationError as exc:
            raise OperationError(
                f"operation {descriptor.operation_id} invocation example "
                f"{example.name!r} does not match its input schema"
            ) from exc
    descriptors[descriptor.operation_id] = descriptor.model_copy(deep=True)
    adapters[descriptor.operation_id] = adapter


__all__ = ["register_operation"]
