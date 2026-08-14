"""Immutable catalog compiled directly from packaged mathematical functions."""

from __future__ import annotations

from typing import Any

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationExample,
)
from jacobian.math_tools import MathTool
from jacobian.operation_discovery import discover_operations


class ServingCatalog:
    """Direct declaration view with no overlay or state directory."""

    def __init__(self, operations: dict[str, MathTool[Any, Any]]) -> None:
        self._operations = operations

    @classmethod
    def open(cls) -> ServingCatalog:
        operations = {
            operation.operation_id: operation
            for _module, declared in load_builtin_operation_modules()
            for operation in declared
            if isinstance(operation, MathTool)
        }
        return cls(operations)

    def operation(self, operation_id: str) -> MathTool[Any, Any] | None:
        """Return the mathematical function selected by a known operation ID."""

        return self._operations.get(operation_id)

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        operation = self.operation(operation_id)
        if operation is None:
            return None
        return _descriptor(operation)

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(self.snapshot(), request)

    def snapshot(self) -> OperationCatalogSnapshot:
        operations = tuple(
            _descriptor(operation) for operation in self._operations.values()
        )
        return OperationCatalogSnapshot(
            operations=tuple(sorted(operations, key=lambda item: item.operation_id)),
        )


def _descriptor(operation: MathTool[Any, Any]) -> OperationDescriptor:
    """Describe one direct mathematical function for discovery."""

    return OperationDescriptor(
        operation_id=operation.operation_id,
        version=operation.version,
        title=operation.title,
        description=operation.description,
        input_schema=operation.request_type.model_json_schema(),
        output_schema=operation.result_type.model_json_schema(),
        read_only=True,
        tags=operation.tags,
        examples=tuple(
            OperationExample(
                name=example.name,
                description=example.description,
                input=dict(example.input),
            )
            for example in operation.examples
        ),
    )


__all__ = ["ServingCatalog"]
