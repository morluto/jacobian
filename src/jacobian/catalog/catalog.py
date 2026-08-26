"""Immutable catalog compiled directly from packaged mathematical functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    MathTool,
    OperationBrowseResult,
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationExample,
)
from jacobian.catalog.search import browse_operations, discover_operations


@dataclass(frozen=True, slots=True)
class _BoundMathTool:
    """One checked existential binding at the heterogeneous catalog boundary."""

    request_type: type[StrictModel]
    result_type: type[StrictModel]
    _run: Callable[[StrictModel], StrictModel]

    def run(self, request: StrictModel) -> StrictModel:
        """Run one parsed request through its checked typed declaration."""

        return self._run(request)


def _bind_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation: MathTool[RequestT, ResultT],
) -> _BoundMathTool:
    """Erase one typed declaration only after retaining its runtime witnesses."""

    def run(request: StrictModel) -> StrictModel:
        if not isinstance(request, operation.request_type):
            raise TypeError(
                f"{operation.operation_id} received a request outside its declared type"
            )
        result = operation.run(request)
        if type(result) is not operation.result_type:
            raise TypeError(
                f"{operation.operation_id} returned a result outside its declared type"
            )
        return result

    return _BoundMathTool(
        request_type=operation.request_type,
        result_type=operation.result_type,
        _run=run,
    )


class Catalog:
    """Direct declaration view with no overlay or state directory."""

    def __init__(self, operations: Iterable[MathTool[Any, Any]]) -> None:
        self._operations = _index_operations(operations)
        self._bindings = {
            operation_id: _bind_operation(operation)
            for operation_id, operation in self._operations.items()
        }

    @classmethod
    def open(cls) -> Catalog:
        return cls(BUILTIN_TOOLS)

    def operation(self, operation_id: str) -> MathTool[Any, Any] | None:
        """Return the mathematical function selected by a known operation ID."""

        return self._operations.get(operation_id)

    def _binding(self, operation_id: str) -> _BoundMathTool | None:
        """Return the private checked binding for one selected declaration."""

        return self._bindings.get(operation_id)

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        operation = self.operation(operation_id)
        if operation is None:
            return None
        return _descriptor(operation)

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(tuple(self._operations.values()), request)

    def browse(
        self,
        *,
        domain: str | None,
        limit: int,
        cursor: str | None,
    ) -> OperationBrowseResult:
        """Return a fresh compact page from the immutable declaration snapshot."""

        return browse_operations(
            tuple(self._operations.values()),
            domain=domain,
            limit=limit,
            cursor=cursor,
        )

    def snapshot(self) -> OperationCatalogSnapshot:
        operations = tuple(
            _descriptor(operation) for operation in self._operations.values()
        )
        return OperationCatalogSnapshot(
            operations=tuple(sorted(operations, key=lambda item: item.operation_id)),
        )


def _index_operations(
    operations: Iterable[MathTool[Any, Any]],
) -> dict[str, MathTool[Any, Any]]:
    indexed: dict[str, MathTool[Any, Any]] = {}
    for operation in operations:
        if operation.operation_id in indexed:
            raise ValueError(
                f"duplicate built-in operation ID: {operation.operation_id}"
            )
        indexed[operation.operation_id] = operation
    return indexed


def _descriptor(operation: MathTool[Any, Any]) -> OperationDescriptor:
    """Describe one direct mathematical function for discovery."""

    return OperationDescriptor(
        operation_id=operation.operation_id,
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


__all__ = ["Catalog"]
