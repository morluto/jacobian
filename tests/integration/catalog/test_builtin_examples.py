"""Executable contracts for examples advertised by the builtin catalog."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.dispatch import invoke_operation

_CATALOG = Catalog.open()


def _builtin_operations() -> tuple[MathTool, ...]:
    return tuple(
        operation
        for descriptor in _CATALOG.snapshot().operations
        if (operation := _CATALOG.operation(descriptor.operation_id)) is not None
    )


@pytest.mark.parametrize(
    "operation",
    _builtin_operations(),
    ids=lambda operation: operation.operation_id,
)
def test_advertised_invocation_example_executes_successfully(
    operation: MathTool,
) -> None:
    operation_id = operation.operation_id
    examples = operation.examples
    assert examples, f"{operation_id} must advertise one executable example"
    for invocation_example in examples:
        public_result = invoke_operation(
            operation_id,
            invocation_example.input,
            _CATALOG,
        )
        assert public_result.operation_id == operation_id
        serialized = public_result.output
        assert serialized, f"{operation_id} example produced an empty result"
        validated = operation.result_type.model_validate(serialized)
        assert validated.model_dump(mode="json") == serialized, (
            operation_id,
            serialized,
            validated.model_dump(mode="json"),
        )
