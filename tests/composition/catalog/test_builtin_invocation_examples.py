"""Executable contracts for examples advertised by the builtin catalog."""

from __future__ import annotations

import pytest

from jacobian.math_tools import MathTool
from jacobian.serving_catalog import ServingCatalog


def _builtin_operations() -> tuple[MathTool, ...]:
    catalog = ServingCatalog.open()
    return tuple(
        operation
        for descriptor in catalog.snapshot().operations
        if (operation := catalog.operation(descriptor.operation_id)) is not None
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
        request = operation.request_type.model_validate(invocation_example.input)
        outcome = operation.run(request)
        assert isinstance(outcome, operation.result_type), (operation_id, outcome)
        serialized = outcome.model_dump(mode="json")
        assert serialized, f"{operation_id} example produced an empty result"
        validated = operation.result_type.model_validate(serialized)
        assert validated.model_dump(mode="json") == serialized, (
            operation_id,
            serialized,
            validated.model_dump(mode="json"),
        )
