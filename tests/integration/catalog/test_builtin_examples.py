"""Executable contracts for examples advertised by the builtin catalog."""

from __future__ import annotations

import shutil
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.dispatch import OperationRequestValidationError, invoke_operation

_CATALOG = Catalog.open()
_SINGULAR_OPERATION_IDS = frozenset(
    {
        "polynomial.ideal.minimal_primes.compute",
        "polynomial.ideal.quotient.compute",
        "polynomial.ideal.radical.compute",
        "polynomial.ideal.saturation.compute",
        "polynomial.map.generic_degree.compute",
    }
)


def _builtin_operations() -> tuple[MathTool[Any, Any], ...]:
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
def test_advertised_invocation_example_executes_when_backend_is_available(
    operation: MathTool[Any, Any],
) -> None:
    operation_id = operation.operation_id
    if operation_id in _SINGULAR_OPERATION_IDS and shutil.which("Singular") is None:
        pytest.skip("the published example is owned by the Singular runtime lane")
    examples = operation.examples
    assert examples, f"{operation_id} must advertise one executable example"
    for invocation_example in examples:
        schema = operation.request_type.model_json_schema()
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(invocation_example.input)
        operation.request_type.model_validate_json(
            encode_strict_json(invocation_example.input), strict=True
        )
        public_result = invoke_operation(
            operation_id,
            invocation_example.input,
            _CATALOG,
        )
        assert public_result.operation_id == operation_id
        serialized = public_result.output
        assert serialized, f"{operation_id} example produced an empty result"
        validated = operation.result_type.model_validate_json(
            encode_strict_json(serialized)
        )
        assert validated.model_dump(mode="json") == serialized, (
            operation_id,
            serialized,
            validated.model_dump(mode="json"),
        )


def test_fixed_point_prefix_dispatch_preserves_tuple_decoding_and_preflights_source() -> (
    None
):
    """JSON arrays remain valid canonical tuples after fixed-point preflight."""

    operation = _CATALOG.operation("substitution.fixed_point_prefix.compute")
    assert operation is not None
    example = operation.examples[0]

    result = invoke_operation(operation.operation_id, example.input, _CATALOG)
    assert result.output["prefix"]["letters"] == [
        "0",
        "1",
        "0",
        "0",
        "1",
        "0",
        "1",
        "0",
    ]

    oversized_mortal_source = {
        "source": {
            "substitution": {
                "morphism": {
                    "source_alphabet": ["0", "1", "2", "3"],
                    "target_alphabet": ["0", "1", "2", "3"],
                    "images": [["0", *(["1"] * 9_999)], [], ["2"] * 10_000, ["3"]],
                }
            },
            "seed": "0",
        },
        "prefix_length": 1,
    }
    with pytest.raises(
        OperationRequestValidationError, match="payload failed validation"
    ) as error:
        invoke_operation(operation.operation_id, oversized_mortal_source, _CATALOG)
    assert (
        "source exceeds the aggregate occurrence bound"
        in error.value.errors()[0]["msg"]
    )
