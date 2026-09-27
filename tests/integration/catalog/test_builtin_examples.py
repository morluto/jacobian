"""Executable contracts for examples advertised by the builtin catalog."""

from __future__ import annotations

import shutil
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.polynomials.multivariate._factor_backend import (
    factor_worker_containment_available,
)

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
_FACTOR_WORKER_OPERATION_IDS = frozenset({"polynomial.multivariate.factor.compute"})


def _builtin_operations() -> tuple[MathTool[Any, Any], ...]:
    # Do not build every input and output JSON schema while pytest is merely
    # collecting this parametrized test. The selected operation's request
    # schema is validated inside the test itself; Catalog.open() still supplies
    # the full dispatch catalog for the public invocation check.
    return BUILTIN_TOOLS


def _builtin_operation_parameters() -> tuple[Any, ...]:
    return tuple(
        pytest.param(
            operation,
            id=operation.operation_id,
            marks=(
                pytest.mark.singular_catalog_example
                if operation.operation_id in _SINGULAR_OPERATION_IDS
                else ()
            ),
        )
        for operation in _builtin_operations()
    )


@pytest.mark.parametrize("operation", _builtin_operation_parameters())
def test_advertised_invocation_example_executes_when_backend_is_available(
    operation: MathTool[Any, Any],
) -> None:
    operation_id = operation.operation_id
    assert _CATALOG.operation(operation_id) is operation
    examples = operation.examples
    assert examples, f"{operation_id} must advertise one executable example"
    request_schema = operation.request_type.model_json_schema()
    Draft202012Validator.check_schema(request_schema)
    request_validator = Draft202012Validator(request_schema)
    for invocation_example in examples:
        request_validator.validate(invocation_example.input)
        operation.request_type.model_validate_json(
            encode_strict_json(invocation_example.input), strict=True
        )

    if operation_id in _SINGULAR_OPERATION_IDS and shutil.which("Singular") is None:
        pytest.skip("the published example is owned by the Singular runtime lane")
    if (
        operation_id in _FACTOR_WORKER_OPERATION_IDS
        and not factor_worker_containment_available()
    ):
        pytest.skip(
            "the published example needs the hard-memory-containment worker lane"
        )
    for invocation_example in examples:
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
