"""Catalog and dispatch contracts for root-system operations."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.root_systems._models import (
    MAX_RANK,
    MAX_REFLECTION_COORDINATE,
)


def test_simple_reflection_catalog_contract_matches_dispatch() -> None:
    """The advertised finite-type and rank rules reach the typed operation."""

    catalog = Catalog.open()
    descriptor = catalog.inspect("root_system.simple_reflection.compute")
    assert descriptor is not None
    properties = descriptor.input_schema["properties"]
    assert properties["matrix"]["minItems"] == 1
    assert properties["matrix"]["maxItems"] == MAX_RANK
    assert f"rank 1 through {MAX_RANK}" in properties["matrix"]["description"]
    assert (
        "finite-type generalized cartan matrix"
        in properties["matrix"]["description"].lower()
    )
    assert (
        "length must equal the Cartan-matrix rank"
        in properties["vector"]["description"]
    )
    assert str(MAX_REFLECTION_COORDINATE) in properties["vector"]["description"]
    assert properties["vector"]["items"] == {
        "maximum": MAX_REFLECTION_COORDINATE,
        "minimum": -MAX_REFLECTION_COORDINATE,
        "type": "integer",
    }
    assert (
        "smaller than the Cartan-matrix rank"
        in properties["simple_index"]["description"]
    )

    operation = catalog.operation("root_system.simple_reflection.compute")
    assert operation is not None
    example = operation.examples[0]
    assert "finite-type generalized Cartan matrix" in example.description
    assert "two simple-root coordinates" in example.description
    result = invoke_operation(operation.operation_id, example.input, catalog)
    assert result.output["reflected_vector"] == [-1, 0]

    malformed = {**example.input, "vector": [1]}
    with pytest.raises(OperationRequestValidationError) as exc_info:
        invoke_operation(operation.operation_id, malformed, catalog)
    assert exc_info.value.errors()[0]["type"] == "root_system.vector_length_mismatch"
