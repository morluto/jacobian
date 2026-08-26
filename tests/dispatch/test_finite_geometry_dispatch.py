"""Dispatch boundary for the finite geometry operations."""

import pytest
from jsonschema.validators import Draft202012Validator
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation, parse_operation_input
from jacobian.math.finite_geometry._models import (
    MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS,
    GrassmannianCountRequest,
)
from jacobian.math.finite_geometry._operations import compute_grassmannian_count


def test_grassmannian_count_past_json_integer_range_round_trips_through_dispatch() -> (
    None
):
    """A lawful exact count past 2^53 survives canonical request transport."""

    last_safe_slice = compute_grassmannian_count(
        GrassmannianCountRequest(
            field_order=2,
            ambient_dimension=14,
            subspace_dimension=7,
        )
    )
    assert int(last_safe_slice.count) <= (1 << 53) - 1

    payload = {
        "field_order": 2,
        "ambient_dimension": 15,
        "subspace_dimension": 7,
    }
    request = parse_operation_input(GrassmannianCountRequest, payload)
    assert request == GrassmannianCountRequest.model_validate_json(
        request.model_dump_json()
    )

    result = invoke_operation(
        "finite_geometry.grassmannian.count", payload, Catalog.open()
    )
    assert result.output["count"] == "246614610741341843"
    assert int(result.output["count"]) > (1 << 53) - 1
    assert type(result).model_validate_json(result.model_dump_json()) == result

    descriptor = Catalog.open().inspect("finite_geometry.grassmannian.count")
    assert descriptor is not None
    assert descriptor.output_schema["properties"]["count"]["type"] == "string"


def test_projective_space_schema_publishes_coupled_enumeration_bound() -> None:
    operation = Catalog.open().operation(
        "finite_geometry.projective_space.enumerate_points"
    )
    assert operation is not None

    schema = operation.request_type.model_json_schema()
    description = schema["description"]
    space_description = schema["properties"]["space"]["description"]
    assert f"q**n <= {MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS}" in description
    assert (
        f"q**len(axis) <= {MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS}"
        in space_description
    )

    validator = Draft202012Validator(schema)
    accepted = {"space": {"field_order": 2, "axis": ["x", "y"]}}
    assert not list(validator.iter_errors(accepted))
    assert (
        invoke_operation(operation.operation_id, accepted, Catalog.open()).output[
            "count"
        ]
        == 3
    )

    structurally_valid_but_too_large = {
        "space": {"field_order": 257, "axis": ["x", "y"]}
    }
    assert not list(validator.iter_errors(structurally_valid_but_too_large))
    with pytest.raises(ValidationError) as exc_info:
        operation.request_type.model_validate(structurally_valid_but_too_large)
    assert (
        exc_info.value.errors()[0]["type"]
        == "finite_geometry.projective_space_too_large"
    )
