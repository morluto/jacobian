"""Dispatch boundary for the finite geometry operations."""

import pytest
from jsonschema.validators import Draft202012Validator

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation, parse_operation_input
from jacobian.math.geometry.finite._models import (
    MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES,
    MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS,
    GrassmannianCountRequest,
)
from jacobian.math.geometry.finite._operations import compute_grassmannian_count


def test_dispatch_round_trips_an_exact_count_past_the_json_integer_range() -> None:
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
    assert (
        f"{MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES}-byte result budget" in description
    )
    assert "bare coordinate tuples" in description

    validator = Draft202012Validator(schema)
    accepted = {"space": {"field_order": 2, "axis": ["x", "y"]}}
    assert not list(validator.iter_errors(accepted))
    output = invoke_operation(operation.operation_id, accepted, Catalog.open()).output
    assert output["sequence"]["coordinates"] == [[0, 1], [1, 0], [1, 1]]

    structurally_valid_but_too_large = {
        "space": {"field_order": 257, "axis": ["x", "y"]}
    }
    assert not list(validator.iter_errors(structurally_valid_but_too_large))
    with pytest.raises(ValueError, match="vector enumeration envelope"):
        invoke_operation(
            operation.operation_id, structurally_valid_but_too_large, Catalog.open()
        )


def test_dispatch_returns_maximal_enumeration_within_transport_limit() -> None:
    """The admitted-envelope-maximal request returns its complete declared
    result: q=2 with 16 axis labels yields all 65,535 projective points of
    PG(15, F_2), and the typed sequence reply -- the parent space once plus
    bare coordinate tuples -- serializes well inside the canonical transport
    limit instead of failing only after enumeration."""
    payload = {"space": {"field_order": 2, "axis": [f"x{i}" for i in range(16)]}}

    result = invoke_operation(
        "finite_geometry.projective_space.enumerate_points", payload, Catalog.open()
    )

    coordinates = result.output["sequence"]["coordinates"]
    assert len(coordinates) == MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS - 1
    assert coordinates[0] == [0] * 15 + [1]
    encoded = len(canonicalize_json(result.output))
    assert encoded <= CanonicalLimits().max_output_bytes
    assert encoded < MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES


def test_dispatch_rejects_untransportable_enumeration_as_invalid_request() -> None:
    """A request whose predicted serialized result exceeds the owner-local
    budget fails admission before execution as a typed invalid request --
    never as a post-enumeration host exception."""
    payload = {
        "space": {"field_order": 2, "axis": ["x", "y" * (9 * 1024 * 1024)]},
    }

    with pytest.raises(ValueError, match="serialized point list"):
        invoke_operation(
            "finite_geometry.projective_space.enumerate_points",
            payload,
            Catalog.open(),
        )
