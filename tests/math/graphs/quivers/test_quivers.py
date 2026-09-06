"""Tests for quiver operations."""

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.quivers._models import (
    AdjacencyMatricesRequest,
    AdjacencyMatricesResult,
    FiniteQuiver,
    FixedLengthPathsRequest,
    VertexProfilesRequest,
    VertexProfilesResult,
)
from jacobian.math.graphs.quivers._tools import TOOLS
from jacobian.math.graphs.quivers.operations import (
    adjacency_matrices,
    fixed_length_paths,
    vertex_profiles,
)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "quiver.adjacency_matrices.compute",
        "quiver.paths.fixed_length.compute",
        "quiver.vertex_profiles.compute",
    }


def test_adjacency_matrices_kronecker() -> None:
    request = AdjacencyMatricesRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1), (0, 1)))
    )
    result = adjacency_matrices(request.quiver)
    assert result.adjacency_matrix == ((0, 2), (0, 0))
    assert result.transpose_matrix == ((0, 0), (2, 0))
    assert (
        AdjacencyMatricesResult.model_validate_json(result.model_dump_json()) == result
    )


def test_adjacency_result_rejects_matrix_shape_forgery() -> None:
    result = adjacency_matrices(FiniteQuiver(vertex_count=2, arrows=((0, 1),)))
    payload = result.model_dump(mode="json")
    payload["adjacency_matrix"]["row_count"] = 1
    with pytest.raises(ValueError, match="shape"):
        AdjacencyMatricesResult.model_validate(payload)


def test_vertex_profiles_kronecker() -> None:
    request = VertexProfilesRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1), (0, 1)))
    )
    result = vertex_profiles(request.quiver)
    assert result.in_degrees == (0, 2)
    assert result.out_degrees == (2, 0)
    assert VertexProfilesResult.model_validate_json(result.model_dump_json()) == result


def test_fixed_length_paths_triangle() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=3, arrows=((0, 1), (1, 2), (2, 0))),
        length=2,
    )
    result = fixed_length_paths(request.quiver, request.length)
    assert result.total_paths == "3"


def test_fixed_length_paths_zero() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1),)),
        length=0,
    )
    result = fixed_length_paths(request.quiver, request.length)
    assert result.total_paths == "2"


def test_fixed_length_paths_returns_counts_beyond_json_number_range() -> None:
    """Exact counts use canonical integers rather than JSON number limits."""
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 8), length=18
    )

    result = fixed_length_paths(request.quiver, request.length)

    assert result.path_matrix == ((str(8**18),),)
    assert result.total_paths == str(8**18)
    assert encode_strict_json(result.model_dump(mode="json"))


def test_fixed_length_paths_handles_large_exact_count_with_small_work() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 32), length=32
    )
    result = fixed_length_paths(request.quiver, request.length)
    assert result.total_paths == str(32**32)


def test_fixed_length_paths_rejects_excessive_matrix_work() -> None:
    request = FixedLengthPathsRequest(quiver=FiniteQuiver(vertex_count=128), length=32)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        fixed_length_paths(request.quiver, request.length)

    assert exc_info.value.errors()[0]["type"] == (
        "quiver.fixed_length_paths_exceeds_envelope"
    )
