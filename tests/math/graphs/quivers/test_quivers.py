"""Tests for quiver operations."""

import json

import pytest
from pydantic import ValidationError

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
from jacobian.math.matrices.values import IntegerMatrix


def test_serialized_quiver_matrix_feeds_smith_unchanged() -> None:
    from jacobian.math.matrices.operations import smith_normal_form_result

    result = adjacency_matrices(FiniteQuiver(vertex_count=2, arrows=((0, 1), (0, 1))))
    matrix = IntegerMatrix.model_validate_json(
        __import__("json").dumps(result.model_dump(mode="json")["adjacency_matrix"])
    )
    assert smith_normal_form_result(matrix).invariant_factors == (2,)


@pytest.mark.parametrize("size", [1, 128])
def test_edgeless_quiver_preserves_all_isolated_vertex_axes(size: int) -> None:
    quiver = FiniteQuiver(vertex_count=size)
    adjacency = adjacency_matrices(quiver)
    paths = fixed_length_paths(quiver, 0)
    for result in (adjacency, paths, vertex_profiles(quiver)):
        assert type(result).model_validate_json(result.model_dump_json()) == result
    assert adjacency.adjacency_matrix.row_count == size
    assert adjacency.adjacency_matrix.column_count == size
    assert all(
        value == 0 for row in adjacency.adjacency_matrix.entries for value in row
    )
    assert paths.path_matrix.entries == tuple(
        tuple(int(i == j) for j in range(size)) for i in range(size)
    )
    assert paths.total_paths == size


def test_quiver_result_parsing_checks_axes_without_proving_adjacency() -> None:
    result = adjacency_matrices(FiniteQuiver(vertex_count=2, arrows=((0, 1),)))
    payload = result.model_dump(mode="json")
    # A false mathematical claim still has a valid structural representation.
    payload["adjacency_matrix"] = IntegerMatrix(entries=((7, 0), (0, 7))).model_dump(
        mode="json"
    )
    assert (
        AdjacencyMatricesResult.model_validate_json(
            json.dumps(payload)
        ).adjacency_matrix.entries[0][0]
        == 7
    )
    payload["adjacency_matrix"] = IntegerMatrix(entries=((7,),)).model_dump(mode="json")
    with pytest.raises(ValidationError, match="both quiver vertex axes"):
        AdjacencyMatricesResult.model_validate_json(json.dumps(payload))


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
    assert result.quiver == request.quiver
    assert result.adjacency_matrix.entries == ((0, 2), (0, 0))
    assert result.transpose_matrix.entries == ((0, 0), (2, 0))
    assert (
        AdjacencyMatricesResult.model_validate_json(result.model_dump_json()) == result
    )


def test_adjacency_result_rejects_matrix_shape_forgery() -> None:
    result = adjacency_matrices(FiniteQuiver(vertex_count=2, arrows=((0, 1),)))
    payload = result.model_dump(mode="json")
    payload["adjacency_matrix"]["row_count"] = 1
    with pytest.raises(ValueError, match="shape"):
        AdjacencyMatricesResult.model_validate_json(json.dumps(payload))


def test_vertex_profiles_kronecker() -> None:
    request = VertexProfilesRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1), (0, 1)))
    )
    result = vertex_profiles(request.quiver)
    assert result.quiver == request.quiver
    assert result.in_degrees == (0, 2)
    assert result.out_degrees == (2, 0)
    assert VertexProfilesResult.model_validate_json(result.model_dump_json()) == result


def test_fixed_length_paths_triangle() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=3, arrows=((0, 1), (1, 2), (2, 0))),
        length=2,
    )
    result = fixed_length_paths(request.quiver, request.length)
    assert result.quiver == request.quiver
    assert result.length == request.length
    assert result.total_paths == 3


def test_fixed_length_paths_zero() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1),)),
        length=0,
    )
    result = fixed_length_paths(request.quiver, request.length)
    assert result.total_paths == 2


def test_fixed_length_paths_returns_counts_beyond_json_number_range() -> None:
    """Exact counts use canonical integers rather than JSON number limits."""
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 8), length=18
    )

    result = fixed_length_paths(request.quiver, request.length)

    assert result.path_matrix.entries == ((8**18,),)
    assert result.total_paths == 8**18
    assert encode_strict_json(result.model_dump(mode="json"))


def test_fixed_length_paths_handles_large_exact_count_with_small_work() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 32), length=32
    )
    result = fixed_length_paths(request.quiver, request.length)
    assert result.total_paths == 32**32


def test_fixed_length_paths_rejects_excessive_matrix_work() -> None:
    request = FixedLengthPathsRequest(quiver=FiniteQuiver(vertex_count=128), length=32)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        fixed_length_paths(request.quiver, request.length)

    assert exc_info.value.errors()[0]["type"] == (
        "quiver.fixed_length_paths_exceeds_envelope"
    )


@pytest.mark.parametrize("length", [-1, 33, True, 1.5])
def test_native_path_length_is_admitted_before_computation(length: int) -> None:
    from jacobian.math.graphs.quivers._path_bounds import fixed_length_paths_envelope

    with pytest.raises(ValidationError):
        FixedLengthPathsRequest.model_validate(
            {"quiver": {"vertex_count": 1}, "length": length}
        )
    with pytest.raises(ValueError, match="integer from 0 through 32"):
        fixed_length_paths_envelope(vertex_count=1, arrow_count=2, length=length)
    with pytest.raises(
        OperationDomainValidationError, match="integer from 0 through 32"
    ):
        fixed_length_paths(FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 2), length)
