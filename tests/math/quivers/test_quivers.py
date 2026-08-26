"""Tests for quiver operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.quivers._models import (
    AdjacencyMatricesRequest,
    FiniteQuiver,
    FixedLengthPathsRequest,
    VertexProfilesRequest,
)
from jacobian.math.quivers._operations import (
    compute_adjacency_matrices,
    compute_fixed_length_paths,
    compute_vertex_profiles,
)
from jacobian.math.quivers._tools import TOOLS


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
    result = compute_adjacency_matrices(request)
    assert result.adjacency_matrix == ((0, 2), (0, 0))
    assert result.transpose_matrix == ((0, 0), (2, 0))


def test_vertex_profiles_kronecker() -> None:
    request = VertexProfilesRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1), (0, 1)))
    )
    result = compute_vertex_profiles(request)
    assert result.in_degrees == (0, 2)
    assert result.out_degrees == (2, 0)


def test_fixed_length_paths_triangle() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=3, arrows=((0, 1), (1, 2), (2, 0))),
        length=2,
    )
    result = compute_fixed_length_paths(request)
    assert result.total_paths == 3


def test_fixed_length_paths_zero() -> None:
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=2, arrows=((0, 1),)),
        length=0,
    )
    result = compute_fixed_length_paths(request)
    assert result.total_paths == 2


def test_fixed_length_paths_admits_transportable_count_boundary() -> None:
    """Eight parallel loops have exactly 8**17 length-17 paths."""
    request = FixedLengthPathsRequest(
        quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 8), length=17
    )

    result = compute_fixed_length_paths(request)

    assert result.path_matrix == ((8**17,),)
    assert result.total_paths == 8**17
    # This is the actual final delivery primitive, not merely model parsing.
    assert encode_strict_json(result.model_dump(mode="json"))


def test_fixed_length_paths_rejects_untransportable_count_before_kernel() -> None:
    """The next power would produce raw JSON integers above 2**53 - 1."""
    with pytest.raises(ValidationError) as exc_info:
        FixedLengthPathsRequest(
            quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 8), length=18
        )

    assert exc_info.value.errors()[0]["type"] == (
        "quiver.fixed_length_paths_exceeds_envelope"
    )


def test_fixed_length_paths_rejects_parallel_loop_explosion_before_kernel() -> None:
    """The reported 32-loop, length-32 request is rejected at admission."""
    with pytest.raises(ValidationError):
        FixedLengthPathsRequest(
            quiver=FiniteQuiver(vertex_count=1, arrows=((0, 0),) * 32), length=32
        )
