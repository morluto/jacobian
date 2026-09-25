"""Bounded graph-to-flag-complex construction."""

from __future__ import annotations

import json
from itertools import combinations
from math import prod

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.topology._models import SimplicialComplexRequest
from jacobian.math.topology.release import (
    CliqueRequest,
    CliqueResult,
    GraphCliqueRequest,
    clique_complex,
    graph_clique_complex,
)


def _complex_request(
    vertex_count: int, edges: tuple[tuple[int, int], ...]
) -> CliqueRequest:
    vertices = tuple(f"v{index}" for index in range(vertex_count))
    incident = {vertex for edge in edges for vertex in edge}
    facets: tuple[tuple[str, ...], ...] = tuple(
        (vertices[left], vertices[right]) for left, right in edges
    )
    facets += tuple(
        (vertices[index],) for index in range(vertex_count) if index not in incident
    )
    return CliqueRequest(
        complex=SimplicialComplexRequest(vertices=vertices, facets=facets)
    )


def _assert_admission_code(
    error: OperationResourceAdmissionError, expected: str
) -> None:
    assert error.errors()[0]["type"] == expected


def test_graph_clique_complex_matches_independent_powerset_oracle() -> None:
    for vertex_count in range(1, 5):
        possible_edges = tuple(combinations(range(vertex_count), 2))
        for mask in range(1 << len(possible_edges)):
            edges = tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            )
            graph = IndexedSimpleUndirectedGraph(vertex_count=vertex_count, edges=edges)
            result = graph_clique_complex(GraphCliqueRequest(graph=graph))
            edge_set = set(edges)
            cliques = {
                subset
                for size in range(1, vertex_count + 1)
                for subset in combinations(range(vertex_count), size)
                if all(
                    (left, right) in edge_set for left, right in combinations(subset, 2)
                )
            }
            maximal = tuple(
                sorted(
                    tuple(f"v{vertex}" for vertex in subset)
                    for subset in cliques
                    if not any(set(subset) < set(other) for other in cliques)
                )
            )
            assert set(result.clique_facets) == set(maximal)
            assert result.clique_complex.closure_size == len(cliques)


def test_graph_clique_complex_boundary_and_oversized_request() -> None:
    full = IndexedSimpleUndirectedGraph(
        vertex_count=8,
        edges=tuple((left, right) for left in range(8) for right in range(left + 1, 8)),
    )
    assert (
        graph_clique_complex(GraphCliqueRequest(graph=full)).clique_complex.closure_size
        == 255
    )

    oversized = IndexedSimpleUndirectedGraph(vertex_count=9, edges=())
    with pytest.raises(OperationResourceAdmissionError, match="between 1 and 8"):
        graph_clique_complex(GraphCliqueRequest(graph=oversized))


def test_complex_clique_rejects_k9_before_materializing_its_faces() -> None:
    edges = tuple(combinations(range(9), 2))
    with pytest.raises(OperationResourceAdmissionError) as error:
        clique_complex(_complex_request(9, edges))

    _assert_admission_code(error.value, "topology.clique.dimension_budget")


def test_complex_clique_rejects_k16_before_materializing_its_faces() -> None:
    edges = tuple(combinations(range(16), 2))
    with pytest.raises(OperationResourceAdmissionError) as error:
        clique_complex(_complex_request(16, edges))

    _assert_admission_code(error.value, "topology.clique.dimension_budget")


def test_complex_clique_preflights_pair_check_work() -> None:
    # A 17-vertex graph has 131,071 possible subsets and a worst-case pair
    # bound above two million; preflight refuses before clique enumeration.
    with pytest.raises(OperationResourceAdmissionError) as error:
        clique_complex(_complex_request(17, ()))

    _assert_admission_code(error.value, "topology.clique.pair_check_budget")


def test_complex_clique_stops_at_face_closure_limit() -> None:
    # The graph is complete multipartite with parts of sizes 3,3,3,3,2,2.
    # It has 2,303 nonempty cliques, no clique larger than six vertices, and
    # 324 maximal cliques. The closure limit is reached before all faces are
    # retained; the independent product count gives the exact expected total.
    parts = (3, 3, 3, 3, 2, 2)
    vertex_count = sum(parts)
    part_for_vertex: list[int] = []
    for part, size in enumerate(parts):
        part_for_vertex.extend([part] * size)
    edges = tuple(
        (left, right)
        for left, right in combinations(range(vertex_count), 2)
        if part_for_vertex[left] != part_for_vertex[right]
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        clique_complex(_complex_request(vertex_count, edges))

    assert prod(size + 1 for size in parts) - 1 == 2_303
    _assert_admission_code(error.value, "topology.clique.face_budget")


def test_complex_clique_enforces_canonical_maximal_facet_limit() -> None:
    # Complete multipartite parts (3,3,3,3,2) give 767 faces but 162
    # maximal cliques, isolating the carrier's 128-facet output bound.
    parts = (3, 3, 3, 3, 2)
    vertex_count = sum(parts)
    part_for_vertex: list[int] = []
    for part, size in enumerate(parts):
        part_for_vertex.extend([part] * size)
    edges = tuple(
        (left, right)
        for left, right in combinations(range(vertex_count), 2)
        if part_for_vertex[left] != part_for_vertex[right]
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        clique_complex(_complex_request(vertex_count, edges))

    assert prod(size + 1 for size in parts) - 1 == 767
    assert prod(parts) == 162
    _assert_admission_code(error.value, "topology.clique.facet_budget")


def test_complex_clique_accepts_exact_facet_bound_and_large_closure() -> None:
    # Complete multipartite parts (4,4,4,2,1,1) give exactly 128 maximal
    # cliques and 1,499 nonempty faces, below the independent 2,048-face cap.
    parts = (4, 4, 4, 2, 1, 1)
    vertex_count = sum(parts)
    part_for_vertex: list[int] = []
    for part, size in enumerate(parts):
        part_for_vertex.extend([part] * size)
    edges = tuple(
        (left, right)
        for left, right in combinations(range(vertex_count), 2)
        if part_for_vertex[left] != part_for_vertex[right]
    )

    result = clique_complex(_complex_request(vertex_count, edges))

    assert prod(parts) == 128
    assert prod(size + 1 for size in parts) - 1 == 1_499
    assert result.clique_complex.closure_size == 1_499
    assert len(result.clique_facets) == 128


def test_sparse_sixteen_cycle_is_accepted_at_candidate_envelope() -> None:
    vertex_count = 16
    edges = tuple(
        (index, (index + 1) % vertex_count) for index in range(vertex_count - 1)
    )
    edges += ((0, vertex_count - 1),)

    result = clique_complex(_complex_request(vertex_count, edges))

    assert result.clique_complex.vertices == tuple(
        sorted(f"v{i}" for i in range(vertex_count))
    )
    assert result.clique_complex.dimension == 1
    assert result.clique_complex.closure_size == 2 * vertex_count
    assert len(result.clique_facets) == vertex_count


def test_near_face_limit_clique_result_round_trips_through_catalog() -> None:
    # Nine distinct K8 facets share a K7. Their complete closure has
    # 2^7 - 1 + 9*2^7 = 1,279 nonempty faces.
    core_size = 7
    outer_count = 9
    vertex_count = core_size + outer_count
    edges = tuple(combinations(range(core_size), 2)) + tuple(
        (core, outer)
        for outer in range(core_size, vertex_count)
        for core in range(core_size)
    )
    request = _complex_request(vertex_count, edges)
    catalog = Catalog.open()
    operation_id = "topology.simplicial_complex.clique.compute"
    result = invoke_operation(
        operation_id,
        {"complex": request.complex.model_dump(mode="json")},
        catalog,
    )
    decoded = CliqueResult.model_validate_json(json.dumps(result.output))

    assert decoded.clique_complex.closure_size == 1_279
    repeated = invoke_operation(
        operation_id,
        {"complex": decoded.clique_complex.model_dump(mode="json")},
        catalog,
    )
    repeated_value = CliqueResult.model_validate_json(json.dumps(repeated.output))
    assert repeated_value.clique_complex == decoded.clique_complex
