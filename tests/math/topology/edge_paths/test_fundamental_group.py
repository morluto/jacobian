"""Tests for the edge-path fundamental-group presentation (#1813)."""

from __future__ import annotations

import itertools
import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.topology._models import (
    HomologyConvention,
    SimplicialComplexRequest,
)
from jacobian.math.topology._simplicial_kernel import integral_homology
from jacobian.math.topology.edge_paths._models import (
    FundamentalGroupPresentationRequest,
    FundamentalGroupPresentationResult,
)
from jacobian.math.topology.edge_paths._tools import TOOLS, _fundamental_group
from jacobian.math.topology.edge_paths.operations import (
    fundamental_group_presentation,
)

_OPERATION_ID = "topology.simplicial.fundamental_group.presentation.compute"

_CIRCLE = (("0", "1"), ("1", "2"), ("0", "2"))

_RP2 = (
    ("0", "1", "2"),
    ("0", "1", "3"),
    ("0", "2", "4"),
    ("0", "3", "5"),
    ("0", "4", "5"),
    ("1", "2", "5"),
    ("1", "3", "4"),
    ("2", "3", "4"),
    ("2", "3", "5"),
    ("1", "4", "5"),
)


def _torus() -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    vertices = tuple(f"v{i}{j}" for i in range(3) for j in range(3))

    def vertex(i: int, j: int) -> str:
        return f"v{i % 3}{j % 3}"

    facets: set[tuple[str, str, str]] = set()
    for i in range(3):
        for j in range(3):
            facets.add(
                tuple(sorted((vertex(i, j), vertex(i + 1, j), vertex(i, j + 1))))
            )
            facets.add(
                tuple(
                    sorted(
                        (
                            vertex(i + 1, j),
                            vertex(i + 1, j + 1),
                            vertex(i, j + 1),
                        )
                    )
                )
            )
    return vertices, tuple(sorted(facets))


def _request(
    vertices: tuple[str, ...],
    facets: tuple[tuple[str, ...], ...],
    base_vertex: str,
) -> FundamentalGroupPresentationRequest:
    return FundamentalGroupPresentationRequest(
        complex=SimplicialComplexRequest(vertices=vertices, facets=facets),
        base_vertex=base_vertex,
    )


def _abelianization_of_integral_h1(
    vertices: tuple[str, ...],
    facets: tuple[tuple[str, ...], ...],
) -> tuple[int, tuple[int, ...]]:
    request = SimplicialComplexRequest(vertices=vertices, facets=facets)
    from jacobian.math.topology._models import canonical_complex

    complex_value = canonical_complex(request.vertices, request.facets)
    result = integral_homology(complex_value, HomologyConvention.UNREDUCED)
    group = next(
        group for group in result.homology.homology_groups if group.degree == 1
    )
    return group.free_rank, tuple(
        int(factor) for factor in group.torsion_invariant_factors
    )


class TestKnownAnswer:
    def test_circle_is_one_free_generator(self) -> None:
        result = fundamental_group_presentation(_request(("0", "1", "2"), _CIRCLE, "0"))
        assert result.presentation.generators == ("g0",)
        assert result.presentation.relators == ()
        assert result.non_tree_edges == (("1", "2"),)
        assert result.abelianization.free_rank == 1
        assert result.abelianization.rank == 0
        assert result.abelianization.torsion_invariant_factors == ()

    def test_point_is_the_trivial_presentation(self) -> None:
        result = fundamental_group_presentation(_request(("p",), (("p",),), "p"))
        assert result.presentation.generators == ()
        assert result.presentation.relators == ()
        assert result.component_vertices == ("p",)
        assert result.edge_words == ()
        assert result.abelianization.free_rank == 0

    def test_wedge_of_two_circles_is_free_of_rank_two(self) -> None:
        facets = (
            ("0", "1"),
            ("0", "2"),
            ("1", "2"),
            ("0", "3"),
            ("0", "4"),
            ("3", "4"),
        )
        result = fundamental_group_presentation(
            _request(("0", "1", "2", "3", "4"), facets, "0")
        )
        assert len(result.non_tree_edges) == 2
        assert result.presentation.relators == ()
        assert result.abelianization.free_rank == 2

    def test_rp2_abelianization_is_order_two(self) -> None:
        result = fundamental_group_presentation(
            _request(tuple(str(i) for i in range(6)), _RP2, "0")
        )
        assert result.abelianization.free_rank == 0
        assert result.abelianization.torsion_invariant_factors == (2,)

    def test_torus_abelianization_is_free_rank_two(self) -> None:
        vertices, facets = _torus()
        result = fundamental_group_presentation(_request(vertices, facets, vertices[0]))
        assert result.abelianization.free_rank == 2
        assert result.abelianization.torsion_invariant_factors == ()


class TestDefiningInvariant:
    def test_abelianization_matches_integral_h1_for_fixtures(self) -> None:
        fixtures = (
            (("0", "1", "2"), _CIRCLE, "0"),
            (tuple(str(i) for i in range(6)), _RP2, "0"),
            (*_torus(), "v00"),
        )
        for vertices, facets, base in fixtures:
            result = fundamental_group_presentation(_request(vertices, facets, base))
            free_rank, torsion = _abelianization_of_integral_h1(vertices, facets)
            assert result.abelianization.free_rank == free_rank
            assert result.abelianization.torsion_invariant_factors == torsion

    def test_tree_spans_component_and_is_acyclic(self) -> None:
        vertices, facets = _torus()
        result = fundamental_group_presentation(_request(vertices, facets, vertices[0]))
        tree = set(result.spanning_tree_edges)
        component = set(result.component_vertices)
        assert len(tree) == len(component) - 1
        adjacency: dict[str, set[str]] = {vertex: set() for vertex in component}
        for left, right in tree:
            adjacency[left].add(right)
            adjacency[right].add(left)
        reached = {result.base_vertex}
        frontier = [result.base_vertex]
        while frontier:
            vertex = frontier.pop()
            for neighbour in adjacency[vertex]:
                if neighbour not in reached:
                    reached.add(neighbour)
                    frontier.append(neighbour)
        assert reached == component

    def test_edge_reversal_is_the_inverse_word(self) -> None:
        result = fundamental_group_presentation(_request(("0", "1", "2"), _CIRCLE, "0"))
        for entry in result.edge_words:
            forward = tuple(
                (letter.generator, letter.exponent) for letter in entry.forward.letters
            )
            backward = tuple(
                (letter.generator, letter.exponent) for letter in entry.backward.letters
            )
            assert backward == tuple(
                (generator, -exponent) for generator, exponent in reversed(forward)
            )

    def test_every_non_tree_edge_gets_exactly_one_generator(self) -> None:
        result = fundamental_group_presentation(
            _request(tuple(str(i) for i in range(6)), _RP2, "0")
        )
        assert len(result.non_tree_edges) == len(result.presentation.generators)
        assert len(result.triangle_relators) == len(result.presentation.relators)
        assert all(
            entry.edge in set(result.non_tree_edges)
            or entry.edge in set(result.spanning_tree_edges)
            for entry in result.edge_words
        )

    def test_relators_are_freely_reduced(self) -> None:
        result = fundamental_group_presentation(
            _request(tuple(str(i) for i in range(6)), _RP2, "0")
        )
        for relator in result.presentation.relators:
            letters = relator.letters
            for left, right in itertools.pairwise(letters):
                assert not (
                    left.generator == right.generator
                    and left.exponent == -right.exponent
                )

    def test_result_round_trips_through_json(self) -> None:
        result = fundamental_group_presentation(
            _request(tuple(str(i) for i in range(6)), _RP2, "0")
        )
        decoded = FundamentalGroupPresentationResult.model_validate_json(
            result.model_dump_json()
        )
        assert decoded == result


class TestBoundaryAndAdversarial:
    def test_disconnected_complex_selects_the_basepoint_component(self) -> None:
        facets = (("0", "1"), ("2", "3"))
        result = fundamental_group_presentation(
            _request(("0", "1", "2", "3"), facets, "0")
        )
        assert result.component_vertices == ("0", "1")
        assert result.non_tree_edges == ()
        assert result.presentation.generators == ()

    def test_base_vertex_must_be_a_vertex(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            fundamental_group_presentation(_request(("0", "1"), (("0", "1"),), "9"))

    def test_catalog_contains_the_published_operation(self) -> None:
        assert _OPERATION_ID in {tool.operation_id for tool in TOOLS}


class TestNativeCatalogParity:
    def test_entry_matches_kernel(self) -> None:
        request = _request(("0", "1", "2"), _CIRCLE, "0")
        assert _fundamental_group(request) == fundamental_group_presentation(request)

    def test_catalog_invocation_matches_native_result(self) -> None:
        request = _request(tuple(str(i) for i in range(6)), _RP2, "0")
        native = fundamental_group_presentation(request)
        catalog = Catalog.open()
        invocation = invoke_operation(
            _OPERATION_ID,
            {
                "complex": {
                    "vertices": list(request.complex.vertices),
                    "facets": [list(facet) for facet in request.complex.facets],
                },
                "base_vertex": "0",
            },
            catalog,
        )
        decoded = FundamentalGroupPresentationResult.model_validate_json(
            json.dumps(invocation.output)
        )
        assert decoded == native
