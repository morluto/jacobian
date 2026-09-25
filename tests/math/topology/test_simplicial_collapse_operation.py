from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_FACES,
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology.simplicial_collapse import (
    ElementaryCollapsePair,
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
    elementary_collapse,
)


def _request(
    facets: tuple[tuple[str, ...], ...],
    *,
    face: tuple[str, ...],
    coface: tuple[str, ...],
    vertices: tuple[str, ...] | None = None,
) -> ElementaryCollapseRequest:
    if vertices is None:
        vertices = tuple(sorted({label for facet in facets for label in facet}))
    return ElementaryCollapseRequest(
        complex=SimplicialComplexRequest(vertices=vertices, facets=facets),
        pair=ElementaryCollapsePair(face=face, coface=coface),
    )


def _faces(facets: tuple[tuple[str, ...], ...]) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for facet in facets:
        for size in range(1, len(facet) + 1):
            result.update(combinations(facet, size))
    return result


def _maximal(faces: set[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            face for face in faces if not any(set(face) < set(other) for other in faces)
        )
    )


def _assert_face_set_oracle(
    result: ElementaryCollapseResult,
    source_facets: tuple[tuple[str, ...], ...],
    face: tuple[str, ...],
    coface: tuple[str, ...],
) -> None:
    source = result.source
    target = result.target
    assert isinstance(source, FiniteSimplicialComplex)
    assert isinstance(target, FiniteSimplicialComplex)
    expected = _faces(source_facets) - {tuple(sorted(face)), tuple(sorted(coface))}
    actual = _faces(target.maximal_simplices)
    assert actual == expected
    assert target.maximal_simplices == _maximal(expected)


class TestElementaryCollapseOperation:
    def test_edge_collapse_returns_source_bound_canonical_target(self) -> None:
        facets = (("a", "b"),)
        request = _request(facets, face=("a",), coface=("a", "b"))
        result = elementary_collapse(request)
        assert result.source.maximal_simplices == facets
        assert result.target.maximal_simplices == (("b",),)
        assert result.removed_pair == ElementaryCollapsePair(
            face=("a",), coface=("a", "b")
        )
        _assert_face_set_oracle(result, facets, ("a",), ("a", "b"))
        assert ElementaryCollapseResult.model_validate(result.model_dump()) == result
        assert (
            elementary_collapse(
                ElementaryCollapseRequest.model_validate(request.model_dump())
            )
            == result
        )

    def test_simplex_boundary_faces_and_other_component_are_retained(self) -> None:
        facets = (("a", "b", "c"), ("d", "e"))
        result = elementary_collapse(
            _request(facets, face=("a", "b"), coface=("a", "b", "c"))
        )
        assert result.target.maximal_simplices == (
            ("a", "c"),
            ("b", "c"),
            ("d", "e"),
        )
        _assert_face_set_oracle(result, facets, ("a", "b"), ("a", "b", "c"))

    def test_nonfree_pair_is_rejected(self) -> None:
        facets = (("a", "b"), ("a", "c"), ("b", "c"))
        with pytest.raises(OperationDomainValidationError) as error:
            elementary_collapse(_request(facets, face=("a",), coface=("a", "b")))
        assert (
            error.value.errors()[0]["type"] == "topology.elementary_collapse.not_free"
        )

    def test_pair_must_be_a_codimension_one_incidence(self) -> None:
        request = _request(
            (("a", "b", "c"),),
            face=("a",),
            coface=("a", "b", "c"),
        )
        with pytest.raises(OperationDomainValidationError) as error:
            elementary_collapse(request)
        assert error.value.errors()[0]["type"] == (
            "topology.elementary_collapse.not_codimension_one"
        )

    def test_collapses_compose_through_serialized_target(self) -> None:
        facets = (("a", "b", "c"),)
        first = elementary_collapse(
            _request(facets, face=("a", "b"), coface=("a", "b", "c"))
        )
        second = elementary_collapse(
            ElementaryCollapseRequest(
                complex=SimplicialComplexRequest.model_validate(
                    first.target.model_dump()
                ),
                pair=ElementaryCollapsePair(face=("a",), coface=("a", "c")),
            )
        )
        assert second.target.maximal_simplices == (("b", "c"),)
        _assert_face_set_oracle(first, facets, ("a", "b"), ("a", "b", "c"))
        _assert_face_set_oracle(
            second,
            first.target.maximal_simplices,
            ("a",),
            ("a", "c"),
        )

    def test_boundary_and_degenerate_one_dimensional_case(self) -> None:
        facets = (("a", "b"), ("c",))
        result = elementary_collapse(_request(facets, face=("a",), coface=("a", "b")))
        assert result.target.maximal_simplices == (("b",), ("c",))
        _assert_face_set_oracle(result, facets, ("a",), ("a", "b"))

        with pytest.raises(ValidationError):
            ElementaryCollapsePair(face=("p",), coface=("p",))

    def test_closure_at_face_bound_is_accepted_and_above_is_rejected(self) -> None:
        blocks = tuple(
            tuple(f"v{block}_{index}" for index in range(8)) for block in range(8)
        )
        cross_edges = tuple((blocks[0][index], blocks[1][index]) for index in range(8))
        facets_at_bound = tuple(sorted((*blocks, *cross_edges)))
        assert len(_faces(facets_at_bound)) == MAX_TOPOLOGY_FACES
        result = elementary_collapse(
            _request(
                facets_at_bound,
                face=blocks[0][:-1],
                coface=blocks[0],
            )
        )
        assert result.source.closure_size == MAX_TOPOLOGY_FACES
        assert result.target.closure_size == MAX_TOPOLOGY_FACES - 2

        facets_over_bound = tuple(sorted((*blocks, *cross_edges, ("v0_0", "v2_0"))))
        assert len(_faces(facets_over_bound)) == MAX_TOPOLOGY_FACES + 1
        with pytest.raises(OperationResourceAdmissionError) as error:
            elementary_collapse(
                _request(
                    facets_over_bound,
                    face=blocks[0][:-1],
                    coface=blocks[0],
                )
            )
        assert error.value.errors()[0]["type"] == (
            "topology.elementary_collapse.face_bound"
        )

    def test_unrepresentable_target_facet_count_is_preflighted(self) -> None:
        large_facet = tuple(f"s{index}" for index in range(8))
        extra_vertices = tuple(f"x{index:02d}" for index in range(56))
        many_edges = tuple(combinations(extra_vertices, 2))[:127]
        facets = tuple(sorted((large_facet, *many_edges)))
        assert len(facets) == 128
        request = _request(
            facets,
            face=large_facet[:-1],
            coface=large_facet,
        )
        with pytest.raises(OperationResourceAdmissionError) as error:
            elementary_collapse(request)
        assert error.value.errors()[0]["type"] == (
            "topology.elementary_collapse.target_facet_bound"
        )

    def test_model_construct_outer_facet_bound_is_checked_before_validation(
        self,
    ) -> None:
        pair = ElementaryCollapsePair(face=("a",), coface=("a", "b"))
        raw_complex = SimplicialComplexRequest.model_construct(
            vertices=("a", "b"),
            facets=tuple(("a", "b") for _ in range(129)),
        )
        forged_request = ElementaryCollapseRequest.model_construct(
            complex=raw_complex,
            pair=pair,
        )
        with pytest.raises(OperationResourceAdmissionError) as error:
            elementary_collapse(forged_request)
        assert error.value.errors()[0]["type"] == (
            "topology.elementary_collapse.facet_bound"
        )

    def test_nested_owner_manifest_publishes_operation(self) -> None:
        matches = [
            tool
            for tool in BUILTIN_TOOLS
            if tool.operation_id
            == "topology.simplicial_complex.elementary_collapse.compute"
        ]
        assert len(matches) == 1
        request = ElementaryCollapseRequest.model_validate(matches[0].examples[0].input)
        assert matches[0].run(request).target.maximal_simplices == (("b",),)
