"""Exact path-group oracle tests for fundamental-group basepoint transport."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.edge_paths._models import (
    FundamentalGroupBasepointChangeRequest,
    PresentationBasepointChangePath,
    PresentationMapCompositionRequest,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    change_fundamental_group_basepoint,
    compose_fundamental_group_maps,
)


def _wedge_of_two_circles():
    return canonical_complex(
        ("a", "b", "c", "d", "e"),
        (("a", "b"), ("b", "c"), ("a", "c"), ("a", "d"), ("d", "e"), ("a", "e")),
    )


def _transport(complex_, path_vertices):
    path = PresentationBasepointChangePath(
        complex=complex_,
        source_base_vertex=path_vertices[0],
        target_base_vertex=path_vertices[-1],
        path_vertices=path_vertices,
    )
    return change_fundamental_group_basepoint(
        FundamentalGroupBasepointChangeRequest(path=path)
    )


def _letters(word):
    return tuple((letter.generator, letter.exponent) for letter in word.letters)


def _reduce(sequence):
    stack = []
    for generator, exponent in sequence:
        if stack and stack[-1] == (generator, -exponent):
            stack.pop()
        else:
            stack.append((generator, exponent))
    return tuple(stack)


def _inverse(sequence):
    return tuple((generator, -exponent) for generator, exponent in reversed(sequence))


def test_loop_path_conjugates_free_generators_and_composes_with_its_inverse():
    complex_ = _wedge_of_two_circles()
    loop = _transport(complex_, ("a", "b", "c", "a"))

    assert len(loop.generator_images) == 2
    first_generator = _letters(loop.generator_images[0])
    second_generator = _letters(loop.generator_images[1])
    assert first_generator == ((0, 1),)
    # Independent free-group path oracle for p^-1 g1 p with p=g0.
    assert second_generator == _reduce(
        (*_inverse(first_generator), (1, 1), *first_generator)
    )
    assert loop.abelianization_map.entries == ((1, 0), (0, 1))

    inverse_loop = _transport(complex_, ("a", "c", "b", "a"))
    composite = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=loop, second=inverse_loop)
    )
    assert tuple(_letters(word) for word in composite.generator_images) == (
        ((0, 1),),
        ((1, 1),),
    )
    assert composite.abelianization_map.entries == ((1, 0), (0, 1))

    between_vertices = _transport(complex_, ("a", "b"))
    reverse_between_vertices = _transport(complex_, ("b", "a"))
    round_trip = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(
            first=between_vertices, second=reverse_between_vertices
        )
    )
    identity_at_a = _transport(complex_, ("a",))
    assert tuple(_letters(word) for word in round_trip.generator_images) == tuple(
        _letters(word) for word in identity_at_a.generator_images
    )
    assert round_trip.abelianization_map.entries == identity_at_a.abelianization_map.entries


def test_path_morphism_serializes_and_binds_exact_basepoints():
    result = _transport(_wedge_of_two_circles(), ("a", "b", "c", "a"))
    from jacobian.math.topology.edge_paths._models import FundamentalGroupMapResult

    restored = FundamentalGroupMapResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert isinstance(restored.map, PresentationBasepointChangePath)
    assert restored.source_presentation.base_vertex == "a"
    assert restored.target_presentation.base_vertex == "a"


def test_transport_replays_triangle_relators_as_target_conjugates():
    complex_ = canonical_complex(
        ("a", "b", "c", "d", "e"),
        (
            ("a", "b"),
            ("b", "c"),
            ("a", "c"),
            ("a", "d", "e"),
        ),
    )
    result = _transport(complex_, ("a", "b", "c", "a"))
    assert tuple(_letters(word) for word in result.generator_images) == (
        ((0, 1),),
        ((0, -1), (1, 1), (0, 1)),
    )
    witness = result.relator_images[0]
    assert witness.target_relator_index == 0
    assert witness.target_orientation == 1
    assert _letters(witness.conjugator) == ((0, -1),)


def test_transport_path_rejects_wrong_endpoints_and_non_edges():
    complex_ = _wedge_of_two_circles()
    with pytest.raises(ValidationError, match="run from the source"):
        PresentationBasepointChangePath(
            complex=complex_,
            source_base_vertex="a",
            target_base_vertex="a",
            path_vertices=("a", "b"),
        )
    with pytest.raises(ValidationError, match="each consecutive path pair"):
        PresentationBasepointChangePath(
            complex=complex_,
            source_base_vertex="a",
            target_base_vertex="e",
            path_vertices=("a", "b", "e"),
        )


def test_basepoint_transport_is_a_typed_composable_math_tool():
    from jacobian.math.topology.edge_paths._models import (
        FundamentalGroupMapResult,
    )
    from jacobian.math.topology.edge_paths.presentation_maps_tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "topology.simplicial.fundamental_group.basepoint_change.compute"
    )
    assert tool.request_type is FundamentalGroupBasepointChangeRequest
    assert tool.result_type is FundamentalGroupMapResult
    result = tool.run(
        FundamentalGroupBasepointChangeRequest(
            path=PresentationBasepointChangePath(
                complex=_wedge_of_two_circles(),
                source_base_vertex="a",
                target_base_vertex="a",
                path_vertices=("a",),
            )
        )
    )
    assert result.generator_images
