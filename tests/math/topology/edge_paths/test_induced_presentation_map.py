"""Exact simplicial maps induce presentation-generator words and relators."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    FundamentalGroupMapRequest,
    PresentationMapCompositionRequest,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    compose_fundamental_group_maps,
    induced_fundamental_group_map,
)


def _complex(vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]):
    return canonical_complex(vertices, facets)


def _circle():
    return _complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))


def _reduce(letters: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for letter in letters:
        if result and result[-1] == (letter[0], -letter[1]):
            result.pop()
        else:
            result.append(letter)
    return tuple(result)


def _letters(word) -> tuple[tuple[int, int], ...]:
    return tuple((letter.generator, letter.exponent) for letter in word.letters)


def _replay_triangle_relations(result) -> None:
    for source_relator, witness in zip(
        result.source_presentation.presentation.relators,
        result.relator_images,
        strict=True,
    ):
        image: list[tuple[int, int]] = []
        for letter in source_relator.letters:
            letters = _letters(result.generator_images[letter.generator])
            if letter.exponent == -1:
                letters = tuple(
                    (generator, -sign) for generator, sign in reversed(letters)
                )
            image.extend(letters)
        reduced_image = _reduce(image)
        conjugator = _letters(witness.conjugator)
        inverse_conjugator = tuple(
            (generator, -sign) for generator, sign in reversed(conjugator)
        )
        if witness.target_relator_index is None:
            expected = ()
        else:
            target_relator = result.target_presentation.presentation.relators[
                witness.target_relator_index
            ]
            relation = _letters(target_relator)
            if witness.target_orientation == -1:
                relation = tuple(
                    (generator, -sign) for generator, sign in reversed(relation)
                )
            expected = _reduce([*conjugator, *relation, *inverse_conjugator])
        assert reduced_image == expected


def test_circle_identity_and_reflection_induce_expected_generator_words() -> None:
    circle = _circle()
    identity = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=circle, target=circle, vertex_map=("a", "b", "c")),
            source_base_vertex="a",
            target_base_vertex="a",
        )
    )
    assert len(identity.generator_images) == 1
    assert _letters(identity.generator_images[0]) == ((0, 1),)
    assert identity.abelianization_map.entries == ((1,),)
    _replay_triangle_relations(identity)

    reflection = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=circle, target=circle, vertex_map=("a", "c", "b")),
            source_base_vertex="a",
            target_base_vertex="a",
        )
    )
    assert _letters(reflection.generator_images[0]) == ((0, -1),)
    assert reflection.abelianization_map.entries == ((-1,),)
    _replay_triangle_relations(reflection)


def test_vertex_relabeling_transports_presentation_generators_coherently() -> None:
    # Two three-edge cycles sharing an edge form a rank-two graph. Relabeling
    # changes the deterministic spanning-tree complement, so the induced map
    # must transport generator axes instead of treating their indices as fixed.
    vertices = ("a", "b", "c", "d")
    edges = (("a", "b"), ("b", "c"), ("a", "c"), ("b", "d"), ("a", "d"))
    source = _complex(vertices, edges)
    relabel = {"a": "z", "b": "a", "c": "x", "d": "b"}
    target = _complex(
        tuple(relabel[vertex] for vertex in vertices),
        tuple(tuple(relabel[vertex] for vertex in edge) for edge in edges),
    )

    forward = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=source,
                target=target,
                vertex_map=tuple(relabel[vertex] for vertex in source.vertices),
            ),
            source_base_vertex="a",
            target_base_vertex="z",
        )
    )
    inverse_relabel = {image: vertex for vertex, image in relabel.items()}
    backward = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=target,
                target=source,
                vertex_map=tuple(inverse_relabel[vertex] for vertex in target.vertices),
            ),
            source_base_vertex="z",
            target_base_vertex="a",
        )
    )

    # The relabeled presentation has a different deterministic generator
    # basis. The vertex bijection swaps those axes, and its inverse must
    # transport them back to the identity on the original presentation.
    assert forward.source_presentation != forward.target_presentation
    assert tuple(_letters(word) for word in forward.generator_images) == (
        ((1, 1),),
        ((0, 1),),
    )
    assert tuple(_letters(word) for word in backward.generator_images) == (
        ((1, 1),),
        ((0, 1),),
    )

    round_trip = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=forward, second=backward)
    )
    assert round_trip.source_presentation == forward.source_presentation
    assert round_trip.target_presentation == forward.source_presentation
    assert tuple(_letters(word) for word in round_trip.generator_images) == (
        ((0, 1),),
        ((1, 1),),
    )
    assert round_trip.abelianization_map.entries == ((1, 0), (0, 1))


def test_collapsing_circle_to_edge_maps_generator_to_identity() -> None:
    source = _circle()
    target = _complex(("x", "y"), (("x", "y"),))
    result = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=source, target=target, vertex_map=("x", "y", "x")),
            source_base_vertex="a",
            target_base_vertex="x",
        )
    )
    assert len(result.generator_images) == 1
    assert result.generator_images[0].letters == ()
    assert result.abelianization_map.entries == ()
    assert result.abelianization_map.row_count == 0
    assert result.abelianization_map.column_count == 1
    _replay_triangle_relations(result)


def test_based_map_requires_base_vertex_images_to_agree() -> None:
    circle = _circle()
    request = FundamentalGroupMapRequest(
        map=SimplicialMap(source=circle, target=circle, vertex_map=("a", "b", "c")),
        source_base_vertex="b",
        target_base_vertex="a",
    )
    with pytest.raises(OperationDomainValidationError, match="must send"):
        induced_fundamental_group_map(request)


def test_filled_triangle_identity_replays_exact_relator_witness() -> None:
    triangle = _complex(("a", "b", "c"), (("a", "b", "c"),))
    result = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=triangle, target=triangle, vertex_map=("a", "b", "c")
            ),
            source_base_vertex="a",
            target_base_vertex="a",
        )
    )
    assert len(result.relator_images) == 1
    witness = result.relator_images[0]
    assert witness.source_relator_index == 0
    assert witness.target_relator_index == 0
    assert witness.target_orientation == 1
    assert witness.conjugator.letters == ()
    _replay_triangle_relations(result)


def test_reversed_and_nontrivially_based_triangle_images_replay() -> None:
    triangle = _complex(("a", "b", "c"), (("a", "b", "c"),))
    reversed_map = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=triangle, target=triangle, vertex_map=("a", "c", "b")
            ),
            source_base_vertex="a",
            target_base_vertex="a",
        )
    )
    assert reversed_map.relator_images[0].target_orientation == -1
    _replay_triangle_relations(reversed_map)

    source = _complex(
        ("a", "b", "c", "r", "x"),
        (("r", "x"), ("a", "x"), ("a", "b", "c")),
    )
    target = _complex(
        ("0", "a", "b", "c", "d"),
        (("0", "a"), ("0", "b"), ("0", "c"), ("a", "b"), ("b", "c", "d")),
    )
    based_map = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=source,
                target=target,
                vertex_map=("b", "c", "d", "0", "a"),
            ),
            source_base_vertex="r",
            target_base_vertex="0",
        )
    )
    assert based_map.relator_images[0].conjugator.letters
    _replay_triangle_relations(based_map)


def test_degenerate_triangle_image_replays_as_free_identity() -> None:
    triangle = _complex(("a", "b", "c"), (("a", "b", "c"),))
    target = _complex(("x", "y"), (("x", "y"),))
    result = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=triangle, target=target, vertex_map=("x", "y", "x")
            ),
            source_base_vertex="a",
            target_base_vertex="x",
        )
    )
    witness = result.relator_images[0]
    assert witness.target_relator_index is None
    assert witness.target_orientation is None
    _replay_triangle_relations(result)
