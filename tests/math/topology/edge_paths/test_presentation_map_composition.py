"""Functorial composition of exact finite pi_1 presentation maps."""

import json

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupWord,
    FundamentalGroupMapRequest,
    FundamentalGroupMapResult,
    PresentationMapCompositionRequest,
    WordLetter,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    compose_fundamental_group_maps,
    induced_fundamental_group_map,
)


def _circle():
    return canonical_complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))


def _triangle():
    return canonical_complex(("a", "b", "c"), (("a", "b", "c"),))


def _map(complex_, labels, source_base="a", target_base="a"):
    return induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=complex_, target=complex_, vertex_map=labels),
            source_base_vertex=source_base,
            target_base_vertex=target_base,
        )
    )


def _letters(word):
    return tuple((letter.generator, letter.exponent) for letter in word.letters)


def _replay(result: FundamentalGroupMapResult) -> None:
    """Independent free-word replay of every relation witness in a result."""

    for relator, witness in zip(
        result.source_presentation.presentation.relators,
        result.relator_images,
        strict=True,
    ):
        image: list[tuple[int, int]] = []
        for letter in relator.letters:
            letters = _letters(result.generator_images[letter.generator])
            if letter.exponent == -1:
                letters = tuple(
                    (generator, -sign) for generator, sign in reversed(letters)
                )
            image.extend(letters)
        reduced: list[tuple[int, int]] = []
        for letter in image:
            if reduced and reduced[-1] == (letter[0], -letter[1]):
                reduced.pop()
            else:
                reduced.append(letter)
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
            scratch = [*conjugator, *relation, *inverse_conjugator]
            collapsed: list[tuple[int, int]] = []
            for letter in scratch:
                if collapsed and collapsed[-1] == (letter[0], -letter[1]):
                    collapsed.pop()
                else:
                    collapsed.append(letter)
            expected = tuple(collapsed)
        assert tuple(reduced) == expected


def test_reflection_composition_matches_identity_independently_and_on_h1():
    circle = _circle()
    reflection = _map(circle, ("a", "c", "b"))
    identity = _map(circle, ("a", "b", "c"))
    identity = _map(circle, ("a", "b", "c"))

    result = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=reflection, second=reflection)
    )

    # Independent free-word oracle: (g^-1)^-1 = g, and the induced Z map is
    # multiplication by (-1)(-1)=1.
    assert tuple(_letters(word) for word in result.generator_images) == (((0, 1),),)
    assert result.abelianization_map.entries == ((1,),)
    assert result.source_presentation == identity.source_presentation
    assert result.target_presentation == identity.target_presentation
    assert tuple(_letters(word) for word in result.generator_images) == tuple(
        _letters(word) for word in identity.generator_images
    )


def test_composition_requires_the_exact_middle_presentation_and_basepoint():
    circle = _circle()
    reflection = _map(circle, ("a", "c", "b"))
    changed_middle = _map(circle, ("c", "a", "b"), "b", "a")
    with pytest.raises(OperationDomainValidationError, match="target presentation"):
        compose_fundamental_group_maps(
            PresentationMapCompositionRequest(
                first=reflection,
                second=changed_middle,
            )
        )


def test_composite_is_the_canonical_composable_carrier():
    circle = _circle()
    identity = _map(circle, ("a", "b", "c"))
    reflection = _map(circle, ("a", "c", "b"))

    composed = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=reflection, second=reflection)
    )
    assert isinstance(composed, FundamentalGroupMapResult)
    assert composed.map.vertex_map == identity.map.vertex_map

    # The serialized composite feeds back unchanged as either operand.
    payload = json.loads(composed.model_dump_json())
    decoded = FundamentalGroupMapResult.model_validate_json(json.dumps(payload))
    assert decoded == composed
    followed = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=decoded, second=reflection)
    )
    assert tuple(_letters(word) for word in followed.generator_images) == (((0, -1),),)
    assert followed.abelianization_map.entries == ((-1,),)


def test_threefold_composition_is_associative_on_both_associations():
    triangle = _triangle()
    identity = _map(triangle, ("a", "b", "c"))
    swap = _map(triangle, ("a", "c", "b"))

    # (h o g) o f with f = swap, g = swap, h = swap.
    hg = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=swap, second=swap)
    )
    left = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=swap, second=hg)
    )
    # h o (g o f).
    gf = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=swap, second=swap)
    )
    right = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=gf, second=swap)
    )

    assert left.map.vertex_map == right.map.vertex_map
    assert tuple(_letters(word) for word in left.generator_images) == tuple(
        _letters(word) for word in right.generator_images
    )
    assert left.abelianization_map == right.abelianization_map
    assert left.relator_images == right.relator_images
    assert left.relator_images[0].target_orientation == -1
    _replay(left)
    _replay(right)
    assert identity.relator_images[0].target_orientation == 1


def test_out_of_axis_generator_word_is_a_structured_rejection():
    circle = _circle()
    identity = _map(circle, ("a", "b", "c"))
    out_of_axis = FiniteGroupWord(letters=(WordLetter(generator=7, exponent=1),))
    bad_first = identity.model_copy(update={"generator_images": (out_of_axis,)})
    # ``model_construct`` reproduces the native callable path that bypasses
    # request parsing, which normally revalidates the nested carriers.
    request = PresentationMapCompositionRequest.model_construct(
        first=bad_first, second=identity
    )
    with pytest.raises(OperationDomainValidationError, match="outside the target axis"):
        compose_fundamental_group_maps(request)


def test_out_of_axis_relator_witness_is_a_structured_rejection():
    triangle = _triangle()
    identity = _map(triangle, ("a", "b", "c"))
    changed = identity.relator_images[0].model_copy(update={"target_relator_index": 9})
    bad_second = identity.model_copy(update={"relator_images": (changed,)})
    request = PresentationMapCompositionRequest.model_construct(
        first=identity, second=bad_second
    )
    with pytest.raises(OperationDomainValidationError, match="outside the target axis"):
        compose_fundamental_group_maps(request)


def test_out_of_axis_conjugator_is_a_structured_rejection():
    triangle = _triangle()
    identity = _map(triangle, ("a", "b", "c"))
    changed = identity.relator_images[0].model_copy(
        update={
            "conjugator": FiniteGroupWord(
                letters=(WordLetter(generator=5, exponent=1),)
            )
        }
    )
    bad_first = identity.model_copy(update={"relator_images": (changed,)})
    request = PresentationMapCompositionRequest.model_construct(
        first=bad_first, second=identity
    )
    with pytest.raises(OperationDomainValidationError, match="outside the target axis"):
        compose_fundamental_group_maps(request)


def test_composition_telescopes_nontrivial_conjugators():
    source = canonical_complex(
        ("a", "b", "c", "r", "x"),
        (("r", "x"), ("a", "x"), ("a", "b", "c")),
    )
    target = canonical_complex(
        ("0", "a", "b", "c", "d"),
        (("0", "a"), ("0", "b"), ("0", "c"), ("a", "b"), ("b", "c", "d")),
    )
    based = induced_fundamental_group_map(
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
    assert based.relator_images[0].conjugator.letters
    identity = _map(target, ("0", "a", "b", "c", "d"), "0", "0")

    composed = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=based, second=identity)
    )
    assert composed.relator_images[0].conjugator == based.relator_images[0].conjugator
    assert (
        composed.relator_images[0].target_orientation
        == based.relator_images[0].target_orientation
    )
    _replay(composed)

    # The composite is a valid operand for a further composition.
    again = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=composed, second=identity)
    )
    assert again.relator_images[0].conjugator == based.relator_images[0].conjugator
    _replay(again)
