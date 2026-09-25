"""Functorial composition of exact finite pi_1 presentation maps."""

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


def _circle():
    return canonical_complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))


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
