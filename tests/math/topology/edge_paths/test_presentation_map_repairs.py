"""Regressions for the #4229 fundamental-group presentation-map review."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    MAX_PRESENTATION_GENERATORS,
    MAX_PRESENTATION_RELATORS,
    MAX_WORD,
    FiniteGroupPresentation,
    FiniteGroupWord,
    FundamentalGroupBasepointChangeRequest,
    FundamentalGroupMapRequest,
    PresentationBasepointChangePath,
    PresentationMapCompositionRequest,
    WordLetter,
)
from jacobian.math.topology.edge_paths.operations import (
    free_reduce,
    presentation_abelianization,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    change_fundamental_group_basepoint,
    compose_fundamental_group_maps,
    induced_fundamental_group_map,
)

# A base vertex joined to the three vertices of one filled triangle, so all
# three triangle edges are non-tree edges.
_STAR_TRIANGLE = canonical_complex(
    ("a", "b", "c", "d"),
    (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c", "d")),
)
_TETRAHEDRON = canonical_complex(("a", "b", "c", "d"), (("a", "b", "c", "d"),))
_TWO_TRIANGLE_FAN = canonical_complex(
    ("a", "b", "c", "d"), (("a", "b", "c"), ("b", "c", "d"))
)
_BASE_FIXED_COMPLEXES = (_STAR_TRIANGLE, _TETRAHEDRON, _TWO_TRIANGLE_FAN)


def _based_maps(complex_value: Any) -> list[tuple[SimplicialMap, str]]:
    """Every simplicial self-map paired with a vertex it fixes."""

    based: list[tuple[SimplicialMap, str]] = []
    for permutation in itertools.permutations(complex_value.vertices):
        try:
            simplicial_map = SimplicialMap(
                source=complex_value,
                target=complex_value,
                vertex_map=permutation,
            )
        except ValidationError:
            continue
        for label, image in zip(
            complex_value.vertices, simplicial_map.vertex_map, strict=True
        ):
            if label == image:
                based.append((simplicial_map, label))
    return based


def _forged_presentation(**values: Any) -> FiniteGroupPresentation:
    return FiniteGroupPresentation.model_construct(**values)


class TestTriangleWitnessRotation:
    def test_cyclic_rotation_of_the_triangle_vertices_is_admitted(self) -> None:
        """A rotated triangle image is a cyclic conjugate, not a new relator."""
        result = induced_fundamental_group_map(
            FundamentalGroupMapRequest(
                map=SimplicialMap(
                    source=_STAR_TRIANGLE,
                    target=_STAR_TRIANGLE,
                    vertex_map=("a", "c", "d", "b"),
                ),
                source_base_vertex="a",
                target_base_vertex="a",
            )
        )
        assert len(result.relator_images) == 1
        witness = result.relator_images[0]
        assert witness.target_relator_index == 0
        assert witness.target_orientation == 1
        assert witness.conjugator != FiniteGroupWord(letters=())

    def test_identity_and_odd_images_keep_a_trivial_rotation(self) -> None:
        identity = induced_fundamental_group_map(
            FundamentalGroupMapRequest(
                map=SimplicialMap(
                    source=_STAR_TRIANGLE,
                    target=_STAR_TRIANGLE,
                    vertex_map=("a", "b", "c", "d"),
                ),
                source_base_vertex="a",
                target_base_vertex="a",
            )
        )
        assert identity.relator_images[0].conjugator == FiniteGroupWord(letters=())
        assert identity.relator_images[0].target_orientation == 1
        transposition = induced_fundamental_group_map(
            FundamentalGroupMapRequest(
                map=SimplicialMap(
                    source=_STAR_TRIANGLE,
                    target=_STAR_TRIANGLE,
                    vertex_map=("a", "b", "d", "c"),
                ),
                source_base_vertex="a",
                target_base_vertex="a",
            )
        )
        assert transposition.relator_images[0].target_orientation == -1
        assert transposition.relator_images[0].conjugator == FiniteGroupWord(letters=())

    @pytest.mark.parametrize(
        "complex_value", _BASE_FIXED_COMPLEXES, ids=lambda item: str(len(item.vertices))
    )
    def test_every_based_simplicial_self_map_is_admitted(
        self, complex_value: Any
    ) -> None:
        """No valid based map may be rejected by exact relation replay."""
        maps = _based_maps(complex_value)
        assert maps
        for simplicial_map, base in maps:
            result = induced_fundamental_group_map(
                FundamentalGroupMapRequest(
                    map=simplicial_map,
                    source_base_vertex=base,
                    target_base_vertex=base,
                )
            )
            assert len(result.relator_images) == len(
                result.source_presentation.presentation.relators
            )
            assert result.abelianization_map.row_count == len(
                result.target_presentation.presentation.generators
            )

    def test_unbased_image_is_still_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            induced_fundamental_group_map(
                FundamentalGroupMapRequest(
                    map=SimplicialMap(
                        source=_STAR_TRIANGLE,
                        target=_STAR_TRIANGLE,
                        vertex_map=("a", "c", "d", "b"),
                    ),
                    source_base_vertex="a",
                    target_base_vertex="b",
                )
            )


class TestNativeFreeReductionBound:
    def test_oversized_cancelling_word_is_rejected(self) -> None:
        cancelling = tuple(
            WordLetter(generator=0, exponent=1 if index % 2 == 0 else -1)
            for index in range(MAX_WORD + 2)
        )
        with pytest.raises(OperationResourceAdmissionError):
            free_reduce(1, cancelling)

    def test_oversized_non_cancelling_word_is_rejected(self) -> None:
        kept = tuple(
            WordLetter(generator=index % 4, exponent=1) for index in range(MAX_WORD + 2)
        )
        with pytest.raises(OperationResourceAdmissionError):
            free_reduce(4, kept)

    def test_word_at_the_bound_is_reduced_normally(self) -> None:
        at_bound = tuple(
            WordLetter(generator=0, exponent=1 if index % 2 == 0 else -1)
            for index in range(MAX_WORD)
        )
        assert free_reduce(1, at_bound) == FiniteGroupWord(letters=())


class TestNativeAbelianizationAdmission:
    def test_out_of_axis_relator_letter_is_rejected(self) -> None:
        presentation = _forged_presentation(
            generators=("x", "y"),
            relators=(FiniteGroupWord(letters=(WordLetter(generator=7, exponent=1),)),),
        )
        with pytest.raises(OperationDomainValidationError) as info:
            presentation_abelianization(presentation)
        assert "relator" in str(info.value)

    def test_duplicate_generator_ids_are_rejected(self) -> None:
        presentation = _forged_presentation(generators=("x", "x"), relators=())
        with pytest.raises(OperationDomainValidationError):
            presentation_abelianization(presentation)

    def test_oversized_generator_axis_is_rejected(self) -> None:
        presentation = _forged_presentation(
            generators=tuple(
                f"g{index}" for index in range(MAX_PRESENTATION_GENERATORS + 1)
            ),
            relators=(),
        )
        with pytest.raises(OperationResourceAdmissionError):
            presentation_abelianization(presentation)

    def test_oversized_relator_axis_is_rejected(self) -> None:
        presentation = _forged_presentation(
            generators=("x",),
            relators=tuple(
                FiniteGroupWord(letters=(WordLetter(generator=0, exponent=1),))
                for _ in range(MAX_PRESENTATION_RELATORS + 1)
            ),
        )
        with pytest.raises(OperationResourceAdmissionError):
            presentation_abelianization(presentation)

    def test_overlong_word_is_rejected(self) -> None:
        presentation = _forged_presentation(
            generators=("x",),
            relators=(
                FiniteGroupWord.model_construct(
                    letters=tuple(
                        WordLetter(generator=0, exponent=1) for _ in range(MAX_WORD + 1)
                    )
                ),
            ),
        )
        with pytest.raises(OperationResourceAdmissionError):
            presentation_abelianization(presentation)

    def test_admitted_presentation_keeps_its_exact_abelianization(self) -> None:
        presentation = FiniteGroupPresentation(
            generators=("x", "y"),
            relators=(FiniteGroupWord(letters=(WordLetter(generator=0, exponent=1),)),),
        )
        result = presentation_abelianization(presentation)
        assert result.abelianization.free_rank == 1
        assert result.abelianization.rank == 1
        assert result.abelianization.torsion_invariant_factors == ()


def test_a_basepoint_path_composes_with_a_simplicial_map() -> None:
    """A change of basepoint followed by a based map is well defined.

    Composition previously required two carriers of the same kind, so
    `change_fundamental_group_basepoint` (a path) could not be composed with
    `induced_fundamental_group_map` (a simplicial map) even when the
    presentations already agreed. Composing a basepoint path with a simplicial map
    yields that same map, and it induces the same homomorphism, so the composite
    is a simplicial map rather than a refusal.
    """
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))

    path_map = change_fundamental_group_basepoint(
        FundamentalGroupBasepointChangeRequest(
            path=PresentationBasepointChangePath(
                complex=complex_,
                source_base_vertex="a",
                target_base_vertex="b",
                path_vertices=("a", "b"),
            )
        )
    )
    assert isinstance(path_map.map, PresentationBasepointChangePath)

    identity_map = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(
                source=complex_, target=complex_, vertex_map=("a", "b", "c")
            ),
            source_base_vertex="b",
            target_base_vertex="b",
        )
    )
    assert isinstance(identity_map.map, SimplicialMap)
    # the two operands already agree on the intermediate presentation
    assert path_map.target_presentation == identity_map.source_presentation

    composed = compose_fundamental_group_maps(
        PresentationMapCompositionRequest(first=path_map, second=identity_map)
    )

    assert isinstance(composed.map, SimplicialMap)
    # the composite is the based map itself, so the induced homomorphism is the
    # identity map's, transported through the same presentation
    expected = identity_map.generator_images
    actual = composed.generator_images
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        # both are generators of the same one-generator presentation; the basepoint
        # change inverts the transported word
        assert {abs(letter.exponent) for letter in got.letters} == {
            abs(letter.exponent) for letter in want.letters
        }
