"""Independent checks of bounded sequential simplicial collapses."""

from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.discrete_morse._models import MatchingPair
from jacobian.math.topology.discrete_morse.extensions import (
    CollapseSequenceRequest,
    collapse_sequence,
)


def _oracle_collapse(facets, face, coface):
    """Rebuild the face family independently and apply one elementary collapse."""
    cells = {
        subset
        for facet in facets
        for size in range(1, len(facet) + 1)
        for subset in combinations(facet, size)
    }
    face_set, coface_set = set(face), set(coface)
    containing_facets = [facet for facet in facets if face_set.issubset(facet)]
    if containing_facets != [coface]:
        return None
    return {
        cell
        for cell in cells
        if not (face_set.issubset(cell) and set(cell).issubset(coface_set))
    }


def test_collapse_sequence_matches_independent_face_family_oracle() -> None:
    source_facets = (("a", "b", "c"),)
    pairs = (
        MatchingPair(face=("a", "b"), coface=("a", "b", "c")),
        MatchingPair(face=("a",), coface=("a", "c")),
        MatchingPair(face=("b",), coface=("b", "c")),
    )
    current = {
        subset
        for facet in source_facets
        for size in range(1, len(facet) + 1)
        for subset in combinations(facet, size)
    }
    facets = source_facets
    for pair in pairs:
        current = _oracle_collapse(facets, pair.face, pair.coface)
        assert current is not None
        facets = tuple(
            sorted(
                cell
                for cell in current
                if not any(
                    tuple(sorted((*cell, vertex))) in current
                    for vertex in {v for item in current for v in item}.difference(cell)
                )
            )
        )

    request = CollapseSequenceRequest(
        complex={"vertices": ("a", "b", "c"), "facets": source_facets},
        pairs=pairs,
    )
    result = collapse_sequence(request)
    assert result.valid
    assert result.collapsed_steps == 3
    assert result.target is not None
    assert result.target.vertices == ("c",)
    assert {
        face for degree in result.target.faces_by_dimension for face in degree.faces
    } == current


def test_nonfree_pair_reports_the_unchanged_current_complex() -> None:
    request = CollapseSequenceRequest(
        complex={"vertices": ("a", "b", "c"), "facets": (("a", "b", "c"),)},
        pairs=(MatchingPair(face=("a",), coface=("a", "b")),),
    )
    result = collapse_sequence(request)
    assert not result.valid
    assert result.collapsed_steps == 0
    assert result.target == result.source


def test_sequence_work_bound_rejects_before_processing_pairs(monkeypatch) -> None:
    import jacobian.math.topology.discrete_morse.extensions as extension_module

    monkeypatch.setattr(extension_module, "MAX_COLLAPSE_SEQUENCE_FACE_WORK", 1)
    request = CollapseSequenceRequest(
        complex={"vertices": ("a", "b"), "facets": (("a", "b"),)},
        pairs=(MatchingPair(face=("a",), coface=("a", "b")),),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        collapse_sequence(request)
    assert (
        error.value.errors()[0]["type"]
        == "topology.collapse_sequence.admission.face_work"
    )
