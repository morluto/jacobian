"""Independent oracle checks for deterministic greedy elementary collapses."""

from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.discrete_morse._models import (
    MorseMatchingOutcome,
)
from jacobian.math.topology.discrete_morse.extensions import (
    CollapseSequenceRequest,
    GreedyCollapseRequest,
    collapse_sequence,
    greedy_collapse,
)
from jacobian.math.topology.discrete_morse.operations import construct_matching


def _all_faces(facets):
    return {
        subset
        for facet in facets
        for size in range(1, len(facet) + 1)
        for subset in combinations(facet, size)
    }


def _facets_of(faces):
    return tuple(
        sorted(
            face for face in faces if not any(set(face) < set(other) for other in faces)
        )
    )


def _oracle(facets):
    """Rebuild all faces and discover free pairs without production helpers."""
    faces = _all_faces(facets)
    pairs = []
    while True:
        maximal = _facets_of(faces)
        candidates = []
        for coface in maximal:
            if len(coface) < 2:
                continue
            for face in combinations(coface, len(coface) - 1):
                containing = [cell for cell in maximal if set(face) <= set(cell)]
                if containing == [coface]:
                    candidates.append((face, coface))
        if not candidates:
            return tuple(pairs), faces
        face, coface = min(candidates)
        pairs.append((face, coface))
        faces.remove(face)
        faces.remove(coface)


@pytest.mark.parametrize(
    "facets",
    [
        (("a", "b", "c"),),
        (("a", "b"), ("b", "c"), ("a", "c")),
        (("a", "b"), ("c",)),
        (("a", "b", "c"), ("c", "d")),
    ],
)
def test_greedy_collapse_matches_naive_face_poset_oracle(facets) -> None:
    vertices = tuple(sorted({vertex for facet in facets for vertex in facet}))
    result = greedy_collapse(
        GreedyCollapseRequest(complex={"vertices": vertices, "facets": facets})
    )
    expected_pairs, expected_faces = _oracle(tuple(sorted(facets)))

    assert tuple((pair.face, pair.coface) for pair in result.pairs) == expected_pairs
    assert result.collapsed_steps == len(expected_pairs)
    assert result.valid
    assert result.target is not None
    assert {
        face for degree in result.target.faces_by_dimension for face in degree.faces
    } == expected_faces


def test_greedy_collapse_composes_with_sequence_and_acyclic_matching() -> None:
    result = greedy_collapse(
        GreedyCollapseRequest(
            complex={"vertices": ("a", "b", "c"), "facets": (("a", "b", "c"),)}
        )
    )
    replayed = collapse_sequence(
        CollapseSequenceRequest(complex=result.source, pairs=result.pairs)
    )
    matching = construct_matching(result.source, result.pairs)

    assert replayed.valid
    assert replayed.target == result.target
    assert matching.outcome is MorseMatchingOutcome.ACYCLIC_MATCHING


@pytest.mark.parametrize(
    ("limit_name", "limit_value"),
    [
        ("MAX_GREEDY_COLLAPSE_WORK", 1),
        ("MAX_TOPOLOGY_FACES", 1),
    ],
)
def test_greedy_collapse_preflight_rejects_before_pair_search(
    monkeypatch, limit_name, limit_value
) -> None:
    import jacobian.math.topology.discrete_morse.extensions as extensions

    monkeypatch.setattr(extensions, limit_name, limit_value)
    monkeypatch.setattr(
        extensions,
        "_first_free_pair",
        lambda facets: pytest.fail("candidate search ran before admission"),
    )
    request = GreedyCollapseRequest(
        complex={"vertices": ("a", "b"), "facets": (("a", "b"),)}
    )
    with pytest.raises(OperationResourceAdmissionError):
        greedy_collapse(request)
