"""Independent face-set oracle for the typed simplicial face polynomial."""

from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.topology._structural import (
    FaceEnumeratorRequest,
    compute_face_enumerator,
)


def _oracle(facets: tuple[tuple[str, ...], ...]) -> tuple[int, ...]:
    """Count distinct subsets directly, independently of the topology kernel."""
    faces = {
        frozenset(face)
        for facet in facets
        for size in range(1, len(facet) + 1)
        for face in combinations(facet, size)
    }
    degree = max(map(len, faces))
    descending = tuple(
        sum(len(face) == size for face in faces) for size in range(degree, 0, -1)
    )
    return (*descending, 1)


@pytest.mark.parametrize(
    "vertices,facets",
    [
        (("a",), (("a",),)),
        (("a", "b", "c"), (("a", "b", "c"),)),
        (("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"))),
        (("a", "b", "c", "d"), (("a", "b", "c"), ("a", "b", "d"))),
    ],
)
def test_face_enumerator_matches_independent_subset_oracle(vertices, facets) -> None:
    result = compute_face_enumerator(
        FaceEnumeratorRequest(complex={"vertices": vertices, "facets": facets})
    )
    assert isinstance(result, IntegerPolynomial)
    assert result.coefficients == _oracle(facets)


def test_face_enumerator_preflights_candidate_expansion(monkeypatch) -> None:
    import jacobian.math.topology._structural as structural

    def unexpected_expansion(_facets, *, max_faces):
        raise AssertionError("face closure expanded before the candidate bound")

    monkeypatch.setattr(structural, "MAX_FACE_ENUMERATOR_CANDIDATES", 1)
    monkeypatch.setattr(structural, "_bounded_face_closure", unexpected_expansion)
    request = FaceEnumeratorRequest(
        complex={"vertices": ("a", "b"), "facets": (("a", "b"),)}
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_face_enumerator(request)
    assert error.value.errors()[0]["type"] == (
        "topology.face_enumerator.admission.face_candidates"
    )


def test_face_enumerator_stops_before_inserting_overflow_face(monkeypatch) -> None:
    import jacobian.math.topology._structural as structural

    original_combinations = structural.combinations
    generated = 0

    def counted_combinations(items, size):
        nonlocal generated
        for face in original_combinations(items, size):
            generated += 1
            yield face

    monkeypatch.setattr(structural, "MAX_TOPOLOGY_FACES", 2)
    monkeypatch.setattr(structural, "combinations", counted_combinations)
    request = FaceEnumeratorRequest(
        complex={"vertices": ("a", "b"), "facets": (("a", "b"),)}
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_face_enumerator(request)
    assert error.value.errors()[0]["type"] == (
        "topology.face_enumerator.admission.output_faces"
    )
    assert generated == 3  # exactly the first face beyond the two-face cap


def test_face_enumerator_accepts_exact_output_face_cap(monkeypatch) -> None:
    import jacobian.math.topology._structural as structural

    monkeypatch.setattr(structural, "MAX_TOPOLOGY_FACES", 3)
    request = FaceEnumeratorRequest(
        complex={"vertices": ("a", "b"), "facets": (("a", "b"),)}
    )
    assert compute_face_enumerator(request).coefficients == (1, 2, 1)
