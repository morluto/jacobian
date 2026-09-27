"""Exact bounded minimal-nonface enumeration."""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

import jacobian.math.topology._structural as structural
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology._simplicial_kernel import canonicalize
from jacobian.math.topology._structural import (
    MAX_MINIMAL_NONFACE_CANDIDATES,
    MinimalNonfacesRequest,
    MinimalNonfacesResult,
    compute_minimal_nonfaces,
)


def _complex(
    vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]
) -> FiniteSimplicialComplex:
    return canonicalize(vertices, facets).complex


def _request(
    vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]
) -> MinimalNonfacesRequest:
    return MinimalNonfacesRequest(complex=_complex(vertices, facets))


def _subset_oracle(
    vertices: tuple[str, ...], facets: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    """Enumerate every subset and every proper subface independently."""
    face_set: set[frozenset[str]] = {frozenset()}
    for facet in facets:
        for size in range(1, len(facet) + 1):
            face_set.update(frozenset(part) for part in combinations(facet, size))
    nonfaces = []
    for size in range(1, len(vertices) + 1):
        for subset in combinations(vertices, size):
            key = frozenset(subset)
            if key in face_set:
                continue
            proper = (
                frozenset(part)
                for proper_size in range(size)
                for part in combinations(subset, proper_size)
            )
            if all(part in face_set for part in proper):
                nonfaces.append(subset)
    return tuple(nonfaces)


@pytest.mark.parametrize(
    ("vertices", "facets"),
    [
        (("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))),
        (("a", "b", "c", "d"), (("a",), ("b",), ("c",), ("d",))),
        (("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d"))),
        (("w", "x", "y", "z"), (("w", "x"), ("y", "z"))),
    ],
)
def test_matches_independent_all_proper_subsets_oracle(vertices, facets) -> None:
    source = _complex(vertices, facets)
    result = compute_minimal_nonfaces(MinimalNonfacesRequest(complex=source))
    assert result.minimal_nonfaces == _subset_oracle(
        source.vertices, source.maximal_simplices
    )
    assert result.source == source
    assert all(tuple(sorted(face)) == face for face in result.minimal_nonfaces)


def test_full_simplex_has_empty_antichain_and_round_trips_exactly() -> None:
    source = _complex(("c", "a", "b"), (("a", "b", "c"),))
    result = compute_minimal_nonfaces(MinimalNonfacesRequest(complex=source))
    assert result.minimal_nonfaces == ()
    assert result.source.vertices == ("a", "b", "c")
    encoded = result.model_dump_json()
    decoded = MinimalNonfacesResult.model_validate_json(encoded)
    assert decoded == result
    assert (
        structural._minimal_nonface_result_cells_bound(result.source)
        <= structural.MAX_MINIMAL_NONFACE_RESULT_CELLS
    )


def test_nonface_labels_keep_the_canonical_source_vertex_axis() -> None:
    source = _complex(
        ("z", "a", "m"),
        (("z", "a"), ("z", "m"), ("a", "m")),
    )
    result = compute_minimal_nonfaces(MinimalNonfacesRequest(complex=source))
    assert result.source.vertices == ("a", "m", "z")
    assert result.minimal_nonfaces == (("a", "m", "z"),)


def test_candidate_cap_accepts_14_vertices_and_rejects_15_before_expansion() -> None:
    accepted_vertices = tuple(f"v{i:02}" for i in range(14))
    accepted = _request(accepted_vertices, tuple((v,) for v in accepted_vertices))
    assert 1 << len(accepted_vertices) == MAX_MINIMAL_NONFACE_CANDIDATES
    result = compute_minimal_nonfaces(accepted)
    assert len(result.minimal_nonfaces) == 91
    encoded = result.model_dump_json()
    assert (
        structural._minimal_nonface_result_cells_bound(result.source)
        <= structural.MAX_MINIMAL_NONFACE_RESULT_CELLS
    )
    assert MinimalNonfacesResult.model_validate_json(encoded) == result

    rejected_vertices = tuple(f"v{i:02}" for i in range(15))
    rejected = _request(rejected_vertices, tuple((v,) for v in rejected_vertices))
    with pytest.raises(OperationResourceAdmissionError) as raised:
        compute_minimal_nonfaces(rejected)
    assert (
        raised.value.errors()[0]["type"]
        == "topology.minimal_nonfaces.powerset_over_envelope"
    )


@pytest.mark.parametrize(
    "max_work,max_cells,expected_code",
    [(0, None, "work_over_envelope"), (None, 1, "output_over_envelope")],
)
def test_work_and_output_are_preflighted_before_face_closure(
    monkeypatch, max_work, max_cells, expected_code
) -> None:
    request = _request(("a", "b", "c"), (("a",), ("b",), ("c",)))
    if max_work is not None:
        monkeypatch.setattr(structural, "MAX_MINIMAL_NONFACE_WORK", max_work)
    if max_cells is not None:
        monkeypatch.setattr(structural, "MAX_MINIMAL_NONFACE_RESULT_CELLS", max_cells)

    def unexpected_closure(_facets):
        pytest.fail("request expanded the source closure before admission")

    monkeypatch.setattr(structural, "face_closure", unexpected_closure)
    with pytest.raises(OperationResourceAdmissionError) as raised:
        compute_minimal_nonfaces(request)
    assert expected_code in raised.value.errors()[0]["type"]


@pytest.mark.parametrize("facets", [(), ((),)])
def test_void_and_zero_vertex_values_are_rejected_by_existing_carrier(facets) -> None:
    malformed = FiniteSimplicialComplex.model_construct(
        vertices=(),
        maximal_simplices=facets,
        faces_by_dimension=(),
        dimension=-1,
        f_vector=(),
        closure_size=0,
    )
    request = MinimalNonfacesRequest.model_construct(complex=malformed)
    with pytest.raises(OperationDomainValidationError) as raised:
        compute_minimal_nonfaces(request)
    assert (
        raised.value.errors()[0]["type"]
        == "topology.minimal_nonfaces.empty_source_unsupported"
    )


def test_zero_vertex_presentation_is_not_a_valid_canonical_request() -> None:
    with pytest.raises(ValidationError):
        SimplicialComplexRequest.model_validate({"vertices": [], "facets": []})
    with pytest.raises(ValidationError):
        SimplicialComplexRequest.model_validate({"vertices": [], "facets": [[]]})
