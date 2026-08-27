from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDescriptor, OperationDiscoveryRequest
from jacobian.catalog.search import (
    discover_operations,
    discovery_relevance,
    discovery_terms,
)


def test_discovery_phrase_matching_respects_token_boundaries() -> None:
    descriptor = OperationDescriptor(
        operation_id="fixture.text.inspect",
        title="Inspect text",
        description="Inspect some paragraph of structured text.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    graph_score = discovery_relevance(descriptor, "graph")
    phrase_score = discovery_relevance(
        descriptor,
        "paragraph of structured text",
    )

    assert graph_score == 0
    assert phrase_score >= 20


def test_standard_det_abbreviation_ranks_determinants_before_charpolys() -> None:
    catalog = Catalog.open()
    cursor: str | None = None
    matches = []
    while True:
        result = catalog.search(
            OperationDiscoveryRequest(query="det", limit=20, cursor=cursor)
        )
        matches.extend(result.matches)
        if result.next_cursor is None:
            break
        cursor = result.next_cursor

    positions = {match.operation_id: index for index, match in enumerate(matches)}

    determinant_ids = (
        "matrix.determinant.compute",
        "matrix.symbolic.determinant.compute",
    )
    characteristic_polynomial_ids = (
        "matrix.characteristic_polynomial.compute",
        "matrix.symbolic.characteristic_polynomial.compute",
    )
    assert set(determinant_ids) <= positions.keys()
    assert set(characteristic_polynomial_ids) <= positions.keys()
    assert all(
        positions[determinant_id] < positions[characteristic_polynomial_id]
        for determinant_id in determinant_ids
        for characteristic_polynomial_id in characteristic_polynomial_ids
    )


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("sum", "sum"),
        ("sums", "sum"),
        ("representations", "representation"),
        ("points", "point"),
        ("distances", "distance"),
        ("counts", "count"),
        ("cardinalities", "cardinality"),
        ("trees", "tree"),
        ("cuts", "cut"),
        ("classes", "class"),
        ("complexes", "complex"),
        ("matches", "match"),
        ("basis", "basis"),
        ("bases", "bases"),
        ("class", "class"),
        ("grass", "grass"),
        ("series", "series"),
        ("dynamics", "dynamics"),
        ("chaos", "chaos"),
        ("guigues", "guigues"),
        ("sims", "sims"),
        ("sos", "sos"),
        ("does", "does"),
        ("lies", "lies"),
    ],
)
def test_discovery_terms_use_a_conservative_inflection_table(
    term: str,
    expected: str,
) -> None:
    assert discovery_terms(term) == frozenset({expected})


@pytest.mark.parametrize(
    ("field", "weight"),
    [
        ("operation_id", 12),
        ("tags", 10),
        ("title", 8),
        ("description", 3),
    ],
)
def test_discovery_inflection_is_symmetric_across_all_searchable_fields(
    field: str,
    weight: int,
) -> None:
    def descriptor(term: str) -> OperationDescriptor:
        values: dict[str, object] = {
            "operation_id": "fixture.object.inspect",
            "title": "Inspect object",
            "description": "Examine object.",
            "tags": (),
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        }
        values[field] = (
            (term,)
            if field == "tags"
            else f"fixture.{term}.inspect"
            if field == "operation_id"
            else term
        )
        return OperationDescriptor.model_validate(values)

    assert discovery_relevance(descriptor("points"), "point") == weight
    assert discovery_relevance(descriptor("point"), "points") == weight


def test_subset_sum_singular_and_plural_queries_rank_the_profile_ahead_of_sidon() -> (
    None
):
    catalog = Catalog.open()
    for query in (
        "all subset sum and repeated representation of a finite integer set",
        "all subset sums and repeated representations of a finite integer set",
    ):
        result = catalog.search(OperationDiscoveryRequest(query=query, limit=10))
        positions = {
            match.operation_id: index for index, match in enumerate(result.matches)
        }

        assert result.query == query
        assert (
            positions["additive.subset_sum.profile.compute"]
            < positions["combinatorics.integer_set.sidon.decide"]
        )


@pytest.mark.parametrize(
    ("queries", "expected_operation_id"),
    [
        (
            (
                "rational point distance profile",
                "rational points distances profile",
            ),
            "geometry.points.distance_profile.compute",
        ),
        (
            (
                "count independent vertex sets by cardinality in a tree",
                "counts independent vertex sets by cardinalities in trees",
            ),
            "graph.polynomial.independence.compute",
        ),
        (
            ("exact maximum cut of a graph", "exact maximum cuts of a graph"),
            "graph.cut.maximum.compute",
        ),
    ],
)
def test_inflected_catalog_queries_retain_the_same_top_result_and_score(
    queries: tuple[str, str],
    expected_operation_id: str,
) -> None:
    catalog = Catalog.open()
    results = tuple(
        catalog.search(OperationDiscoveryRequest(query=query, limit=10))
        for query in queries
    )

    assert all(
        result.matches[0].operation_id == expected_operation_id for result in results
    )
    assert (
        results[0].matches[0].relevance_score == results[1].matches[0].relevance_score
    )


def test_exact_identifier_phrase_keeps_priority_after_inflection_normalization() -> (
    None
):
    operations = tuple(
        OperationDescriptor(
            operation_id=operation_id,
            title="Inspect object",
            description="Examine object.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        for operation_id in ("fixture.point.compute", "fixture.points.compute")
    )

    result = discover_operations(
        operations,
        OperationDiscoveryRequest(query="fixture.points.compute", limit=2),
    )

    assert result.matches[0].operation_id == "fixture.points.compute"


def test_protected_mathematical_queries_retain_their_top_catalog_matches() -> None:
    catalog = Catalog.open()
    expected_top_matches = {
        "basis": "lattice.canonical_basis.compute",
        "class": "probability.markov_chain.communicating_classes.compute",
        "series": "formal_series.rational.compose.compute",
        "sos": "polynomial.sos.decomposition.check",
    }

    for query, expected_operation_id in expected_top_matches.items():
        result = catalog.search(OperationDiscoveryRequest(query=query, limit=5))

        assert result.matches[0].operation_id == expected_operation_id


def test_inflected_discovery_ties_are_deterministic_and_operation_id_ordered() -> None:
    operations = tuple(
        OperationDescriptor(
            operation_id=operation_id,
            title="Inspect points",
            description="Examine objects.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        for operation_id in ("fixture.z.inspect", "fixture.a.inspect")
    )
    request = OperationDiscoveryRequest(query="point", limit=2)

    forward = discover_operations(operations, request)
    reverse = discover_operations(tuple(reversed(operations)), request)

    assert forward == reverse
    assert [match.operation_id for match in forward.matches] == [
        "fixture.a.inspect",
        "fixture.z.inspect",
    ]
