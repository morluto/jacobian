from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationDiscoveryRequest,
)
from jacobian.catalog.search import discovery_relevance, normalized_discovery_terms


def _positions(query: str) -> dict[str, int]:
    catalog = Catalog.open()
    cursor: str | None = None
    matches: list[OperationDiscoveryMatch] = []
    while True:
        result = catalog.search(
            OperationDiscoveryRequest(query=query, limit=20, cursor=cursor)
        )
        matches.extend(result.matches)
        if result.next_cursor is None:
            break
        cursor = result.next_cursor
    return {match.operation_id: index for index, match in enumerate(matches)}


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
    matches: list[OperationDiscoveryMatch] = []
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


def test_discovery_normalizes_only_audited_ordinary_plural_forms() -> None:
    assert normalized_discovery_terms("subset sums and repeated representations") == {
        "subset",
        "sum",
        "repeated",
        "representation",
    }
    assert normalized_discovery_terms("basis class series") == {
        "basis",
        "class",
        "series",
    }


def test_plural_queries_preserve_their_semantic_catalog_routing() -> None:
    subset_positions = _positions(
        "all subset sums and repeated representations of a finite integer set"
    )
    assert (
        subset_positions["additive.subset_sum.profile.compute"]
        < subset_positions["combinatorics.integer_set.sidon.decide"]
    )

    tree_positions = _positions(
        "counts independent vertex sets by cardinalities in trees"
    )
    assert (
        tree_positions["graph.polynomial.independence.compute"]
        < tree_positions["graph.independent_set.maximal.decide"]
    )


def test_euler_phi_discovery_terms_outrank_generic_inverse_and_solver_operations() -> (
    None
):
    for query, displaced in (
        ("inverse totient preimages", "arithmetic.dirichlet_inverse.compute"),
        ("totient inverse image", "matrix.inverse.compute"),
        ("solve phi(n)=m", "matrix.symbolic.linear_system.solve"),
    ):
        positions = _positions(query)
        assert (
            positions["number_theory.euler_phi.preimages.compute"]
            < positions[displaced]
        )


def test_divisor_discovery_terms_preserve_complete_divisor_postconditions() -> None:
    for query, operation_id, displaced in (
        (
            "aliquot proper divisor list",
            "integer.compute.divisors",
            "graph.chip_firing.canonical_divisor.compute",
        ),
        (
            "sum of proper divisors",
            "integer.compute.divisor_sum",
            "additive.subset_sum.profile.compute",
        ),
        (
            "aliquot sum",
            "integer.compute.divisor_sum",
            "additive.subset_sum.profile.compute",
        ),
    ):
        positions = _positions(query)
        assert positions[operation_id] < positions[displaced]


def test_t_codegree_discovery_terms_route_to_incidence_containment_profiles() -> None:
    for query in (
        "compute t-codegrees of a finite hypergraph",
        "uniform codegree profile",
    ):
        positions = _positions(query)
        assert (
            positions["incidence.containment_profiles.compute"]
            < positions["hypergraph.parameters.compute"]
        )
