from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDescriptor, OperationDiscoveryRequest
from jacobian.catalog.search import discovery_relevance


def test_discovery_phrase_matching_respects_token_boundaries() -> None:
    descriptor = OperationDescriptor(
        operation_id="fixture.text.inspect",
        version="1",
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
