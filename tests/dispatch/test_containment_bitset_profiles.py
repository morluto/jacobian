"""Admitted bitset profiles and source-bound trade data across JSON consumers."""

import json
from itertools import combinations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.designs.incidence_structures import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceTradeResult,
    containment_profile,
)
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    verify_incidence_moment_comparison,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph


@pytest.mark.parametrize(
    ("point_count", "edge_count", "order"), [(14, 2500, 7), (0, 0, 0), (0, 0, 1)]
)
def test_profile_json_preserves_new_boundary_and_real_source_consumer(
    point_count: int, edge_count: int, order: int
) -> None:
    points = tuple(f"p{i}" for i in range(point_count))
    source = FiniteHypergraph(
        vertices=points, edges=tuple((f"e{i}", points) for i in range(edge_count))
    )
    native = containment_profile(source, order)
    catalog = Catalog.open()
    output = invoke_operation(
        "incidence.containment_profiles.compute",
        {"incidence": source.model_dump(mode="json"), "t": order},
        catalog,
    ).output
    decoded = ContainmentProfileResult.model_validate_json(json.dumps(output))
    assert decoded == native
    assert decoded.subset_profile == tuple(
        (subset, edge_count) for subset in combinations(points, order)
    )
    # The unchanged canonical source from the actual result feeds its existing
    # hypergraph consumer; no reconstructed or narrowed source substitutes.
    parameters = invoke_operation(
        "hypergraph.parameters.compute", {"hypergraph": output["incidence"]}, catalog
    ).output
    assert parameters["vertex_count"] == point_count
    assert parameters["edge_count"] == edge_count
    assert parameters["total_incidences"] == point_count * edge_count


def test_repeated_order_trade_json_handoff_and_false_claim_rejection() -> None:
    points = tuple(f"p{i}" for i in range(14))
    left = IncidenceStructure(
        points=points,
        block_ids=tuple(f"a{i}" for i in range(100)),
        blocks=(points,) * 100,
    )
    right = IncidenceStructure(
        points=points,
        block_ids=tuple(f"b{i}" for i in range(99)),
        blocks=(points,) * 99,
    )
    output = invoke_operation(
        "incidence.trade.check",
        {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "max_order": 7,
        },
        Catalog.open(),
    ).output
    result = IncidenceTradeResult.model_validate_json(json.dumps(output))
    assert result.zeroth_difference == 1
    assert not result.positive_moments_equal
    for comparison in result.comparisons:
        assert tuple(
            (entry.subset, entry.left_multiplicity, entry.right_multiplicity)
            for entry in comparison.differences
        ) == tuple(
            (subset, 100, 99) for subset in combinations(points, comparison.order)
        )
        assert verify_incidence_moment_comparison(comparison)
    # Mutate retained source and derived arithmetic independently. Both claims
    # remain structurally valid but their consumer must reject the relation.
    comparison_wire = result.comparisons[-1].model_dump(mode="json")
    comparison_wire["left_total"] += 1
    forged_total = IncidenceMomentComparison.model_validate_json(
        json.dumps(comparison_wire)
    )
    assert not verify_incidence_moment_comparison(forged_total)
    comparison_wire = result.comparisons[-1].model_dump(mode="json")
    comparison_wire["left"]["blocks"][0] = []
    forged_source = IncidenceMomentComparison.model_validate_json(
        json.dumps(comparison_wire)
    )
    assert not verify_incidence_moment_comparison(forged_source)
