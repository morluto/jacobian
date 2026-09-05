"""Existing atomic operations for edge-clique partition mathematics."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest

_CANDIDATES = "graph.clique_candidate_hypergraph.construct"
_PACKING = "hypergraph.maximum_weight_packing.compute"
_CHECK = "graph.edge_clique_partition.check"
_SIGNED = "graph.signed_clique_weight.maximum.compute"
_COLORING = "graph.coloring.chromatic_number.check"


@pytest.mark.parametrize(
    "query",
    [
        "For the graph K5 minus one edge, compute the minimum edge clique partition "
        "and the fractional edge clique partition with equality constraints, "
        "allowing every clique size. Return an integral partition and exact "
        "rational primal and unrestricted-sign dual certificates, and "
        "independently verify the certificates.",
        "Minimum edge-clique partition number: edge-disjoint cliques of all sizes, "
        "an integral partition certificate and a fractional equality partition "
        "dual with signed edge weights.",
        "Partition graph edges exactly once into complete subgraphs; optimize "
        "the number of parts and certify the fractional edge partition bound "
        "with unrestricted-sign dual weights over all nontrivial cliques.",
    ],
)
def test_edge_partition_operations_share_first_page(query: str) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=10)).matches
    ids = [match.operation_id for match in matches]
    relevant = {_CANDIDATES, _PACKING, _CHECK, _SIGNED}
    assert relevant <= set(ids)
    for unrelated in (_COLORING, "combinatorial_map.dual.compute"):
        if unrelated in ids:
            assert max(ids.index(item) for item in relevant) < ids.index(unrelated)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "unrestricted-sign edge weights fractional edge clique partition dual "
            "constraints over every nontrivial clique, not just maximal cliques",
            _SIGNED,
        ),
        (
            "all nontrivial cliques including nonmaximal cliques as edge resources "
            "for edge-disjoint clique partition",
            _CANDIDATES,
        ),
        (
            "verify an edge clique partition covers every graph edge exactly once",
            _CHECK,
        ),
        ("check a graph vertex coloring chromatic number certificate", _COLORING),
        (
            "check vertex chromatic number using nonnegative fractional clique "
            "vertex weights and a proper coloring",
            _COLORING,
        ),
        (
            "edge clique partition with disjoint parts, not an overlapping edge cover",
            _PACKING,
        ),
    ],
)
def test_distinct_mathematical_needs_retain_routing(query: str, expected: str) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=10)).matches
    assert matches[0].operation_id == expected
