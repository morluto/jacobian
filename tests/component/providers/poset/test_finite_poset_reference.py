from __future__ import annotations

from itertools import combinations

import networkx as nx

from jacobian.contracts.posets import (
    FinitePosetRequest,
    LinearExtensionRequest,
    MobiusFunctionRequest,
    PosetRequest,
)
from jacobian.domains.posets.operations import (
    _linear_extensions,
    _materialized_poset,
    _mobius,
    _width,
)
from jacobian_checkers.finite_posets import (
    _replay_linear_extensions,
    _replay_mobius,
    _replay_width,
)


def test_all_forward_dag_presentations_through_order_five() -> None:
    checked = 0
    for order in range(6):
        labels = tuple(chr(ord("a") + index) for index in range(order))
        candidate_edges = list(combinations(labels, 2))
        for edge_mask in range(1 << len(candidate_edges)):
            graph = nx.DiGraph()
            graph.add_nodes_from(labels)
            graph.add_edges_from(
                edge
                for index, edge in enumerate(candidate_edges)
                if edge_mask & (1 << index)
            )
            reduction = sorted(nx.transitive_reduction(graph).edges())
            request = FinitePosetRequest(
                elements=labels,
                relation=tuple(
                    {"lower": lower, "upper": upper} for lower, upper in reduction
                ),
                interpretation="COVER_EDGES",
            )
            poset = _materialized_poset(request)
            source = {"poset": poset.model_dump(mode="json")}

            width = _width(PosetRequest(poset=poset)).model_dump(mode="json")
            assert _replay_width(source, width)

            linear = _linear_extensions(LinearExtensionRequest(poset=poset)).model_dump(
                mode="json"
            )
            assert linear["count"] == sum(1 for _ in nx.all_topological_sorts(graph))
            assert _replay_linear_extensions(source, linear)

            mobius_request = MobiusFunctionRequest(poset=poset)
            mobius = _mobius(mobius_request).model_dump(mode="json")
            assert _replay_mobius(
                mobius_request.model_dump(mode="json"),
                mobius,
            )
            checked += 1
    assert checked == 1_100
