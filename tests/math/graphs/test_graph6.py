from __future__ import annotations

from random import Random

import networkx as nx
import pytest

from jacobian.math.graphs.graph6 import decode_graph6


def test_decode_graph6_matches_networkx_on_standard_encodings() -> None:
    decoded = decode_graph6("Bw")
    graph = nx.from_graph6_bytes(b"Bw")
    assert decoded.order == graph.order() == 3
    assert [(edge.first, edge.second) for edge in decoded.edges] == sorted(
        graph.edges()
    )
    assert decoded.degrees == (2, 2, 2)


def test_decode_graph6_strips_standard_header() -> None:
    assert decode_graph6(">>graph6<<Bw").graph6 == "Bw"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (":A", "only standard graph6"),
        ("&A", "only standard graph6"),
        ("~A", "extended header"),
        ("Bw\n", "malformed"),
        ("A@", "padding bits"),
        ("A", "length"),
    ],
)
def test_decode_graph6_rejects_encodings_networkx_would_repair(
    payload: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        decode_graph6(payload)


def test_decode_graph6_agrees_with_networkx_for_small_orders() -> None:
    random = Random(1119)
    for order in range(63):
        graph = nx.gnp_random_graph(order, 0.35, seed=random.randrange(10**6))
        payload = nx.to_graph6_bytes(graph, header=False).decode("ascii").rstrip("\n")
        decoded = decode_graph6(payload)
        assert decoded.order == order
        assert [(edge.first, edge.second) for edge in decoded.edges] == sorted(
            (min(left, right), max(left, right)) for left, right in graph.edges
        )
        assert decoded.degrees == tuple(graph.degree(vertex) for vertex in range(order))
