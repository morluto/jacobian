"""One-shot VF2 worker for the bounded graph-isomorphism adapter."""

from __future__ import annotations

import json
import sys
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism as nx_isomorphism


def _graph(payload: dict[str, Any]) -> nx.Graph[int] | nx.DiGraph[int]:
    graph: nx.Graph[int] | nx.DiGraph[int]
    graph = nx.DiGraph() if payload["directed"] else nx.Graph()
    graph.add_nodes_from(range(payload["vertex_count"]))
    graph.add_edges_from(payload["edges"])
    return graph


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        graph_a = _graph(payload["graph_a"])
        graph_b = _graph(payload["graph_b"])
        matcher: Any
        if payload["graph_a"]["directed"]:
            matcher = nx_isomorphism.DiGraphMatcher(graph_a, graph_b)
        else:
            matcher = nx_isomorphism.GraphMatcher(graph_a, graph_b)
        if not matcher.is_isomorphic():
            response: dict[str, Any] = {"ok": True, "mapping": None}
        else:
            response = {
                "ok": True,
                "mapping": sorted(next(matcher.isomorphisms_iter()).items()),
            }
    except Exception as exc:
        response = {"ok": False, "error": type(exc).__name__}
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
