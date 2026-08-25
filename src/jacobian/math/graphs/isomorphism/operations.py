"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from jacobian.math.graphs.isomorphism._canonicalization_bounds import (
    require_admitted_colored_graph_canonicalization,
)
from jacobian.math.graphs.isomorphism._models import ColoredGraphCanonicalizationResult
from jacobian.math.graphs.isomorphism._operations import (
    canonicalize_colored_graph_kernel,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


def canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Return the exact color-preserving canonical form and one relabeling."""

    require_admitted_colored_graph_canonicalization(graph)
    return canonicalize_colored_graph_kernel(graph)


__all__ = ["canonicalize_colored_graph"]
