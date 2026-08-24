"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationRequest,
    ColoredGraphCanonicalizationResult,
)
from jacobian.math.graphs.isomorphism._operations import (
    compute_colored_graph_canonicalization,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


def canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Return the exact color-preserving canonical form and one relabeling."""

    return compute_colored_graph_canonicalization(
        ColoredGraphCanonicalizationRequest(colored_graph=graph)
    )


__all__ = ["canonicalize_colored_graph"]
