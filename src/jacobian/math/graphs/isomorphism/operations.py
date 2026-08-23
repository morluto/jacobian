"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from jacobian.math.graphs.isomorphism._canonicalization import (
    canonicalize_colored_graph_data,
)
from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationRequest,
    ColoredGraphCanonicalizationResult,
    GraphRelabelingPair,
)
from jacobian.math.graphs.isomorphism.values import ColoredUndirectedGraph


def canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Return the exact color-preserving canonical form and one relabeling."""

    request = ColoredGraphCanonicalizationRequest(colored_graph=graph)
    canonical_graph, relabeling = canonicalize_colored_graph_data(request.colored_graph)
    return ColoredGraphCanonicalizationResult(
        source_graph=request.colored_graph,
        canonical_graph=canonical_graph,
        relabeling=tuple(
            GraphRelabelingPair(
                source_vertex=source,
                canonical_vertex=target,
            )
            for source, target in relabeling
        ),
    )


__all__ = ["canonicalize_colored_graph"]
