"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.math.graphs.isomorphism._canonicalization import (
    canonicalize_colored_graph_data,
)
from jacobian.math.graphs.isomorphism._canonicalization_bounds import (
    require_admitted_colored_graph_canonicalization,
)
from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationResult,
    GraphRelabelingPair,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


def _canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Construct the exact canonical value from one admitted graph value."""

    require_admitted_colored_graph_canonicalization(graph)
    canonical_graph, relabeling = canonicalize_colored_graph_data(graph)
    return ColoredGraphCanonicalizationResult._from_kernel(
        source_graph=graph,
        canonical_graph=canonical_graph,
        relabeling=tuple(
            GraphRelabelingPair(source_vertex=source, canonical_vertex=target)
            for source, target in relabeling
        ),
    )


def canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Return the exact color-preserving canonical form and one relabeling.

    Owner-local admission keeps the same typed outcome as the wire path:
    over-bound graphs raise the public ``ValidationError``, not a core-level
    ``PydanticCustomError``, and no wire request is constructed.
    """

    try:
        return _canonicalize_colored_graph(graph)
    except PydanticCustomError as error:
        raise ValidationError.from_exception_data(
            title="canonicalize_colored_graph",
            line_errors=[{"type": error, "input": graph}],
        ) from error


__all__ = ["canonicalize_colored_graph"]
