"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

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
    """Return the exact color-preserving canonical form and one relabeling.

    Owner-local admission keeps the same typed outcome as the wire path:
    over-bound graphs raise the public ``ValidationError``, not a core-level
    ``PydanticCustomError``, and no wire request is constructed.
    """

    try:
        require_admitted_colored_graph_canonicalization(graph)
    except PydanticCustomError as error:
        raise ValidationError.from_exception_data(
            title="canonicalize_colored_graph",
            line_errors=[{"type": error, "input": graph}],
        ) from error
    return canonicalize_colored_graph_kernel(graph)


__all__ = ["canonicalize_colored_graph"]
