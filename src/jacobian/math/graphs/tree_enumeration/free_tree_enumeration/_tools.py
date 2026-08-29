"""Typed declarations for the free-tree enumeration operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.tree_enumeration.free_tree_enumeration._models import (
    FreeTreeEnumerationRequest,
    FreeTreeEnumerationResult,
)
from jacobian.math.graphs.tree_enumeration.free_tree_enumeration.operations import (
    enumerate_free_trees,
)


def _enumerate(request: FreeTreeEnumerationRequest) -> FreeTreeEnumerationResult:
    return enumerate_free_trees(request.order)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.tree.free.enumerate",
        title="Enumerate non-isomorphic free trees of a given order",
        description=(
            "For one admitted nonnegative order n, return a finite canonical "
            "family containing exactly one SimpleUndirectedGraph representative "
            "of every isomorphism class of free trees on n vertices, together "
            "with its exact cardinality."
        ),
        request_type=FreeTreeEnumerationRequest,
        result_type=FreeTreeEnumerationResult,
        run=_enumerate,
        tags=("graph", "tree", "enumeration", "exact"),
        examples=(
            example(
                "order_4",
                "Enumerate the two non-isomorphic trees on 4 vertices.",
                {"order": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
