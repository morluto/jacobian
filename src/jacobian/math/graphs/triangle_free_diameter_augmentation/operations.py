"""Native Python API for triangle-free diameter augmentation."""

from __future__ import annotations

from jacobian.math.graphs.triangle_free_diameter_augmentation._augmentation_z3 import (
    _require_admitted_request,
    solve_triangle_free_diameter_augmentation_values,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationBudget,
    TriangleFreeDiameterAugmentationResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def triangle_free_diameter_augmentation(
    graph: SimpleUndirectedGraph,
    target_diameter: int,
    *,
    resource_budget: TriangleFreeDiameterAugmentationBudget | None = None,
) -> TriangleFreeDiameterAugmentationResult:
    """Return the bounded minimum triangle-free augmentation for ``graph``.

    Native callers supply the canonical graph value directly. The public
    default envelope is the same conservative order/target envelope
    advertised by the catalog; MCP-specific parsing remains in the private
    wire adapter.
    """

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError(
            "triangle_free_diameter_augmentation expects a SimpleUndirectedGraph"
        )
    if type(target_diameter) is not int:
        raise TypeError("target_diameter must be an int")
    budget = resource_budget or TriangleFreeDiameterAugmentationBudget()
    # Admission is shared; reuse same validation as worker path
    _require_admitted_request(graph, target_diameter, budget)
    return solve_triangle_free_diameter_augmentation_values(
        graph, target_diameter, budget
    )


def _compute_triangle_free_diameter_augmentation(
    request: TriangleFreeDiameterAugmentationBudget,  # placeholder for catalog wrapper
) -> TriangleFreeDiameterAugmentationResult:
    # This wrapper is not used directly; catalog uses request type
    raise NotImplementedError


__all__ = ["triangle_free_diameter_augmentation"]
