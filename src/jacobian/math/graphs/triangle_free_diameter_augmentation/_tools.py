"""Typed declaration for triangle-free diameter augmentation."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationRequest,
    TriangleFreeDiameterAugmentationResult,
)


def _run(
    request: TriangleFreeDiameterAugmentationRequest,
) -> TriangleFreeDiameterAugmentationResult:
    from jacobian.math.graphs.triangle_free_diameter_augmentation._augmentation_z3 import (
        solve_triangle_free_diameter_augmentation_values,
    )

    return solve_triangle_free_diameter_augmentation_values(
        request.graph, request.target_diameter, request.resource_budget
    )


TOOLS = (
    MathTool(
        operation_id="graph.triangle_free_diameter_augmentation.minimum",
        title="Minimum triangle-free diameter augmentation",
        description=(
            "Given a connected triangle-free simple graph G and target diameter r>=1, "
            "return the minimum number of missing edges to add while preserving "
            "triangle-freeness and achieving diameter at most r, with one sorted "
            "realizing edge set; infeasible targets return INFEASIBLE and "
            "budget-exhausted requests return SOLVER_BUDGET_EXCEEDED without witness."
        ),
        request_type=TriangleFreeDiameterAugmentationRequest,
        result_type=TriangleFreeDiameterAugmentationResult,
        run=_run,
        tags=(
            "graph",
            "triangle-free",
            "diameter",
            "augmentation",
            "exact",
            "bounded",
            "z3",
        ),
        examples=(
            OperationExample(
                name="path_four_target_two",
                description=(
                    "Augment the path 0-1-2-3 to diameter 2; the unique one-edge solution is (0,3) forming C4, which remains triangle-free."
                ),
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"]],
                    },
                    "target_diameter": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
