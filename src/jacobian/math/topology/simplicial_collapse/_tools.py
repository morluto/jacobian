"""Public declaration for the elementary simplicial-collapse operation."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_collapse._models import (
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
)
from jacobian.math.topology.simplicial_collapse.operations import elementary_collapse


def _run(request: Any) -> ElementaryCollapseResult:
    return elementary_collapse(request)


TOOLS = (
    MathTool(
        operation_id="topology.simplicial_complex.elementary_collapse.compute",
        title="Perform an elementary simplicial collapse",
        description=(
            "Remove a supplied free codimension-one face and its unique "
            "containing maximal simplex. Return canonical source and target "
            "complexes with the removed pair; a face contained in another "
            "simplex is rejected."
        ),
        request_type=ElementaryCollapseRequest,
        result_type=ElementaryCollapseResult,
        run=_run,
        tags=("topology", "simplicial", "collapse", "exact"),
        examples=(
            OperationExample(
                name="collapse_endpoint_of_edge",
                description="Remove free vertex a and its unique edge ab.",
                input={
                    "complex": {"vertices": ["a", "b"], "facets": [["a", "b"]]},
                    "pair": {"face": ["a"], "coface": ["a", "b"]},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
