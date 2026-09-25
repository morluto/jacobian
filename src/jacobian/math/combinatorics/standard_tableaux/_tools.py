"""Public bounded standard-tableau enumeration operation."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.standard_tableaux._models import (
    StandardTableauEnumerationRequest,
    StandardTableauEnumerationResult,
)
from jacobian.math.combinatorics.standard_tableaux.enumeration import (
    enumerate_standard_young_tableaux,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="combinatorics.standard_young_tableaux.enumerate",
        title="Enumerate standard Young tableaux",
        description=(
            "Return the complete standard Young tableau family of one partition "
            "in canonical lexicographic row order. The full family is admitted "
            "before tableau construction by exact hook-length count, aggregate "
            "cell, and output-size bounds."
        ),
        request_type=StandardTableauEnumerationRequest,
        result_type=StandardTableauEnumerationResult,
        run=lambda request: enumerate_standard_young_tableaux(request.partition),
        tags=("combinatorics", "tableau", "enumeration", "exact"),
        discovery_terms=(
            "enumerate standard Young tableaux",
            "all standard tableaux of a partition shape",
        ),
        examples=(
            OperationExample(
                name="standard_tableaux_of_shape_2_1",
                description=(
                    "Enumerate all standard tableaux of shape (2,1); complete "
                    "output is bounded by 4096 tableaux, 100000 aggregate cells, "
                    "25000000 construction-work cells, and 2000000 estimated bytes."
                ),
                input={"partition": {"parts": [2, 1]}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
