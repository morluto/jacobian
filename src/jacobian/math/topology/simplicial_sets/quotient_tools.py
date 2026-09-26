"""Public operation for finite simplicial-set congruence quotients."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.quotient import simplicial_set_quotient
from jacobian.math.topology.simplicial_sets.quotient_models import (
    SimplicialSetQuotientRequest,
    SimplicialSetQuotientResult,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

_DELTA_ONE = standard_simplex(1, 2).model_dump(mode="json")
_POINT_QUOTIENT_CLASSES = [[0, 0], [0, 0, 0], [0, 0, 0, 0]]

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.quotient_by_congruence.compute",
        title="Quotient a finite simplicial set by a congruence",
        description=(
            "Form the degreewise quotient of a finite simplicial-set prefix. "
            "Equal class IDs define each degree's equivalence relation; every "
            "visible face and degeneracy must carry equivalent simplices to "
            "equivalent simplices. Return the checked quotient prefix and its "
            "surjective simplicial projection map."
        ),
        request_type=SimplicialSetQuotientRequest,
        result_type=SimplicialSetQuotientResult,
        run=simplicial_set_quotient,
        tags=("topology", "simplicial-set", "quotient", "exact"),
        discovery_terms=(
            "simplicial set quotient",
            "quotient by simplicial congruence",
            "degreewise congruence quotient",
        ),
        examples=(
            OperationExample(
                name="collapse_delta_one_to_point",
                description=(
                    "Identify all simplices degreewise in a finite Delta[1] "
                    "prefix; the quotient is the one-point simplicial set."
                ),
                input={
                    "simplicial_set": _DELTA_ONE,
                    "degree_class_ids": _POINT_QUOTIENT_CLASSES,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
