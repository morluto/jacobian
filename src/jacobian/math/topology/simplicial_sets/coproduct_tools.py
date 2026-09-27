"""Public declaration for finite simplicial-set coproducts."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.coproduct import simplicial_set_coproduct
from jacobian.math.topology.simplicial_sets.coproduct_models import (
    SimplicialSetCoproductRequest,
    SimplicialSetCoproductResult,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

_DELTA_ONE = standard_simplex(1, 1).model_dump(mode="json")
_COPRODUCT_INPUT = {"left": _DELTA_ONE, "right": _DELTA_ONE}


def _coproduct(
    request: SimplicialSetCoproductRequest,
) -> SimplicialSetCoproductResult:
    return simplicial_set_coproduct(request)


TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.coproduct.compute",
        title="Construct a finite simplicial-set coproduct",
        description=(
            "Construct the degreewise disjoint union of two finite truncated "
            "simplicial sets with factor-tagged simplex labels and componentwise "
            "face and degeneracy maps. Return the exact tagged axes and typed "
            "factor inclusions. Factors must have the same maximum degree; "
            "aggregate simplex counts, map rows, and serialized output are "
            "admitted before expansion."
        ),
        request_type=SimplicialSetCoproductRequest,
        result_type=SimplicialSetCoproductResult,
        run=_coproduct,
        tags=("topology", "simplicial-set", "coproduct", "exact"),
        discovery_terms=(
            "simplicial set coproduct",
            "disjoint union of simplicial sets",
            "coproduct of simplicial sets",
        ),
        examples=(
            OperationExample(
                name="delta_one_coproduct",
                description=(
                    "Build Delta[1] disjoint-union Delta[1] through degree 1. "
                    "Every simplex label records whether it belongs to the "
                    "left or right factor, and both inclusions are returned."
                ),
                input=_COPRODUCT_INPUT,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
