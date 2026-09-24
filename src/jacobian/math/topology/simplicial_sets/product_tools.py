"""Public declaration for the finite simplicial-set product."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.product import simplicial_set_product
from jacobian.math.topology.simplicial_sets.product_models import (
    SimplicialSetProductRequest,
    SimplicialSetProductResult,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

_DELTA_ONE = standard_simplex(1, 1).model_dump(mode="json")
_PRODUCT_INPUT = {"left": _DELTA_ONE, "right": _DELTA_ONE}


def _product(request: SimplicialSetProductRequest) -> SimplicialSetProductResult:
    return simplicial_set_product(request)


TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.product.compute",
        title="Construct a finite simplicial-set product",
        description=(
            "Construct the degreewise Cartesian product of two finite truncated "
            "simplicial sets with componentwise face and degeneracy maps. Return "
            "the exact left-major simplex-pair axes and both projection maps. "
            "Factors must have the same maximum degree; product simplex counts, "
            "map rows, identity accounting, and serialized output are admitted "
            "before expansion."
        ),
        request_type=SimplicialSetProductRequest,
        result_type=SimplicialSetProductResult,
        run=_product,
        tags=("topology", "simplicial-set", "product", "exact"),
        discovery_terms=(
            "simplicial set product",
            "Cartesian product of simplicial sets",
            "product simplicial set",
        ),
        examples=(
            OperationExample(
                name="delta_one_product",
                description=(
                    "Build Delta[1] x Delta[1] through degree 1. The degree-zero "
                    "axis is the four ordered pairs of vertices, and the result "
                    "retains both simplicial projection maps."
                ),
                input=_PRODUCT_INPUT,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
