"""A complete exact Jacobian on a rational map's common regular locus."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.rational_functions.maps._models import (
    RationalFunctionMapJacobian,
    RationalMapJacobianRequest,
)
from jacobian.math.polynomials.rational_functions.maps.operations import jacobian_matrix


def _compute(request: RationalMapJacobianRequest) -> RationalFunctionMapJacobian:
    return jacobian_matrix(request.source)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="rational_function_map.jacobian.compute",
        title="Compute the Jacobian of a rational coordinate map",
        description=(
            "Return every exact partial derivative of a multivariate rational map. "
            "Rows follow the named target coordinates and columns follow the ordered "
            "source variables. Retain the entire source map and its common "
            "denominator-nonzero locus even when derivative entries cancel to zero "
            "or polynomials. Empty axes give the corresponding empty matrix."
        ),
        request_type=RationalMapJacobianRequest,
        result_type=RationalFunctionMapJacobian,
        run=_compute,
        tags=("rational-function", "map", "jacobian", "derivative", "exact"),
        examples=(
            OperationExample(
                name="reciprocal_coordinate",
                description="The derivative of u=1/x retains the source locus x nonzero.",
                input={
                    "source": {
                        "source_variables": ["x"],
                        "target_coordinates": ["u"],
                        "components": [
                            {
                                "variables": ["x"],
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0],
                                        }
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                },
            ),
        ),
    ),
)
