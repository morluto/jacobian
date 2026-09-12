"""Publication of the polynomial differential-form wedge operation."""

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.differential_forms.operations import wedge
from jacobian.math.polynomials.differential_forms.values import (
    PolynomialDifferentialForm,
)


class WedgeRequest(StrictModel):
    left: PolynomialDifferentialForm
    right: PolynomialDifferentialForm


TOOLS = (
    MathTool(
        operation_id="differential_form.wedge.compute",
        title="Compute the exact wedge product of polynomial differential forms",
        description=(
            "Compute the sparse polynomial form alpha wedge beta using the "
            "increasing differential-index basis and exact permutation signs. "
            "Both forms must use the same ordered variable axis; repeated "
            "differentials vanish and a degree above the ambient dimension "
            "returns the canonical zero form."
        ),
        request_type=WedgeRequest,
        result_type=PolynomialDifferentialForm,
        run=lambda request: wedge(request.left, request.right),
        tags=("polynomial", "differential-form", "wedge", "exterior-algebra", "exact"),
        examples=(
            OperationExample(
                name="dx_wedge_dy",
                description=(
                    "Compute (x dx) wedge (y dy) = xy dx wedge dy; both forms "
                    "must use the same ordered QQ variable axis."
                ),
                input={
                    "left": {
                        "variables": ["x", "y"],
                        "degree": 1,
                        "components": [
                            {
                                "indices": [0],
                                "coefficient": {
                                    "variables": ["x", "y"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [1, 0],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                    },
                    "right": {
                        "variables": ["x", "y"],
                        "degree": 1,
                        "components": [
                            {
                                "indices": [1],
                                "coefficient": {
                                    "variables": ["x", "y"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0, 1],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
