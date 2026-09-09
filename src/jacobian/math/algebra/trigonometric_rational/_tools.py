"""Public trigonometric-rational normalization declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.algebra.trigonometric_rational.operations import (
    TrigonometricRationalNormalizeRequest,
    TrigonometricRationalNormalizeResult,
    normalize_trigonometric_rational,
)

TOOLS = (
    MathTool(
        operation_id="algebra.trigonometric_rational.normalize",
        title="Normalize a rational trigonometric expression",
        description=(
            "Normalize a bounded typed expression in sines and cosines of integer affine angle forms "
            "to a canonical exact QQ(i) Laurent numerator and denominator, retaining its denominator-nonzero locus."
        ),
        request_type=TrigonometricRationalNormalizeRequest,
        result_type=TrigonometricRationalNormalizeResult,
        run=normalize_trigonometric_rational,
        tags=(
            "algebra",
            "trigonometric",
            "laurent",
            "gaussian-rational",
            "normalize",
            "exact",
        ),
        examples=(
            OperationExample(
                name="pythagorean_identity",
                description="Normalize sin(x)^2 + cos(x)^2 to one.",
                input={
                    "variables": ["x"],
                    "expression": {
                        "kind": "ADD",
                        "children": [
                            {
                                "kind": "POWER",
                                "base": {
                                    "kind": "SINE",
                                    "angle": {"coefficients": [1]},
                                },
                                "exponent": 2,
                            },
                            {
                                "kind": "POWER",
                                "base": {
                                    "kind": "COSINE",
                                    "angle": {"coefficients": [1]},
                                },
                                "exponent": 2,
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)
