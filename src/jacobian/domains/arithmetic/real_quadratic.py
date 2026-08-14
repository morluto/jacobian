"""Typed real-quadratic order operation and checker declaration."""

from jacobian.domains._examples import example
from jacobian.domains.arithmetic._support import arithmetic_operation
from jacobian.math.real_quadratic import (
    RealQuadraticOrderRequest,
    RealQuadraticOrderValue,
    real_quadratic_order,
)

REAL_QUADRATIC_OPERATIONS = (
    arithmetic_operation(
        "arithmetic.real_quadratic.order.compute",
        "Compare exact real quadratic values",
        (
            "Compare two bounded values a+b*sqrt(d) in one shared real quadratic "
            "field, returning their exact difference and squared-magnitude sign data."
        ),
        RealQuadraticOrderRequest,
        RealQuadraticOrderValue,
        real_quadratic_order,
        "arithmetic",
        "real-quadratic",
        "quadratic-surd",
        "exact-order",
        examples=(
            example(
                "pang_m4_scalar_gap",
                "Compare 3*sqrt(3)/8 with 1/2+sqrt(3)/20 exactly.",
                {
                    "left": {
                        "rational_part": {"num": "0", "den": "1"},
                        "radical_coefficient": {"num": "3", "den": "8"},
                        "radicand": 3,
                    },
                    "right": {
                        "rational_part": {"num": "1", "den": "2"},
                        "radical_coefficient": {"num": "1", "den": "20"},
                        "radicand": 3,
                    },
                },
            ),
        ),
    ),
)

__all__ = ["REAL_QUADRATIC_OPERATIONS"]
