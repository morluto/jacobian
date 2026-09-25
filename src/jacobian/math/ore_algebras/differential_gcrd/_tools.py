"""Catalog declaration for the bounded differential GCRD operation."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.differential_gcrd._models import (
    DifferentialOperatorGCRDRequest,
    DifferentialOperatorGCRDResult,
)
from jacobian.math.ore_algebras.differential_gcrd.operations import (
    differential_operator_gcrd,
)


def _coefficient(value: int) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": {
            "terms": [
                {"coefficient": {"num": str(value), "den": "1"}, "exponents": [0]}
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
        },
    }


def _operator(constant: int) -> dict[str, Any]:
    return {
        "variable": "x",
        "terms": [
            {"order": 0, "coefficient": _coefficient(constant)},
            {"order": 1, "coefficient": _coefficient(1)},
        ],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="ore.operator.gcrd.compute",
        title="Compute a first-order differential GCRD",
        description=(
            "Compute the monic greatest common right divisor of differential "
            "operators in QQ(x)<D> when both inputs have order at most one and "
            "rational constant coefficients. Returns exact left cofactors and "
            "a left Bezout relation. A nonzero Bezout identity for one proves "
            "the coprime result; a shared normalized first-order operator is "
            "the GCRD. Coefficients may be rational constants, represented on "
            "the existing QQ(x) axis. Input rational scalars are limited to "
            "20 decimal digits. This bounded contract does not claim higher "
            "order or variable-coefficient GCRDs."
        ),
        request_type=DifferentialOperatorGCRDRequest,
        result_type=DifferentialOperatorGCRDResult,
        run=lambda request: differential_operator_gcrd(request.left, request.right),
        tags=("ore-algebra", "differential-operator", "gcrd", "exact"),
        discovery_terms=(
            "greatest common right divisor of differential operators",
            "first-order differential GCRD",
            "Ore operator right gcd",
            "Bezout identity for differential operators",
        ),
        examples=(
            OperationExample(
                name="coprime_first_order_operators",
                description=(
                    "Compute the GCRD of D+1 and D-1. Their exact left Bezout "
                    "coefficients are 1/2 and -1/2, so the GCRD is one."
                ),
                input={"left": _operator(1), "right": _operator(-1)},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
