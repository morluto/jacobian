"""Operations for polynomials over exact cyclotomic coefficient fields."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.cyclotomic_coefficients._models import (
    CyclotomicPolynomial,
    RationalPolynomialCyclotomicEmbeddingRequest,
)
from jacobian.math.polynomials.cyclotomic_coefficients.operations import (
    embed_rational_polynomial,
)


def _example_request() -> dict[str, Any]:
    return {
        "polynomial": {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "2", "den": "1"}, "exponents": [1]},
                    {"coefficient": {"num": "3", "den": "1"}, "exponents": [0]},
                ]
            },
        },
        "field": {"order": 5},
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.cyclotomic_coefficient.embed.compute",
        title="Embed a rational polynomial in a cyclotomic coefficient field",
        description=(
            "Apply the canonical inclusion QQ -> QQ(zeta_n) to every coefficient "
            "of a sparse rational polynomial. The result preserves the exact "
            "ordered variables and monomial support and uses the shared "
            "RationalCyclotomicField and RationalCyclotomicElement carriers. The "
            "field order is at most 128, each scalar coordinate at most 256 "
            "decimal digits, and the complete output is bounded by 16,384 "
            "coefficient coordinates and 10 MiB. This is an exact coefficient "
            "map; it makes no claim that the polynomial has a particular source "
            "or satisfies an additional identity."
        ),
        request_type=RationalPolynomialCyclotomicEmbeddingRequest,
        result_type=CyclotomicPolynomial,
        run=embed_rational_polynomial,
        tags=("polynomial", "cyclotomic", "coefficient-ring", "exact"),
        discovery_terms=(
            "polynomial over a cyclotomic field",
            "change polynomial coefficient ring",
            "embed rational polynomial",
        ),
        examples=(
            OperationExample(
                name="embed-rational-polynomial-in-q-zeta5",
                description="Embed 2*x + 3 from QQ into QQ(zeta_5), retaining the x axis.",
                input=_example_request(),
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
