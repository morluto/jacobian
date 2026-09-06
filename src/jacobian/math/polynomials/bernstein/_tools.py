"""One canonical change of polynomial basis, not a positivity workflow."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.analysis.intervals import RationalBox
from jacobian.math.polynomials.bernstein.operations import bernstein_coefficients
from jacobian.math.polynomials.bernstein.values import (
    Multidegree,
    RationalBernsteinPolynomial,
)
from jacobian.math.polynomials.values import RationalPolynomial


class BernsteinRequest(StrictModel):
    polynomial: RationalPolynomial
    box: RationalBox
    multidegree: Multidegree = Field(min_length=1, max_length=8)


def _run(request: BernsteinRequest) -> RationalBernsteinPolynomial:
    return bernstein_coefficients(request.polynomial, request.box, request.multidegree)


TOOLS: MathTools = (
    MathTool(
        operation_id="polynomial.bernstein.coefficients.compute",
        title="Convert a rational polynomial to tensor-product Bernstein coefficients",
        description=(
            "Return the exact Bernstein coefficient tensor on a rational box at a declared multidegree, "
            "retaining the source polynomial, ordered axes, and box. Coefficients are in increasing "
            "lexicographic index order, last axis fastest. The box must have positive widths and "
            "the multidegree must dominate coordinate degrees. Coefficient extrema give a convex-hull "
            "enclosure, not necessarily the exact range; a nonpositive coefficient does not disprove positivity. "
            "Admission bounds 65536 tensor entries, 4 million representation characters, 8192-digit "
            "components, and 20 million height-weighted arithmetic steps."
        ),
        request_type=BernsteinRequest,
        result_type=RationalBernsteinPolynomial,
        run=_run,
        tags=(
            "polynomial",
            "Bernstein",
            "basis",
            "tensor-product",
            "rational",
            "box",
            "rectangle",
            "coefficients",
        ),
        examples=(
            OperationExample(
                name="quadratic_unit_interval",
                description="Represent x^2 at degree two on [0,1], giving coefficients (0,0,1).",
                input={
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                }
                            ]
                        },
                    },
                    "box": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "intervals": [
                            {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                            }
                        ],
                    },
                    "multidegree": [2],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
