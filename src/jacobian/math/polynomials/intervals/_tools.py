"""Public declaration for rational-polynomial box enclosure."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.polynomials.intervals._models import (
    BOX_ENCLOSURE_ADMISSION_SUMMARY,
    PolynomialBoxEnclosureRequest,
    PolynomialBoxEnclosureResult,
)
from jacobian.math.polynomials.intervals.operations import polynomial_box_enclosure


def compute_polynomial_box_enclosure(
    request: PolynomialBoxEnclosureRequest,
) -> PolynomialBoxEnclosureResult:
    return PolynomialBoxEnclosureResult._from_kernel(
        request,
        enclosure=polynomial_box_enclosure(request.polynomial, request.box),
    )


def _rational(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def _term(
    coefficient: int,
    exponents: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "coefficient": _rational(coefficient),
        "exponents": list(exponents),
    }


TOOLS: MathTools = (
    MathTool(
        operation_id="polynomial.box.enclosure.compute",
        title="Enclose a rational polynomial on a rational box",
        description=(
            "Deterministic exact rational enclosure of one sparse scalar QQ polynomial "
            "on a complete axis-aligned box. It contains the full image, need not be "
            "the exact range, and containing zero does not prove a root. "
            f"{BOX_ENCLOSURE_ADMISSION_SUMMARY}"
        ),
        request_type=PolynomialBoxEnclosureRequest,
        result_type=PolynomialBoxEnclosureResult,
        run=compute_polynomial_box_enclosure,
        tags=(
            "polynomial",
            "multivariate",
            "sparse",
            "evaluation",
            "interval",
            "box",
            "axis-aligned",
            "enclosure",
            "exact",
            "rational",
            "uniform",
            "nonzero",
            "jacobian",
            "bounded",
        ),
        examples=(
            example(
                "positive_bivariate_box",
                "Enclose x^2 + y + 1 on [1,2] x [0,1]; the box must use "
                "the polynomial's complete ordered (x, y) axis.",
                {
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x", "y"],
                        "polynomial": {
                            "terms": [
                                _term(1, (2, 0)),
                                _term(1, (0, 1)),
                                _term(1, (0, 0)),
                            ]
                        },
                    },
                    "box": {
                        "domain": "QQ",
                        "variables": ["x", "y"],
                        "intervals": [
                            {"lower": _rational(1), "upper": _rational(2)},
                            {"lower": _rational(0), "upper": _rational(1)},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
