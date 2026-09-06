"""Public declaration for exact boxed real-root certification."""

from typing import Any

from jacobian.catalog.models import MathTool, MathTools, OperationExample

from ._models import (
    ROOT_BOX_ADMISSION_SUMMARY,
    PolynomialSystemRootBoxRequest,
    PolynomialSystemRootBoxResult,
)
from .operations import certify_real_root_box, verify_real_root_box


def compute_polynomial_system_root_box(
    request: PolynomialSystemRootBoxRequest,
) -> PolynomialSystemRootBoxResult:
    return certify_real_root_box(request.polynomial_map, request.box)


def _rational(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def _term(coefficient: int, exponent: int) -> dict[str, Any]:
    return {
        "coefficient": _rational(coefficient),
        "exponents": [exponent],
    }


TOOLS: MathTools = (
    MathTool(
        operation_id="polynomial.system.real_root_box.certify",
        title="Certify a real root of a polynomial system in a box",
        description=(
            "Deterministically certify that one complete rational box contains "
            "exactly one nonsingular real zero of a square QQ polynomial system, "
            "prove complete-box exclusion, or return UNKNOWN. Exact midpoint, "
            "preconditioner, point value, interval Jacobian, and Krawczyk-image "
            "data retain the defining evidence; failed inclusion is never a root "
            "conclusion. "
            f"{ROOT_BOX_ADMISSION_SUMMARY}"
        ),
        request_type=PolynomialSystemRootBoxRequest,
        result_type=PolynomialSystemRootBoxResult,
        run=compute_polynomial_system_root_box,
        tags=(
            "polynomial",
            "system",
            "multivariate",
            "real-root",
            "isolating-box",
            "krawczyk",
            "interval-newton",
            "existence",
            "uniqueness",
            "nonsingular",
            "no-root",
            "exact",
            "rational",
            "certificate",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="square_root_two",
                description="Certify the unique positive zero of x^2 - 2 in [1, 2]; "
                "the system must be square and the box must use its complete "
                "ordered axis.",
                input={
                    "polynomial_map": {
                        "input_variables": ["x"],
                        "output_polynomials": [
                            {
                                "domain": "QQ",
                                "variables": ["x"],
                                "polynomial": {"terms": [_term(1, 2), _term(-2, 0)]},
                            }
                        ],
                    },
                    "box": {
                        "domain": "QQ",
                        "variables": ["x"],
                        "intervals": [{"lower": _rational(1), "upper": _rational(2)}],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS", "verify_real_root_box"]
