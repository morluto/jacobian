"""Algebraic number arithmetic operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.algebraic_numbers._models import (
    _MAX_RESULT_DIGITS,
    AlgebraicAdditionRequest,
    AlgebraicMultiplicationRequest,
)
from jacobian.math.number_theory.algebraic_numbers._radix_prefix import (
    RadixPrefixRequest,
    RadixPrefixResult,
    radix_prefix,
)
from jacobian.math.number_theory.algebraic_numbers.operations import (
    add_quadratic,
    multiply_quadratic,
)
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue


def compute_algebraic_add(
    request: AlgebraicAdditionRequest,
) -> RealQuadraticValue:
    return add_quadratic(request.left, request.right)


def compute_algebraic_multiply(
    request: AlgebraicMultiplicationRequest,
) -> RealQuadraticValue:
    return multiply_quadratic(request.left, request.right)


def compute_radix_prefix(request: RadixPrefixRequest) -> RadixPrefixResult:
    return radix_prefix(request.value, request.base, request.fractional_places)


def _element(a_num: int, b_num: int, d: int) -> dict[str, object]:
    return {
        "rational_part": {"num": str(a_num), "den": "1"},
        "radical_coefficient": {"num": str(b_num), "den": "1"},
        "radicand": d,
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="algebraic_number.radix_prefix.compute",
        title="Compute an exact radix prefix of a real algebraic number",
        description=(
            "Return the exact base-b prefix of one canonical real algebraic "
            "value: its integer part and exactly the requested number of "
            "base-b fractional digits. Rational values use the canonical "
            "terminating-zero convention; irrational values are compared "
            "exactly against a scaled defining polynomial, never by binary "
            "floating point. Source growth, root-isolation precision, exact "
            "arithmetic, and exact result allocation are admitted before "
            "backend expansion. The result asserts only the requested finite "
            "prefix."
        ),
        request_type=RadixPrefixRequest,
        result_type=RadixPrefixResult,
        run=compute_radix_prefix,
        tags=("algebraic-number", "radix", "exact"),
        examples=(
            OperationExample(
                name="sqrt_two_decimal",
                description="The first twenty decimal places of sqrt(2).",
                input={
                    "value": {
                        "polynomial": ["1", "0", "-2"],
                        "real_root_index": 1,
                    },
                    "base": 10,
                    "fractional_places": 20,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_number.add.compute",
        title="Add two elements of Q(sqrt(d))",
        description="Compute the exact sum of two elements in the quadratic "
        "number field Q(sqrt(d)). Each element is represented as "
        "a + b*sqrt(d) with rational coefficients a and b. Both "
        "operands must use the same square-free radicand d, and the "
        f"component-wise sum must fit within the {_MAX_RESULT_DIGITS}-digit canonical "
        "rational bound; requests whose sum would exceed that bound "
        "are rejected.",
        request_type=AlgebraicAdditionRequest,
        result_type=RealQuadraticValue,
        run=compute_algebraic_add,
        tags=("algebraic-number", "quadratic-field", "exact"),
        examples=(
            OperationExample(
                name="add_sqrt2",
                description="Compute (1 + sqrt(2)) + (3 + 2*sqrt(2)) in Q(sqrt(2)). "
                "Both operands must share one square-free radicand, and "
                f"the component-wise sum must stay within the {_MAX_RESULT_DIGITS}-digit "
                "canonical rational bound.",
                input={
                    "left": _element(1, 1, 2),
                    "right": _element(3, 2, 2),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_number.multiply.compute",
        title="Multiply two elements of Q(sqrt(d))",
        description="Compute the exact product of two elements in the quadratic "
        "number field Q(sqrt(d)). Each element is represented as "
        "a + b*sqrt(d) with rational coefficients a and b. Both "
        "operands must use the same square-free radicand d, and the "
        "exact product components (ac + b*e*d and a*e + b*c) must fit "
        f"within the {_MAX_RESULT_DIGITS}-digit canonical rational bound; requests whose "
        "product would exceed that bound are rejected.",
        request_type=AlgebraicMultiplicationRequest,
        result_type=RealQuadraticValue,
        run=compute_algebraic_multiply,
        tags=("algebraic-number", "quadratic-field", "exact"),
        examples=(
            OperationExample(
                name="multiply_sqrt2",
                description="Compute (1 + sqrt(2)) * (1 - sqrt(2)) = -1 in Q(sqrt(2)). "
                "Both operands must share one square-free radicand, and "
                "the exact product components must stay within the "
                f"{_MAX_RESULT_DIGITS}-digit canonical rational bound.",
                input={
                    "left": _element(1, 1, 2),
                    "right": _element(1, -1, 2),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
