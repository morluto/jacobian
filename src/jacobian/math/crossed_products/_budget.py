"""Mathematical admission bounds for finite-coset crossed-product products."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.crossed_products.values import (
    MAX_EXPONENT_DIGITS,
    FiniteCosetCrossedProductElement,
)

MAX_CONVOLUTION_PAIRS = 1_024
MAX_MULTIPLICATION_SCALAR_WORK = 80_000
MAX_COEFFICIENT_INTERMEDIATE_DIGITS = 19


def _reject(*, location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"crossed_product.{code}",
        message=message,
    )


def _maximum_absolute(values: tuple[int, ...]) -> int:
    return max((abs(value) for value in values), default=0)


def require_multiplication_budget(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
) -> None:
    """Preflight all work, intermediate, support, exponent, and output bounds."""

    if left.presentation != right.presentation:
        _reject(
            location=("left", "right"),
            code="presentation_mismatch",
            message="crossed-product operands must have the same presentation",
        )

    pair_count = len(left.terms) * len(right.terms)
    if pair_count > MAX_CONVOLUTION_PAIRS:
        _reject(
            location=("left", "right"),
            code="convolution_work_bound",
            message=(
                f"operand supports exceed the {MAX_CONVOLUTION_PAIRS}-pair "
                "sparse convolution budget"
            ),
        )

    dimension = left.presentation.lattice_rank
    scalar_work = pair_count * (dimension**2 + 3 * dimension + 1)
    if scalar_work > MAX_MULTIPLICATION_SCALAR_WORK:
        _reject(
            location=("left", "right"),
            code="scalar_work_bound",
            message="crossed-product multiplication exceeds its scalar-work budget",
        )

    characteristic = left.presentation.characteristic
    coefficient_intermediate = (characteristic - 1) ** 2 + characteristic - 1
    if coefficient_intermediate >= 10**MAX_COEFFICIENT_INTERMEDIATE_DIGITS:
        _reject(
            location=("left", "right"),
            code="coefficient_growth_bound",
            message="coefficient accumulation exceeds its exact-integer bound",
        )

    if pair_count:
        left_height = _maximum_absolute(
            tuple(exponent for term in left.terms for exponent in term.exponents)
        )
        right_height = _maximum_absolute(
            tuple(exponent for term in right.terms for exponent in term.exponents)
        )
        action_height = _maximum_absolute(
            tuple(
                entry
                for matrix in left.presentation.action_matrices
                for row in matrix
                for entry in row
            )
        )
        cocycle_height = _maximum_absolute(
            tuple(
                entry
                for row in left.presentation.cocycle_table
                for vector in row
                for entry in vector
            )
        )
        # For every output coordinate,
        # |u + rho(q)v + c(q,r)| <= U + d*A*V + C.
        exponent_height = (
            left_height + dimension * action_height * right_height + cocycle_height
        )
        if exponent_height >= 10**MAX_EXPONENT_DIGITS:
            _reject(
                location=("left", "right"),
                code="exponent_growth_bound",
                message=(
                    f"predicted product exponents exceed the {MAX_EXPONENT_DIGITS}-digit "
                    "carrier bound"
                ),
            )


__all__ = ["MAX_CONVOLUTION_PAIRS", "require_multiplication_budget"]
