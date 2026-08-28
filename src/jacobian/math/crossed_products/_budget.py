"""Mathematical admission bounds for finite-coset crossed-product products."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH
from jacobian.math.crossed_products.values import (
    MAX_EXPONENT_DIGITS,
    MAX_PRESENTATION_INTEGER_DIGITS,
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductPresentation,
)

MAX_CONVOLUTION_PAIRS = 1_024
MAX_MULTIPLICATION_SCALAR_WORK = 80_000
MAX_COEFFICIENT_INTERMEDIATE_DIGITS = 19
MAX_SERIALIZED_RESULT_BYTES = 10 * 1024 * 1024


def _reject(*, location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"crossed_product.{code}",
        message=message,
    )


def _maximum_absolute(values: tuple[int, ...]) -> int:
    return max((abs(value) for value in values), default=0)


def _presentation_serialized_upper_bound(
    presentation: FiniteCosetCrossedProductPresentation,
) -> int:
    """Conservatively bound JSON bytes for one retained presentation."""

    coset_count = len(presentation.cosets)
    dimension = presentation.lattice_rank
    label_bytes = 2 + 12 * MAX_OPAQUE_LABEL_LENGTH
    presentation_integer_bytes = 3 + MAX_PRESENTATION_INTEGER_DIGITS
    return (
        4_096
        + (coset_count + dimension + 1) * label_bytes
        + coset_count**2 * label_bytes
        + coset_count * dimension**2 * presentation_integer_bytes
        + coset_count**2 * dimension * presentation_integer_bytes
    )


def _term_serialized_upper_bound(dimension: int) -> int:
    label_bytes = 2 + 12 * MAX_OPAQUE_LABEL_LENGTH
    exponent_bytes = 3 + MAX_EXPONENT_DIGITS
    return 128 + label_bytes + 10 + dimension * exponent_bytes


def _result_serialized_upper_bound(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
    predicted_product_terms: int,
) -> int:
    presentation_bytes = _presentation_serialized_upper_bound(left.presentation)
    term_bytes = _term_serialized_upper_bound(left.presentation.lattice_rank)
    return (
        4_096
        + 3 * presentation_bytes
        + (len(left.terms) + len(right.terms) + predicted_product_terms) * term_bytes
    )


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
    if len(str(coefficient_intermediate)) > MAX_COEFFICIENT_INTERMEDIATE_DIGITS:
        _reject(
            location=("left", "right"),
            code="coefficient_growth_bound",
            message="coefficient accumulation exceeds its exact-integer bound",
        )

    if pair_count:
        left_height = _maximum_absolute(
            tuple(
                parse_canonical_integer(exponent)
                for term in left.terms
                for exponent in term.exponents
            )
        )
        right_height = _maximum_absolute(
            tuple(
                parse_canonical_integer(exponent)
                for term in right.terms
                for exponent in term.exponents
            )
        )
        action_height = _maximum_absolute(
            tuple(
                parse_canonical_integer(entry)
                for matrix in left.presentation.action_matrices
                for row in matrix
                for entry in row
            )
        )
        cocycle_height = _maximum_absolute(
            tuple(
                parse_canonical_integer(entry)
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
        if len(str(exponent_height)) > MAX_EXPONENT_DIGITS:
            _reject(
                location=("left", "right"),
                code="exponent_growth_bound",
                message=(
                    f"predicted product exponents exceed the {MAX_EXPONENT_DIGITS}-digit "
                    "carrier bound"
                ),
            )

    # Every input pair contributes to at most one support key, so pair_count is
    # the exact a-priori product-support bound. This includes all three retained
    # presentations and every source/product term.
    serialized_bound = _result_serialized_upper_bound(left, right, pair_count)
    if serialized_bound > MAX_SERIALIZED_RESULT_BYTES:
        _reject(
            location=("left", "right"),
            code="result_size_bound",
            message=(
                "predicted source-bound result exceeds the "
                f"{MAX_SERIALIZED_RESULT_BYTES}-byte output bound"
            ),
        )


__all__ = ["MAX_CONVOLUTION_PAIRS", "require_multiplication_budget"]
