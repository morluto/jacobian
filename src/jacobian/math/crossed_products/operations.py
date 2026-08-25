"""Exact multiplication in finite-coset crossed products."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH
from jacobian.math.crossed_products.values import (
    MAX_EXPONENT_DIGITS,
    MAX_PRESENTATION_INTEGER_DIGITS,
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductPresentation,
    FiniteCosetCrossedProductTerm,
    _integer_matrix_vector_product,
)

MAX_CONVOLUTION_PAIRS = 1_024
MAX_MULTIPLICATION_SCALAR_WORK = 80_000
MAX_COEFFICIENT_INTERMEDIATE_DIGITS = 19
MAX_SERIALIZED_RESULT_BYTES = 10 * 1024 * 1024


def _integer_actions(
    presentation: FiniteCosetCrossedProductPresentation,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(tuple(parse_canonical_integer(entry) for entry in row) for row in matrix)
        for matrix in presentation.action_matrices
    )


def _integer_cocycle(
    presentation: FiniteCosetCrossedProductPresentation,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(parse_canonical_integer(entry) for entry in vector) for vector in row
        )
        for row in presentation.cocycle_table
    )


def _maximum_absolute(values: tuple[int, ...]) -> int:
    return max((abs(value) for value in values), default=0)


def _presentation_serialized_upper_bound(
    presentation: FiniteCosetCrossedProductPresentation,
) -> int:
    """Conservatively bound JSON bytes for one retained presentation.

    A Unicode label character may occupy two escaped UTF-16 surrogate chunks,
    hence the deliberately loose 12-byte-per-character bound.  Integer values
    are quoted canonical decimal strings.
    """

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
    # Includes field names, delimiters, and a ten-digit prime-field residue.
    return 128 + label_bytes + 10 + dimension * exponent_bytes


def _result_serialized_upper_bound(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
    predicted_product_terms: int,
) -> int:
    presentation = left.presentation
    presentation_bytes = _presentation_serialized_upper_bound(presentation)
    term_bytes = _term_serialized_upper_bound(presentation.lattice_rank)
    return (
        4_096
        + 3 * presentation_bytes
        + (len(left.terms) + len(right.terms) + predicted_product_terms) * term_bytes
    )


def _require_multiplication_budget(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
) -> None:
    """Preflight all work, intermediate, support, exponent, and output bounds."""

    if left.presentation != right.presentation:
        raise ValueError("crossed-product operands must have the same presentation")

    pair_count = len(left.terms) * len(right.terms)
    if pair_count > MAX_CONVOLUTION_PAIRS:
        raise ValueError(
            "operand supports exceed the 1024-pair sparse convolution budget"
        )

    dimension = left.presentation.lattice_rank
    scalar_work = pair_count * (dimension**2 + 3 * dimension + 1)
    if scalar_work > MAX_MULTIPLICATION_SCALAR_WORK:
        raise ValueError(
            "crossed-product multiplication exceeds its scalar-work budget"
        )

    # Coefficients are reduced after every product accumulation.  The largest
    # Python integer formed is old + left*right < (p-1) + (p-1)^2.
    characteristic = left.presentation.characteristic
    coefficient_intermediate = (characteristic - 1) ** 2 + characteristic - 1
    if len(str(coefficient_intermediate)) > MAX_COEFFICIENT_INTERMEDIATE_DIGITS:
        raise ValueError("coefficient accumulation exceeds its exact-integer bound")

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
        # For every output coordinate:
        # |u + rho(q)v + c(q,r)| <= U + d*A*V + C.
        exponent_height = (
            left_height + dimension * action_height * right_height + cocycle_height
        )
        if len(str(exponent_height)) > MAX_EXPONENT_DIGITS:
            raise ValueError(
                "predicted product exponents exceed the 64-digit carrier bound"
            )

    # Every input pair contributes to at most one support key, so pair_count is
    # also the exact a-priori upper bound on output support.  This bound includes
    # three retained presentations and all source/product terms.
    serialized_bound = _result_serialized_upper_bound(left, right, pair_count)
    if serialized_bound > MAX_SERIALIZED_RESULT_BYTES:
        raise ValueError(
            "predicted source-bound result exceeds the 10 MiB output bound"
        )


def _multiply_admitted(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
) -> FiniteCosetCrossedProductElement:
    presentation = left.presentation
    characteristic = presentation.characteristic
    coset_index = {
        label: position for position, label in enumerate(presentation.cosets)
    }
    actions = _integer_actions(presentation)
    cocycle = _integer_cocycle(presentation)
    coefficients: dict[tuple[int, tuple[int, ...]], int] = {}

    for left_term in left.terms:
        left_coset = coset_index[left_term.coset]
        left_exponents = tuple(
            parse_canonical_integer(entry) for entry in left_term.exponents
        )
        for right_term in right.terms:
            right_coset = coset_index[right_term.coset]
            right_exponents = tuple(
                parse_canonical_integer(entry) for entry in right_term.exponents
            )
            product_coset_label = presentation.quotient_multiplication[left_coset][
                right_coset
            ]
            product_coset = coset_index[product_coset_label]
            transformed = _integer_matrix_vector_product(
                actions[left_coset], right_exponents
            )
            product_exponents = tuple(
                left_coordinate + transformed_coordinate + cocycle_coordinate
                for left_coordinate, transformed_coordinate, cocycle_coordinate in zip(
                    left_exponents,
                    transformed,
                    cocycle[left_coset][right_coset],
                    strict=True,
                )
            )
            key = (product_coset, product_exponents)
            coefficient = (
                coefficients.get(key, 0)
                + left_term.coefficient * right_term.coefficient
            ) % characteristic
            if coefficient:
                coefficients[key] = coefficient
            else:
                coefficients.pop(key, None)

    return FiniteCosetCrossedProductElement(
        presentation=presentation,
        terms=tuple(
            FiniteCosetCrossedProductTerm(
                coefficient=coefficient,
                coset=presentation.cosets[coset],
                exponents=tuple(format_canonical_integer(entry) for entry in exponents),
            )
            for (coset, exponents), coefficient in sorted(coefficients.items())
        ),
    )


def multiply(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
) -> FiniteCosetCrossedProductElement:
    """Multiply two finite-support elements in one explicit crossed product."""

    _require_multiplication_budget(left, right)
    return _multiply_admitted(left, right)


__all__ = ["multiply"]
