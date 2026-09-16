"""Koszul construction admission shared by the native kernel and the catalog.

Every expansion quantity — sequence length, wedge basis counts, differential
entry counts, polynomial term/degree/coefficient growth, and the exact
``d^2 = 0`` replay work — is bounded once here, before any basis, matrix, or
product is materialized. The replay is kernel work charged in the same
admission.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.values import (
    MAX_KOSZUL_COEFFICIENT_DIGITS,
    MAX_KOSZUL_DEGREE,
    MAX_KOSZUL_DIFFERENTIAL_ENTRIES,
    MAX_KOSZUL_PRODUCT_TERMS,
    MAX_KOSZUL_REPLAY_TERM_WORK,
    MAX_KOSZUL_SEQUENCE_LENGTH,
    MAX_KOSZUL_TERMS,
    MAX_KOSZUL_TOTAL_BASIS,
    MAX_KOSZUL_VARIABLES,
)
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial

SparsePolynomial = dict[tuple[int, ...], Fraction]


def _resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("sequence",), code=code, message=message
    )


def sparse_terms(polynomial: RationalPolynomial) -> SparsePolynomial:
    """Convert one canonical sparse value into exact term arithmetic data."""

    return {
        term.exponents: Fraction(term.coefficient.num, term.coefficient.den)
        for term in polynomial.polynomial.terms
    }


def _admit_sequence_elements(
    variables: tuple[PolynomialVariable, ...],
    sequence: tuple[RationalPolynomial, ...],
) -> None:
    """Bound one ring binding and every element's term/degree/digit growth."""

    for position, element in enumerate(sequence):
        if element.domain != "QQ" or element.variables != variables:
            raise OperationDomainValidationError(
                location=("sequence", position),
                code="koszul.ring_mismatch",
                message=(
                    "every sequence element must belong to the single "
                    "declared ordered QQ polynomial ring"
                ),
            )
        terms = element.polynomial.terms
        if len(terms) > MAX_KOSZUL_TERMS:
            raise _resource(
                "koszul.term_budget",
                f"sequence element {position} has {len(terms)} terms, "
                f"exceeding the {MAX_KOSZUL_TERMS}-term envelope",
            )
        for term in terms:
            if any(exponent > MAX_KOSZUL_DEGREE for exponent in term.exponents):
                raise _resource(
                    "koszul.degree_budget",
                    f"sequence element {position} exceeds the "
                    f"{MAX_KOSZUL_DEGREE}-degree-per-variable envelope",
                )
            digits = max(
                len(format_canonical_integer(abs(term.coefficient.num))),
                len(format_canonical_integer(term.coefficient.den)),
            )
            if digits > MAX_KOSZUL_COEFFICIENT_DIGITS:
                raise _resource(
                    "koszul.coefficient_digit_budget",
                    f"sequence element {position} has a coefficient beyond "
                    f"the {MAX_KOSZUL_COEFFICIENT_DIGITS}-digit envelope",
                )


def admit_koszul_construction(
    variables: tuple[PolynomialVariable, ...],
    sequence: tuple[RationalPolynomial, ...],
) -> tuple[SparsePolynomial, ...]:
    """Preflight one Koszul construction and prepare exact term data.

    Raises ``OperationDomainValidationError`` for a request that is not one
    sequence over one declared ring, and ``OperationResourceAdmissionError``
    for a structurally valid request outside the published envelope.
    """

    if len(variables) > MAX_KOSZUL_VARIABLES:
        raise _resource(
            "koszul.variable_budget",
            f"the Koszul ambient ring admits at most {MAX_KOSZUL_VARIABLES} "
            f"variables, got {len(variables)}",
        )
    length = len(sequence)
    if length > MAX_KOSZUL_SEQUENCE_LENGTH:
        raise _resource(
            "koszul.sequence_length_budget",
            f"the Koszul construction admits sequences of length at most "
            f"{MAX_KOSZUL_SEQUENCE_LENGTH}, got {length}",
        )
    _admit_sequence_elements(variables, sequence)

    # Derived expansion ceilings, charged before any materialization.
    total_basis = sum(comb(length, degree) for degree in range(length + 1))
    if total_basis > MAX_KOSZUL_TOTAL_BASIS:
        raise _resource(
            "koszul.total_basis_budget",
            f"the complete wedge basis has {total_basis} elements, exceeding "
            f"the {MAX_KOSZUL_TOTAL_BASIS}-element envelope",
        )
    differential_entries = length * 2 ** max(0, length - 1)
    if differential_entries > MAX_KOSZUL_DIFFERENTIAL_ENTRIES:
        raise _resource(
            "koszul.differential_entry_budget",
            f"the differentials can carry {differential_entries} "
            f"contributions, exceeding the "
            f"{MAX_KOSZUL_DIFFERENTIAL_ENTRIES}-entry envelope",
        )

    prepared = tuple(sparse_terms(element) for element in sequence)
    if length >= 2:
        # The exact d^2 = 0 replay forms every unordered pair product once
        # and adds the two signed orderings for every wedge basis element
        # containing the pair; each pair occurs in 2^(c-2) basis elements.
        replay_term_work = 0
        for i in range(length):
            for j in range(i + 1, length):
                pair_terms = max(1, len(prepared[i])) * max(1, len(prepared[j]))
                if pair_terms > MAX_KOSZUL_PRODUCT_TERMS:
                    raise _resource(
                        "koszul.product_term_budget",
                        f"the d^2 replay product of sequence elements {i} "
                        f"and {j} can exceed the "
                        f"{MAX_KOSZUL_PRODUCT_TERMS}-term envelope",
                    )
                # Both signed orderings of every pair product are
                # materialized on each of the 2^(c-2) basis elements.
                replay_term_work += 2 ** (length - 1) * pair_terms
        if replay_term_work > MAX_KOSZUL_REPLAY_TERM_WORK:
            raise _resource(
                "koszul.replay_work_budget",
                f"the exact d^2 = 0 replay can perform {replay_term_work} "
                f"term operations, exceeding the "
                f"{MAX_KOSZUL_REPLAY_TERM_WORK}-operation envelope",
            )
    return prepared


__all__ = ["SparsePolynomial", "admit_koszul_construction", "sparse_terms"]
