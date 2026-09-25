"""Exact bounded commutator of two sparse free-algebra polynomials."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, gcd, log10

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    MAX_FREE_ALGEBRA_WORD_LENGTH,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras.commutator._models import (
    MAX_COMMUTATOR_OUTPUT_WORD_CELLS,
    MAX_COMMUTATOR_WORK,
    FreeAlgebraCommutatorResult,
)


def _reject_resource(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("polynomials",),
        code=f"free_algebra.commutator.{code}",
        message=message,
    )


def _admit_polynomial(
    value: FreeAlgebraPolynomial, *, label: str
) -> FreeAlgebraPolynomial:
    if not isinstance(value, FreeAlgebraPolynomial):
        raise OperationDomainValidationError(
            location=(label,),
            code="free_algebra.commutator.polynomial_type",
            message="commutator operands must be free-algebra polynomial values",
        )
    try:
        return FreeAlgebraPolynomial.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="free_algebra.commutator.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        ) from exc


def _lcm(left: int, right: int) -> int:
    return left // gcd(left, right) * right


def _scaled_coefficients(
    polynomial: FreeAlgebraPolynomial,
) -> tuple[int, tuple[tuple[tuple[str, ...], int], ...]]:
    fractions = tuple(
        (term.word, term.coefficient.as_fraction()) for term in polynomial.terms
    )
    common_denominator = 1
    for _, coefficient in fractions:
        common_denominator = _lcm(common_denominator, coefficient.denominator)
    return common_denominator, tuple(
        (word, coefficient.numerator * (common_denominator // coefficient.denominator))
        for word, coefficient in fractions
    )


def _preflight(
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
) -> tuple[
    tuple[tuple[tuple[str, ...], int], ...],
    tuple[tuple[tuple[str, ...], int], ...],
    int,
    int,
]:
    if left.alphabet != right.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.commutator.alphabet_mismatch",
            message="both polynomials must use the same ordered generator alphabet",
        )
    for side, polynomial in (("left", left), ("right", right)):
        if len(polynomial.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                "operand_term_budget",
                f"{side} operand exceeds the {MAX_FREE_ALGEBRA_OPERAND_TERMS}-term budget",
            )
        if any(
            len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH for term in polynomial.terms
        ):
            _reject_resource(
                "operand_word_length_budget",
                f"{side} operand word exceeds the {MAX_FREE_ALGEBRA_WORD_LENGTH}-letter budget",
            )

    if left == right:
        return (), (), 1, 1

    pair_count = len(left.terms) * len(right.terms)
    # Each pair can contribute to both uv and vu. This input-only upper bound
    # proves result term and word-cell limits before any concatenated words are
    # made. It can be conservative when commutator terms later cancel.
    candidate_terms = 2 * pair_count
    if candidate_terms > MAX_FREE_ALGEBRA_RESULT_TERMS:
        _reject_resource(
            "result_term_budget",
            "the commutator can exceed the 4,096-term result bound",
        )
    max_word_length = max((len(term.word) for term in left.terms), default=0) + max(
        (len(term.word) for term in right.terms), default=0
    )
    if candidate_terms * max_word_length > MAX_COMMUTATOR_OUTPUT_WORD_CELLS:
        _reject_resource(
            "output_word_cells",
            "the commutator can exceed the aggregate output word-cell bound",
        )

    if pair_count == 0:
        return (), (), 1, 1

    left_denominator_sizes = tuple(
        len(str(term.coefficient.as_fraction().denominator)) for term in left.terms
    )
    right_denominator_sizes = tuple(
        len(str(term.coefficient.as_fraction().denominator)) for term in right.terms
    )
    left_denominator_digits = sum(left_denominator_sizes)
    right_denominator_digits = sum(right_denominator_sizes)
    # LCM accumulation cannot exceed the product of these bounded input
    # denominators; reject before constructing that common scaling value.
    if (
        left_denominator_digits + right_denominator_digits
        > 2 * MAX_FREE_ALGEBRA_OPERAND_TERMS * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
    ):
        _reject_resource("denominator_growth", "common denominator bound exceeded")

    # Bound the sequential LCM work before computing either common
    # denominator. At each step the accumulated value has at most the sum of
    # the preceding denominator digits.
    left_lcm_work = sum(
        sum(left_denominator_sizes[: index + 1]) * size
        for index, size in enumerate(left_denominator_sizes)
    )
    right_lcm_work = sum(
        sum(right_denominator_sizes[: index + 1]) * size
        for index, size in enumerate(right_denominator_sizes)
    )
    lcm_work = left_lcm_work + right_lcm_work
    if lcm_work > MAX_COMMUTATOR_WORK:
        _reject_resource(
            "work_bound",
            f"common-denominator work estimate {lcm_work} exceeds {MAX_COMMUTATOR_WORK}",
        )
    left_common, left_scaled = _scaled_coefficients(left)
    right_common, right_scaled = _scaled_coefficients(right)
    left_scaled_digits = max(
        (len(str(abs(value))) for _, value in left_scaled), default=1
    )
    right_scaled_digits = max(
        (len(str(abs(value))) for _, value in right_scaled), default=1
    )
    # With one term pair, distinct uv and vu words each have one contribution;
    # if they coincide, the signed contributions cancel exactly. For larger
    # supports use the simple worst-case bound of two contributions per pair.
    contributions_per_word = 1 if pair_count == 1 else candidate_terms
    numerator_digits = (
        left_scaled_digits
        + right_scaled_digits
        + ceil(log10(contributions_per_word + 1))
    )
    denominator_digits = len(str(left_common)) + len(str(right_common))
    if max(numerator_digits, denominator_digits) > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            "coefficient_growth",
            "predicted exact commutator coefficient exceeds the 64-digit bound",
        )

    # Count decimal digit products as a conservative exact-arithmetic work
    # proxy, plus word copying and sorting. The bound precedes convolution.
    work_bound = (
        left_denominator_digits**2
        + right_denominator_digits**2
        + 2 * pair_count * left_scaled_digits * right_scaled_digits
        + candidate_terms * numerator_digits
        + candidate_terms * max_word_length
        + candidate_terms * max(1, candidate_terms.bit_length()) * max_word_length
    )
    if work_bound > MAX_COMMUTATOR_WORK:
        _reject_resource(
            "work_bound",
            f"commutator work estimate {work_bound} exceeds {MAX_COMMUTATOR_WORK}",
        )
    return left_scaled, right_scaled, left_common, right_common


def _commutator_numerators(
    left: tuple[tuple[tuple[str, ...], int], ...],
    right: tuple[tuple[tuple[str, ...], int], ...],
) -> dict[tuple[str, ...], int]:
    numerators: dict[tuple[str, ...], int] = {}
    for left_word, left_coefficient in left:
        for right_word, right_coefficient in right:
            contribution = left_coefficient * right_coefficient
            forward = left_word + right_word
            reverse = right_word + left_word
            numerators[forward] = numerators.get(forward, 0) + contribution
            numerators[reverse] = numerators.get(reverse, 0) - contribution
    return numerators


def commutator(
    left: FreeAlgebraPolynomial,
    right: FreeAlgebraPolynomial,
) -> FreeAlgebraCommutatorResult:
    """Return the exact associative-algebra commutator ``left*right-right*left``."""

    left = _admit_polynomial(left, label="left")
    right = _admit_polynomial(right, label="right")
    left_scaled, right_scaled, left_denominator, right_denominator = _preflight(
        left, right
    )
    denominator = left_denominator * right_denominator
    values = _commutator_numerators(left_scaled, right_scaled)
    ordered = tuple(
        sorted(
            (
                (word, Fraction(numerator, denominator))
                for word, numerator in values.items()
                if numerator
            ),
            key=lambda item: canonical_word_key(left.alphabet, item[0]),
            reverse=True,
        )
    )
    polynomial = FreeAlgebraPolynomial.model_construct(
        alphabet=left.alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(coefficient),
                word=word,
            )
            for word, coefficient in ordered
        ),
    )
    return FreeAlgebraCommutatorResult.model_construct(
        left=left, right=right, commutator=polynomial
    )


__all__ = ["commutator"]
