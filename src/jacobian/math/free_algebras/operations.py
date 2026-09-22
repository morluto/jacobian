"""Native exact free associative algebra polynomial multiplication.

The single V1 operation establishes one atomic mathematical postcondition:
the exact distributive noncommutative product of two sparse ``QQ``-linear
polynomials over one shared ordered generator alphabet.  All semantic
admission is performed once, here, before the multiplication kernel runs; the
result value is constructed through a trusted factory rather than replaying
the product.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from math import gcd
from typing import Any

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._kernel import multiply_sparse
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_GS_COMPOSITIONS,
    MAX_FREE_ALGEBRA_GS_PAIR_CHECKS,
    MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_BASIS,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_SERIALIZED_BYTES,
    MAX_FREE_ALGEBRA_IDEAL_PREFIX_TOTAL_TERMS,
    MAX_FREE_ALGEBRA_LETTER_LENGTH,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    MAX_FREE_ALGEBRA_TERM_PAIRS,
    MAX_FREE_ALGEBRA_WORD_LENGTH,
    FreeAlgebraIdeal,
    FreeAlgebraIdealPrefixResult,
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialProductResult,
    FreeAlgebraTerm,
    GroebnerShirshovResult,
    canonical_word_key,
)


def _reject_resource(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.{code}",
        message=message,
    )


def _admit_polynomial(
    value: FreeAlgebraPolynomial, *, label: str
) -> FreeAlgebraPolynomial:
    try:
        return FreeAlgebraPolynomial.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="free_algebra.polynomial_shape",
            message="the free-algebra polynomial is not canonical",
        ) from exc


def _admit_product(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> tuple[FreeAlgebraPolynomial, FreeAlgebraPolynomial]:
    """Admit one product request before any word-pair expansion.

    The structural value contracts already established alphabet distinctness,
    canonical support, and per-coefficient digit bounds.  This shared
    admission helper adds the product operation's envelope: alphabet identity,
    operand term and word-length budgets, the term-pair product count, the
    result term count, and coefficient growth.
    """

    left = _admit_polynomial(left, label="left")
    right = _admit_polynomial(right, label="right")
    if left.alphabet != right.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.alphabet_mismatch",
            message=(
                "both operands must be bound to the same ordered generator alphabet"
            ),
        )

    max_operand_digits = 0
    for side, polynomial in (("left", left), ("right", right)):
        if len(polynomial.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                (side, "terms"),
                "operand_term_budget",
                f"{side} operand exceeds the "
                f"{MAX_FREE_ALGEBRA_OPERAND_TERMS}-term multiplication budget",
            )
        for index, term in enumerate(polynomial.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
                _reject_resource(
                    (side, "terms", index, "word"),
                    "operand_word_length_budget",
                    f"{side} operand word exceeds the "
                    f"{MAX_FREE_ALGEBRA_WORD_LENGTH}-letter multiplication "
                    "budget",
                )
            max_operand_digits = max(
                max_operand_digits,
                canonical_rational_component_digits(term.coefficient),
            )

    term_pair_count = len(left.terms) * len(right.terms)
    if term_pair_count > MAX_FREE_ALGEBRA_TERM_PAIRS:
        _reject_resource(
            ("left", "terms"),
            "term_pair_budget",
            "product term pairs exceed the "
            f"{MAX_FREE_ALGEBRA_TERM_PAIRS}-pair multiplication budget",
        )
    # Distinct product words are a subset of the term pairs, so this bounds
    # the result term count before product expansion.
    if term_pair_count > MAX_FREE_ALGEBRA_RESULT_TERMS:
        _reject_resource(
            ("left", "terms"),
            "result_term_budget",
            "product can exceed the "
            f"{MAX_FREE_ALGEBRA_RESULT_TERMS}-term result budget",
        )

    addition_digits = len(str(term_pair_count - 1)) if term_pair_count >= 2 else 0
    predicted_coefficient_digits = 2 * max_operand_digits + addition_digits
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            ("left", "terms"),
            "coefficient_growth_budget",
            "predicted product coefficient growth exceeds the "
            f"{MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS}-digit multiplication budget",
        )
    return left, right


def multiply(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> FreeAlgebraPolynomialProductResult:
    """Multiply two sparse noncommutative polynomials exactly."""

    left, right = _admit_product(left, right)
    product, ledger = multiply_sparse(left, right)
    return FreeAlgebraPolynomialProductResult.model_construct(
        left=left,
        right=right,
        product=product,
        ledger=ledger,
    )


def _map(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _encode(
    alphabet: tuple[str, ...], values: Mapping[tuple[str, ...], Fraction]
) -> FreeAlgebraPolynomial:
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(coefficient), word=word
        )
        for word, coefficient in sorted(
            ((word, value) for word, value in values.items() if value),
            key=lambda item: canonical_word_key(alphabet, item[0]),
            reverse=True,
        )
    )
    return FreeAlgebraPolynomial(alphabet=alphabet, terms=terms)


def _contexts(alphabet: tuple[str, ...], length: int) -> tuple[tuple[str, ...], ...]:
    return tuple(product(alphabet, repeat=length)) if length else ((),)


def _ideal_prefix(
    ideal: FreeAlgebraIdeal, degree: int
) -> tuple[FreeAlgebraPolynomial, ...]:
    values: list[FreeAlgebraPolynomial] = []
    for generator in ideal.generators:
        generator_degree = max((len(term.word) for term in generator.terms), default=0)
        if generator_degree > degree:
            continue
        context_length = degree - generator_degree
        if ideal.side == "two-sided":
            context_pairs = tuple(
                (left_context, right_context)
                for left_length in range(context_length + 1)
                for left_context in _contexts(ideal.alphabet, left_length)
                for right_context in _contexts(
                    ideal.alphabet, context_length - left_length
                )
            )
        elif ideal.side == "left":
            context_pairs = tuple(
                (context, ()) for context in _contexts(ideal.alphabet, context_length)
            )
        else:
            context_pairs = tuple(
                ((), context) for context in _contexts(ideal.alphabet, context_length)
            )
        for left_context, right_context in context_pairs:
            left = FreeAlgebraPolynomial(
                alphabet=ideal.alphabet,
                terms=(
                    FreeAlgebraTerm(
                        coefficient=CanonicalRational.from_fraction(Fraction(1)),
                        word=left_context,
                    ),
                ),
            )
            right = FreeAlgebraPolynomial(
                alphabet=ideal.alphabet,
                terms=(
                    FreeAlgebraTerm(
                        coefficient=CanonicalRational.from_fraction(Fraction(1)),
                        word=right_context,
                    ),
                ),
            )
            if ideal.side == "left":
                values.append(multiply(left, generator).product)
            elif ideal.side == "right":
                values.append(multiply(generator, right).product)
            else:
                values.append(
                    multiply(multiply(left, generator).product, right).product
                )
    return tuple(values)


def _admit_ideal(ideal: FreeAlgebraIdeal) -> FreeAlgebraIdeal:
    """Revalidate typed ideals before selecting a side or expanding them."""
    try:
        return FreeAlgebraIdeal.model_validate(ideal.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="free_algebra.ideal_shape",
            message="the ideal presentation is not canonical",
        ) from exc


def _as_ideal(ideal: FreeAlgebraIdeal | Mapping[str, Any]) -> FreeAlgebraIdeal:
    try:
        parsed = (
            ideal
            if isinstance(ideal, FreeAlgebraIdeal)
            else FreeAlgebraIdeal.model_validate(ideal)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="free_algebra.ideal_shape",
            message="the ideal presentation is not canonical",
        ) from exc
    return _admit_ideal(parsed)


def ideal_generated_prefix(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> FreeAlgebraIdealPrefixResult:
    value = _as_ideal(ideal)
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_degree_bound",
            message="ideal prefix degree exceeds the admitted envelope",
        )
    if len(value.generators) > MAX_FREE_ALGEBRA_OPERAND_TERMS or any(
        len(generator.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS
        or any(
            len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH for term in generator.terms
        )
        for generator in value.generators
    ):
        raise OperationResourceAdmissionError(
            location=("ideal",),
            code="free_algebra.ideal_generator_budget",
            message="ideal generator expansion exceeds the admitted envelope",
        )
    predicted_basis = 0
    predicted_terms = 0
    predicted_bytes = 0
    for generator in value.generators:
        generator_degree = max((len(term.word) for term in generator.terms), default=0)
        if generator_degree > degree:
            continue
        remaining = degree - generator_degree
        contexts = (
            len(value.alphabet) ** remaining
            if value.side != "two-sided"
            else (remaining + 1) * len(value.alphabet) ** remaining
        )
        generator_terms = len(generator.terms)
        predicted_basis += contexts
        predicted_terms += contexts * generator_terms
        # This is intentionally source-derived and conservative: every term
        # carries a coefficient and a word, plus the enclosing basis record.
        term_bytes = (
            2 * MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
            + degree * (MAX_FREE_ALGEBRA_LETTER_LENGTH + 8)
            + 128
        )
        predicted_bytes += contexts * (generator_terms * term_bytes + 128)
        if any(
            canonical_rational_component_digits(term.coefficient)
            > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS
            for term in generator.terms
        ):
            _reject_resource(
                ("ideal", "generators"),
                "ideal_prefix_coefficient_growth",
                "ideal prefix coefficients exceed the aggregate carrier budget",
            )
    if predicted_basis > MAX_FREE_ALGEBRA_IDEAL_PREFIX_BASIS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_basis",
            message="ideal prefix basis cardinality exceeds the admitted envelope",
        )
    if predicted_terms > MAX_FREE_ALGEBRA_IDEAL_PREFIX_TOTAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_terms",
            message="ideal prefix aggregate terms exceed the admitted envelope",
        )
    if predicted_bytes > MAX_FREE_ALGEBRA_IDEAL_PREFIX_SERIALIZED_BYTES:
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.ideal_prefix_serialized_size",
            message="ideal prefix serialized aggregate exceeds the admitted envelope",
        )
    basis = _ideal_prefix(value, degree)
    return FreeAlgebraIdealPrefixResult.model_construct(
        ideal=value, degree=degree, basis=basis
    )


def _leading(value: FreeAlgebraPolynomial) -> tuple[tuple[str, ...], Fraction] | None:
    if not value.terms:
        return None
    term = value.terms[0]
    return term.word, term.coefficient.as_fraction()


_MAX_GS_COEFFICIENT_COMPONENT = 10**MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS - 1


def _component_product_fits(left: int, right: int) -> bool:
    left = abs(left)
    right = abs(right)
    return not left or not right or left <= _MAX_GS_COEFFICIENT_COMPONENT // right


def _require_product_fits(left: Fraction, right: Fraction) -> None:
    if not left or not right:
        return
    left_num = abs(left.numerator)
    right_num = abs(right.numerator)
    left_den = left.denominator
    right_den = right.denominator
    cancel_left = gcd(left_num, right_den)
    cancel_right = gcd(right_num, left_den)
    if not (
        _component_product_fits(left_num // cancel_left, right_num // cancel_right)
        and _component_product_fits(left_den // cancel_right, right_den // cancel_left)
    ):
        _reject_resource(
            ("ideal", "generators"),
            "gs_coefficient_growth_budget",
            "GS coefficient multiplication exceeds the admitted exact-digit envelope",
        )


def _require_difference_fits(left: Fraction, right: Fraction) -> None:
    if left == right:
        return
    common = gcd(left.denominator, right.denominator)
    left_multiplier = right.denominator // common
    right_multiplier = left.denominator // common
    if not (
        _component_product_fits(left.numerator, left_multiplier)
        and _component_product_fits(right.numerator, right_multiplier)
        and _component_product_fits(left.denominator, left_multiplier)
    ):
        _reject_resource(
            ("ideal", "generators"),
            "gs_coefficient_growth_budget",
            "GS coefficient subtraction exceeds the admitted exact-digit envelope",
        )
    left_component = left.numerator * left_multiplier
    right_component = right.numerator * right_multiplier
    if (left_component < 0) != (right_component < 0) and abs(
        left_component
    ) > _MAX_GS_COEFFICIENT_COMPONENT - abs(right_component):
        _reject_resource(
            ("ideal", "generators"),
            "gs_coefficient_growth_budget",
            "GS coefficient subtraction exceeds the admitted exact-digit envelope",
        )


def _admitted_quotient(numerator: Fraction, denominator: Fraction) -> Fraction:
    reciprocal = Fraction(denominator.denominator, denominator.numerator)
    _require_product_fits(numerator, reciprocal)
    return numerator * reciprocal


def _subtract(
    left: Mapping[tuple[str, ...], Fraction],
    right: Mapping[tuple[str, ...], Fraction],
    scale: Fraction = Fraction(1),
) -> dict[tuple[str, ...], Fraction]:
    result = dict(left)
    for word, coefficient in right.items():
        _require_product_fits(scale, coefficient)
        scaled = scale * coefficient
        existing = result.get(word, Fraction(0))
        _require_difference_fits(existing, scaled)
        result[word] = existing - scaled
        if not result[word]:
            del result[word]
    return result


def _multiply_monomial(
    value: FreeAlgebraPolynomial, prefix: tuple[str, ...], suffix: tuple[str, ...]
) -> FreeAlgebraPolynomial:
    return _encode(
        value.alphabet,
        {
            prefix + word + suffix: coefficient
            for word, coefficient in _map(value).items()
        },
    )


def _normal_form(
    value: FreeAlgebraPolynomial, basis: tuple[FreeAlgebraPolynomial, ...], degree: int
) -> FreeAlgebraPolynomial:
    current = _map(value)
    changed = True
    reduction_steps = 0
    while changed:
        changed = False
        for reducer in basis:
            leading = _leading(reducer)
            if leading is None:
                continue
            leading_word, leading_coefficient = leading
            for word in tuple(sorted(current, key=lambda item: (len(item), item))):
                if len(word) > degree:
                    continue
                for start in range(len(word) - len(leading_word) + 1):
                    if word[start : start + len(leading_word)] != leading_word:
                        continue
                    factor = _admitted_quotient(current[word], leading_coefficient)
                    replacement = _multiply_monomial(
                        reducer, word[:start], word[start + len(leading_word) :]
                    )
                    reduction_steps += 1
                    if reduction_steps > MAX_FREE_ALGEBRA_GS_REDUCTION_STEPS:
                        raise OperationResourceAdmissionError(
                            location=("degree",),
                            code="free_algebra.gs_reduction_budget",
                            message="GS normal-form reduction exceeds its admitted work envelope",
                        )
                    current = _subtract(current, _map(replacement), factor)
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return _encode(value.alphabet, current)


def _compositions(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial, degree: int
) -> tuple[FreeAlgebraPolynomial, ...]:
    left_leading = _leading(left)
    right_leading = _leading(right)
    if left_leading is None or right_leading is None:
        return ()
    lw, lc = left_leading
    rw, rc = right_leading
    values: list[FreeAlgebraPolynomial] = []
    # Proper overlaps of leading words, plus inclusion ambiguities.
    for overlap in range(1, min(len(lw), len(rw))):
        if lw[-overlap:] == rw[:overlap]:
            first = _multiply_monomial(left, (), rw[overlap:])
            second = _multiply_monomial(right, lw[:-overlap], ())
            scale = _admitted_quotient(lc, rc)
            candidate = _encode(
                left.alphabet, _subtract(_map(first), _map(second), scale)
            )
            if max((len(term.word) for term in candidate.terms), default=0) <= degree:
                values.append(candidate)
    if len(lw) >= len(rw) and any(
        lw[start : start + len(rw)] == rw for start in range(len(lw) - len(rw) + 1)
    ):
        start = next(
            start
            for start in range(len(lw) - len(rw) + 1)
            if lw[start : start + len(rw)] == rw
        )
        first = left
        second = _multiply_monomial(right, lw[:start], lw[start + len(rw) :])
        scale = _admitted_quotient(lc, rc)
        values.append(
            _encode(left.alphabet, _subtract(_map(first), _map(second), scale))
        )
    return tuple(values)


def groebner_shirshov_through_degree(
    ideal: FreeAlgebraIdeal | Mapping[str, Any], degree: int
) -> GroebnerShirshovResult:
    value = _as_ideal(ideal)
    if value.side != "two-sided":
        raise OperationDomainValidationError(
            location=("ideal", "side"),
            code="free_algebra.gs_requires_two_sided",
            message="Groebner-Shirshov completion is defined here for two-sided ideals",
        )
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= MAX_FREE_ALGEBRA_WORD_LENGTH
    ):
        raise OperationResourceAdmissionError(
            location=("degree",),
            code="free_algebra.gs_degree_bound",
            message="GS degree exceeds the admitted envelope",
        )
    if any(
        len(generator.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS
        or any(
            len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH for term in generator.terms
        )
        for generator in value.generators
    ):
        raise OperationResourceAdmissionError(
            location=("ideal",),
            code="free_algebra.gs_generator_budget",
            message="GS generator expansion exceeds the admitted envelope",
        )
    basis = list(value.generators)
    compositions: list[FreeAlgebraPolynomial] = []
    # Completion is complete only at a fixed point.  In particular, (f, g)
    # and (g, f) are distinct ordered ambiguities in a noncommutative algebra;
    # considering only i <= j silently loses the reverse overlap.  The pair
    # and composition bounds are operation work bounds, not a stopping rule.
    seen = set(basis)
    while True:
        current_basis = tuple(basis)
        pair_count = len(current_basis) * len(current_basis)
        if pair_count > MAX_FREE_ALGEBRA_GS_PAIR_CHECKS:
            raise OperationResourceAdmissionError(
                location=("degree",),
                code="free_algebra.gs_pair_budget",
                message="GS ordered-pair completion exceeds its admitted work envelope",
            )
        additions: list[FreeAlgebraPolynomial] = []
        for left in current_basis:
            for right in current_basis:
                for candidate in _compositions(left, right, degree):
                    remainder = _normal_form(candidate, current_basis, degree)
                    compositions.append(remainder)
                    if len(compositions) > MAX_FREE_ALGEBRA_GS_COMPOSITIONS:
                        raise OperationResourceAdmissionError(
                            location=("degree",),
                            code="free_algebra.gs_composition_budget",
                            message="GS completion exceeds its admitted composition work envelope",
                        )
                    if (
                        remainder.terms
                        and remainder not in seen
                        and remainder not in additions
                    ):
                        additions.append(remainder)
        if not additions:
            # Every ordered pair of the final basis has now reduced every
            # degree-bounded overlap/inclusion composition to zero.  Only at
            # this fixed point may the result claim COMPLETE_THROUGH_DEGREE.
            break
        if len(basis) + len(additions) > MAX_FREE_ALGEBRA_RESULT_TERMS:
            raise OperationResourceAdmissionError(
                location=("degree",),
                code="free_algebra.gs_output",
                message="GS completion exceeds the admitted basis envelope",
            )
        basis.extend(additions)
        seen.update(additions)
    return GroebnerShirshovResult.model_construct(
        ideal=value, degree=degree, basis=tuple(basis), compositions=tuple(compositions)
    )


__all__ = ["groebner_shirshov_through_degree", "ideal_generated_prefix", "multiply"]
