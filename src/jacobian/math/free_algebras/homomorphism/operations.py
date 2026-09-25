"""Bounded exact evaluation of free associative algebra homomorphisms."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
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
from jacobian.math.free_algebras.homomorphism._models import (
    FreeAlgebraHomomorphism,
    FreeAlgebraHomomorphismApplyRequest,
    FreeAlgebraHomomorphismApplyResult,
)


def _resource(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("polynomial",), code=f"free_algebra.{code}", message=message
    )


def _admit(
    request: FreeAlgebraHomomorphismApplyRequest,
) -> tuple[
    FreeAlgebraHomomorphism,
    FreeAlgebraPolynomial,
    dict[str, FreeAlgebraPolynomial],
]:
    """Establish all substitution growth bounds before distributive expansion."""
    hom, polynomial = request.homomorphism, request.polynomial
    try:
        polynomial = FreeAlgebraPolynomial.model_validate(polynomial.model_dump())
        hom = type(hom).model_validate(hom.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(),
            code="free_algebra.homomorphism_shape",
            message="homomorphism request is not canonical",
        ) from exc

    if len(polynomial.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
        _resource(
            "homomorphism_operand_terms",
            "source polynomial exceeds the 64-term substitution budget",
        )
    image_by_letter = dict(zip(hom.source_alphabet, hom.generator_images, strict=True))
    for letter, image in image_by_letter.items():
        if len(image.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _resource(
                "homomorphism_image_terms",
                f"image of {letter!r} exceeds the 64-term substitution budget",
            )
        if any(
            len(image_term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH
            for image_term in image.terms
        ):
            _resource(
                "homomorphism_image_word_length",
                f"image of {letter!r} contains a word longer than 32 letters",
            )
    image_support_bound = {
        letter: len(image.terms) for letter, image in image_by_letter.items()
    }
    image_word_bound = {
        letter: max((len(term.word) for term in image.terms), default=0)
        for letter, image in image_by_letter.items()
    }
    expansion_bound = 0
    max_digits = 0
    for term in polynomial.terms:
        if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
            _resource(
                "homomorphism_word_length",
                "source word exceeds the 32-letter substitution budget",
            )
        count = 1
        for letter in term.word:
            count *= image_support_bound[letter]
            if count > MAX_FREE_ALGEBRA_RESULT_TERMS:
                _resource(
                    "homomorphism_expansion",
                    "a substituted word exceeds the 4096-term expansion budget",
                )
        expansion_bound += count
        if expansion_bound > MAX_FREE_ALGEBRA_RESULT_TERMS:
            _resource(
                "homomorphism_expansion",
                "substitution exceeds the 4096-term expansion budget",
            )
        max_image_length = sum(image_word_bound[letter] for letter in term.word)
        if max_image_length > 2 * MAX_FREE_ALGEBRA_WORD_LENGTH:
            _resource(
                "homomorphism_word_length",
                "substituted words exceed the 64-letter result bound",
            )
        term_digits = canonical_rational_component_digits(term.coefficient)
        for letter in term.word:
            term_digits += max(
                (
                    canonical_rational_component_digits(image_term.coefficient)
                    for image_term in image_by_letter[letter].terms
                ),
                default=0,
            )
        max_digits = max(max_digits, term_digits)
    # Each coefficient is a sum of at most the admitted expansion count.
    addition_digits = len(str(expansion_bound - 1)) if expansion_bound > 1 else 0
    predicted_digits = max_digits + addition_digits
    if predicted_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _resource(
            "homomorphism_coefficient_growth",
            "predicted substitution coefficient growth exceeds the 64-digit result bound",
        )
    return hom, polynomial, image_by_letter


def _substitute(
    polynomial: FreeAlgebraPolynomial,
    hom: FreeAlgebraHomomorphism,
    image_by_letter: dict[str, FreeAlgebraPolynomial],
) -> FreeAlgebraPolynomial:
    values: defaultdict[tuple[str, ...], Fraction] = defaultdict(Fraction)
    for term in polynomial.terms:
        partial: dict[tuple[str, ...], Fraction] = {(): term.coefficient.as_fraction()}
        for letter in term.word:
            next_values: defaultdict[tuple[str, ...], Fraction] = defaultdict(Fraction)
            for prefix, prefix_coefficient in partial.items():
                for image_term in image_by_letter[letter].terms:
                    word = prefix + image_term.word
                    next_values[word] += (
                        prefix_coefficient * image_term.coefficient.as_fraction()
                    )
            partial = next_values
        for word, coefficient in partial.items():
            values[word] += coefficient
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(coefficient), word=word
        )
        for word, coefficient in sorted(
            ((word, value) for word, value in values.items() if value),
            key=lambda item: canonical_word_key(hom.target_alphabet, item[0]),
            reverse=True,
        )
    )
    return FreeAlgebraPolynomial(alphabet=hom.target_alphabet, terms=terms)


def apply(
    request: FreeAlgebraHomomorphismApplyRequest,
) -> FreeAlgebraHomomorphismApplyResult:
    """Substitute generator images in word order and collect exact terms."""
    hom, polynomial, image_by_letter = _admit(request)
    image = _substitute(polynomial, hom, image_by_letter)
    return FreeAlgebraHomomorphismApplyResult.model_construct(
        homomorphism=hom, polynomial=polynomial, image=image
    )
