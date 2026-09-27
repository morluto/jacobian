"""Exact contracts for free-algebra polynomial homomorphisms."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialHomomorphism,
    FreeAlgebraPolynomialSubstitutionRequest,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import multiply, substitute_polynomial

OPERATION_ID = "free_algebra.polynomial.substitute.compute"


def _polynomial(
    alphabet: tuple[str, ...],
    coefficients: dict[tuple[str, ...], int | Fraction],
) -> FreeAlgebraPolynomial:
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
            word=word,
        )
        for word, coefficient in sorted(
            coefficients.items(),
            key=lambda item: canonical_word_key(alphabet, item[0]),
            reverse=True,
        )
        if coefficient
    )
    return FreeAlgebraPolynomial(alphabet=alphabet, terms=terms)


def _coefficients(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _oracle(
    substitution: FreeAlgebraPolynomialHomomorphism,
    polynomial: FreeAlgebraPolynomial,
) -> dict[tuple[str, ...], Fraction]:
    """Independently expand each source word by recursive concatenation."""
    images = dict(zip(substitution.source_alphabet, substitution.images, strict=True))
    result: dict[tuple[str, ...], Fraction] = {}
    for source_term in polynomial.terms:
        partial: dict[tuple[str, ...], Fraction] = {
            (): source_term.coefficient.as_fraction()
        }
        for letter in source_term.word:
            next_partial: dict[tuple[str, ...], Fraction] = {}
            for prefix, prefix_coefficient in partial.items():
                for image_term in images[letter].terms:
                    word = prefix + image_term.word
                    contribution = (
                        prefix_coefficient * image_term.coefficient.as_fraction()
                    )
                    next_partial[word] = (
                        next_partial.get(word, Fraction(0)) + contribution
                    )
            partial = next_partial
        for word, coefficient in partial.items():
            result[word] = result.get(word, Fraction(0)) + coefficient
    return {word: coefficient for word, coefficient in result.items() if coefficient}


def test_exact_polynomial_substitution_matches_independent_word_expansion() -> None:
    target = ("a", "b")
    substitution = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=target,
        images=(
            _polynomial(target, {("a",): Fraction(1, 2), ("b",): 1}),
            _polynomial(target, {("a", "b"): -2, (): Fraction(1, 3)}),
        ),
    )
    source = _polynomial(
        ("x", "y"),
        {("x", "y"): 2, ("y", "x"): -1, ("x",): Fraction(3, 2), (): 4},
    )

    result = substitute_polynomial(substitution, source)

    assert result.alphabet == target
    assert _coefficients(result) == _oracle(substitution, source)


def test_zero_generator_image_annihilates_words_and_preserves_unit() -> None:
    target = ("a",)
    substitution = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=target,
        images=(_polynomial(target, {("a",): 2}), _polynomial(target, {})),
    )
    source = _polynomial(("x", "y"), {("x", "y"): 7, (): 3, ("y",): 9})

    assert _coefficients(substitute_polynomial(substitution, source)) == {
        (): Fraction(3)
    }


def test_substitution_preserves_products_and_composes() -> None:
    ab = ("a", "b")
    bc = ("u", "v")
    first = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x", "y"),
        target_alphabet=ab,
        images=(_polynomial(ab, {("a",): 1, ("b",): 1}), _polynomial(ab, {("b",): 1})),
    )
    second = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=ab,
        target_alphabet=bc,
        images=(
            _polynomial(bc, {("u",): 1, ("v",): -1}),
            _polynomial(bc, {("u", "v"): 1}),
        ),
    )
    p = _polynomial(("x", "y"), {("x",): 1, ("y",): -1})
    q = _polynomial(("x", "y"), {("y", "x"): 1, (): 2})

    # Multiplicativity is checked against the existing independent product kernel.
    image_product = substitute_polynomial(first, multiply(p, q).product)
    product_images = multiply(
        substitute_polynomial(first, p),
        substitute_polynomial(first, q),
    ).product
    assert _coefficients(image_product) == _coefficients(product_images)

    composed_images = tuple(
        substitute_polynomial(second, image) for image in first.images
    )
    composed = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=first.source_alphabet,
        target_alphabet=second.target_alphabet,
        images=composed_images,
    )
    sequential = substitute_polynomial(second, substitute_polynomial(first, p))
    direct = substitute_polynomial(composed, p)
    assert _coefficients(sequential) == _coefficients(direct)


def test_source_and_target_alphabets_are_bound() -> None:
    target = ("a",)
    substitution = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x",),
        target_alphabet=target,
        images=(_polynomial(target, {("a",): 1}),),
    )
    with pytest.raises(ValueError):
        FreeAlgebraPolynomialSubstitutionRequest(
            substitution=substitution,
            polynomial=_polynomial(("y",), {("y",): 1}),
        )
    with pytest.raises(ValueError):
        FreeAlgebraPolynomialHomomorphism(
            source_alphabet=("x",),
            target_alphabet=target,
            images=(_polynomial(("b",), {("b",): 1}),),
        )


def test_expansion_work_is_admitted_before_cartesian_expansion() -> None:
    target = ("a", "b")
    substitution = FreeAlgebraPolynomialHomomorphism(
        source_alphabet=("x",),
        target_alphabet=target,
        images=(_polynomial(target, {("a",): 1, ("b",): 1}),),
    )
    source = _polynomial(("x",), {("x",) * 16: 1})
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        substitute_polynomial(substitution, source)


def test_catalog_declaration_and_example() -> None:
    declaration = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    assert declaration.examples
    request = FreeAlgebraPolynomialSubstitutionRequest.model_validate_json(
        encode_strict_json(declaration.examples[0].input), strict=True
    )
    assert request.polynomial.alphabet == request.substitution.source_alphabet
    public = declaration.run(request)
    assert public.model_dump(mode="json") == {
        "alphabet": ["a", "b"],
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "word": []}],
    }
