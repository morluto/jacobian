"""Exact contracts for composition of free-algebra homomorphisms."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialHomomorphism,
    FreeAlgebraPolynomialHomomorphismCompositionRequest,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import compose_polynomial_homomorphisms

OPERATION_ID = "free_algebra.homomorphism.compose.compute"


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


def _map(
    source: tuple[str, ...],
    target: tuple[str, ...],
    images: tuple[dict[tuple[str, ...], int | Fraction], ...],
) -> FreeAlgebraPolynomialHomomorphism:
    return FreeAlgebraPolynomialHomomorphism(
        source_alphabet=source,
        target_alphabet=target,
        images=tuple(_polynomial(target, image) for image in images),
    )


def test_identity_maps_are_two_sided_and_preserve_empty_axes() -> None:
    identity = _map(("x", "y"), ("x", "y"), ({("x",): 1}, {("y",): 1}))
    other = _map(("x", "y"), (), ({(): 2}, {(): -3}))
    empty_identity = _map((), (), ())

    assert compose_polynomial_homomorphisms(identity, other) == other
    assert compose_polynomial_homomorphisms(other, empty_identity) == other
    assert (
        compose_polynomial_homomorphisms(empty_identity, empty_identity)
        == empty_identity
    )


def test_composition_order_zero_images_and_noncommuting_products() -> None:
    f = _map(
        ("x", "y"),
        ("u", "v"),
        ({("u",): 1, ("v",): 1}, {("v", "u"): 1}),
    )
    g = _map(("u", "v"), ("a", "b"), ({("a",): 1}, {("b",): 1}))

    result = compose_polynomial_homomorphisms(f, g)

    assert result.source_alphabet == ("x", "y")
    assert result.target_alphabet == ("a", "b")
    assert _coefficients(result.images[0]) == {
        ("a",): Fraction(1),
        ("b",): Fraction(1),
    }
    assert _coefficients(result.images[1]) == {("b", "a"): Fraction(1)}

    zero_second = _map(("u", "v"), ("a", "b"), ({("a",): 1}, {}))
    zero_composition = compose_polynomial_homomorphisms(f, zero_second)
    assert zero_composition.images[1].terms == ()


def test_zero_image_annihilates_long_prefix_before_output_length_admission() -> None:
    f = _map(("x",), ("u", "v"), ({("u",) * 31 + ("v",): 1},))
    g = _map(("u", "v"), ("a",), ({("a",) * 32: 1}, {}))

    result = compose_polynomial_homomorphisms(f, g)

    assert result.images[0].terms == ()


def test_zero_image_does_not_inflate_surviving_coefficient_bound() -> None:
    f = _map(
        ("x",),
        ("u", "v"),
        ({("u",) * 31 + ("v",): 1, (): 1},),
    )
    g = _map(("u", "v"), ("a",), ({("a",): 100}, {}))

    result = compose_polynomial_homomorphisms(f, g)

    assert _coefficients(result.images[0]) == {(): Fraction(1)}


def test_composition_is_associative_for_small_maps() -> None:
    f = _map(("x",), ("u", "v"), ({("u", "v"): 1, (): 1},))
    g = _map(("u", "v"), ("r",), ({("r",): 1}, {("r", "r"): 1}))
    h = _map(("r",), ("a", "b"), ({("a",): 1, ("b",): -1},))

    left = compose_polynomial_homomorphisms(compose_polynomial_homomorphisms(f, g), h)
    right = compose_polynomial_homomorphisms(f, compose_polynomial_homomorphisms(g, h))

    assert left == right


def test_wrong_intermediate_alphabet_and_order_are_rejected() -> None:
    f = _map(("x",), ("u", "v"), ({("u",): 1},))
    g = _map(("v", "u"), ("a",), ({("a",): 1}, {("a",): 1}))

    with pytest.raises(ValueError, match="f target alphabet must equal g source"):
        FreeAlgebraPolynomialHomomorphismCompositionRequest(f=f, g=g)
    with pytest.raises(ValueError, match="f target alphabet must equal g source"):
        compose_polynomial_homomorphisms(f, g)


def _matrix_add(left, right):
    return tuple(
        tuple(left[row][col] + right[row][col] for col in range(2)) for row in range(2)
    )


def _matrix_multiply(left, right):
    return tuple(
        tuple(
            sum((left[row][k] * right[k][col] for k in range(2)), Fraction(0))
            for col in range(2)
        )
        for row in range(2)
    )


def _evaluate(polynomial: FreeAlgebraPolynomial, matrices):
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    result = ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0)))
    for term in polynomial.terms:
        value = identity
        for letter in term.word:
            value = _matrix_multiply(value, matrices[letter])
        value = tuple(
            tuple(entry * term.coefficient.as_fraction() for entry in row)
            for row in value
        )
        result = _matrix_add(result, value)
    return result


def test_composition_agrees_with_independent_matrix_evaluation() -> None:
    f = _map(("x",), ("u", "v"), ({("u",): 1, ("v",): 1},))
    g = _map(("u", "v"), ("a", "b"), ({("a",): 1}, {("b",): 1}))
    composed = compose_polynomial_homomorphisms(f, g)
    matrix_values = {
        "a": ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
        "b": ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
    }

    actual = _evaluate(composed.images[0], matrix_values)
    evaluated_g_images = {
        letter: _evaluate(image, matrix_values)
        for letter, image in zip(g.source_alphabet, g.images, strict=True)
    }
    expected = _evaluate(f.images[0], evaluated_g_images)
    assert actual == expected


def test_aggregate_growth_is_preflighted_before_any_image_expansion(
    monkeypatch,
) -> None:
    source = tuple(f"x{i}" for i in range(26))
    middle = ("a", "b", "c")
    target = ("z" * 64,)
    f = _map(source, middle, tuple({tuple(middle): 1} for _ in source))
    g = _map(
        middle,
        target,
        tuple({(target[0],) * 11: 1, (target[0],) * 10: 1} for _ in middle),
    )
    calls = 0

    def unexpected_expansion(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("expansion started before aggregate admission")

    monkeypatch.setattr(
        operations, "_expand_polynomial_substitution", unexpected_expansion
    )
    with pytest.raises(OperationResourceAdmissionError, match="allocation bound"):
        compose_polynomial_homomorphisms(f, g)
    assert calls == 0


def test_composition_request_json_and_catalog_result_round_trip() -> None:
    tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    request = FreeAlgebraPolynomialHomomorphismCompositionRequest.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert isinstance(result, FreeAlgebraPolynomialHomomorphism)
    restored = FreeAlgebraPolynomialHomomorphism.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result
    assert _coefficients(restored.images[0]) == {("a",): Fraction(3)}
