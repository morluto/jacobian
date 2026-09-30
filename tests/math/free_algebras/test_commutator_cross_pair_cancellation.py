"""Exact cross-pair cancellation under bounded commutator admission."""

from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial, FreeAlgebraTerm
from jacobian.math.free_algebras.commutator._models import FreeAlgebraCommutatorResult
from jacobian.math.free_algebras.commutator.operations import commutator


def polynomial(x: int, y: int, constant: int = 0) -> FreeAlgebraPolynomial:
    pairs = [(("y",), y), (("x",), x), ((), constant)]
    return FreeAlgebraPolynomial(
        alphabet=("x", "y"),
        terms=tuple(
            FreeAlgebraTerm(
                word=word, coefficient=CanonicalRational.from_fraction(Fraction(c))
            )
            for word, c in pairs
            if c
        ),
    )


@pytest.mark.parametrize("scale", [10**31, 10**60])
@pytest.mark.parametrize("constant", [0, 1])
def test_proportional_nonconstant_parts_cancel(scale: int, constant: int) -> None:
    left = polynomial(scale, scale)
    right = polynomial(2 * scale, 2 * scale, constant)
    result = commutator(left, right)
    assert result.commutator.terms == ()
    assert (
        FreeAlgebraCommutatorResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_nonzero_small_difference_survives_large_cross_pair_cancellation() -> None:
    n = 10**60
    result = commutator(polynomial(n, n), polynomial(n, n + 1))
    assert {
        term.word: term.coefficient.as_fraction() for term in result.commutator.terms
    } == {
        ("x", "y"): Fraction(n),
        ("y", "x"): Fraction(-n),
    }
    assert (
        FreeAlgebraCommutatorResult.model_validate_json(result.model_dump_json())
        == result
    )


@pytest.mark.parametrize("perturbation", (0, 1))
def test_dense_large_coefficients_reuse_one_bounded_integer_convolution(
    perturbation: int,
) -> None:
    scale = 10**60
    words = tuple(sorted(product(("x", "y"), repeat=5), reverse=True))
    left = FreeAlgebraPolynomial(
        alphabet=("x", "y"),
        terms=tuple(
            FreeAlgebraTerm(word=word, coefficient=CanonicalRational(num=scale, den=1))
            for word in words
        ),
    )
    right = FreeAlgebraPolynomial(
        alphabet=("x", "y"),
        terms=tuple(
            FreeAlgebraTerm(
                word=word,
                coefficient=CanonicalRational(
                    num=2 * scale + (perturbation if index == 0 else 0), den=1
                ),
            )
            for index, word in enumerate(words)
        ),
    )
    result = commutator(left, right)
    # Bilinearity gives [P, 2P + delta*w0] = delta*[P, w0]. This oracle uses
    # just the 32 source words, independently of the 1,024-pair convolution.
    expected: dict[tuple[str, ...], Fraction] = {}
    for word in words:
        forward, backward = word + words[0], words[0] + word
        expected[forward] = expected.get(forward, Fraction()) + perturbation * scale
        expected[backward] = expected.get(backward, Fraction()) - perturbation * scale
    assert {
        term.word: term.coefficient.as_fraction() for term in result.commutator.terms
    } == {word: coefficient for word, coefficient in expected.items() if coefficient}
    assert (
        FreeAlgebraCommutatorResult.model_validate_json(result.model_dump_json())
        == result
    )
