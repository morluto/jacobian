"""Exact normal-word quotient bases and Hilbert prefixes."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import quotient_normal_word_profile

OPERATION_ID = "free_algebra.two_sided_quotient.normal_word_profile.compute"


def _poly(
    alphabet: tuple[str, ...], terms: dict[tuple[str, ...], int]
) -> FreeAlgebraPolynomial:
    ordered = sorted(
        terms.items(),
        key=lambda item: canonical_word_key(alphabet, item[0]),
        reverse=True,
    )
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(coefficient),
                word=word,
            )
            for word, coefficient in ordered
            if coefficient
        ),
    )


def _ideal(
    alphabet: tuple[str, ...], generators: tuple[FreeAlgebraPolynomial, ...]
) -> FreeAlgebraIdeal:
    return FreeAlgebraIdeal(alphabet=alphabet, generators=generators, side="two-sided")


def _words_of_degree(alphabet: tuple[str, ...], degree: int):
    return tuple(product(alphabet, repeat=degree))


def test_commutator_quotient_matches_independent_hand_normal_word_oracle() -> None:
    alphabet = ("x", "y")
    commutator = _poly(alphabet, {("x", "y"): 1, ("y", "x"): -1})
    result = quotient_normal_word_profile(_ideal(alphabet, (commutator,)), 5)

    # In QQ<x,y>/(xy-yx), canonical words are x^i y^(n-i). This hand oracle
    # independently enumerates the normal words, without using the GS output.
    expected = tuple(
        tuple(("x",) * x_count + ("y",) * (degree - x_count))
        for degree in range(6)
        for x_count in range(degree, -1, -1)
    )
    actual = tuple(
        word for component in result.components for word in component.normal_words
    )
    assert actual == expected
    assert result.hilbert_function == (1, 2, 3, 4, 5, 6)
    assert result.leading_words == (("y", "x"),)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_profile_covers_zero_and_finite_dimensional_quotients() -> None:
    alphabet = ("x", "y")
    free = quotient_normal_word_profile(_ideal(alphabet, ()), 3)
    assert free.hilbert_function == (1, 2, 4, 8)
    assert tuple(
        word for component in free.components for word in component.normal_words
    ) == tuple(
        word for degree in range(4) for word in _words_of_degree(alphabet, degree)
    )

    variables = _ideal(
        alphabet,
        (
            _poly(alphabet, {("x",): 1}),
            _poly(alphabet, {("y",): 1}),
        ),
    )
    point = quotient_normal_word_profile(variables, 4)
    assert point.hilbert_function == (1, 0, 0, 0, 0)
    assert point.components[0].normal_words == ((),)


def test_profile_preserves_empty_alphabet_and_empty_ideal() -> None:
    result = quotient_normal_word_profile(_ideal((), ()), 3)
    assert result.hilbert_function == (1, 0, 0, 0)
    assert result.components[0].normal_words == ((),)
    assert all(not component.normal_words for component in result.components[1:])


def test_word_search_is_admitted_before_completion() -> None:
    alphabet = ("x", "y")
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        quotient_normal_word_profile(_ideal(alphabet, ()), 14)
    assert (
        exc_info.value.errors()[0]["type"]
        == "free_algebra.quotient_profile_search_budget"
    )


def test_profile_operation_is_published() -> None:
    operation = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    assert operation.request_type.__name__ == "FreeAlgebraQuotientProfileRequest"
    assert operation.result_type.__name__ == "FreeAlgebraQuotientProfileResult"
    assert len(operation.examples) == 1
    alphabet = ("x", "y")
    commutator = _poly(alphabet, {("x", "y"): 1, ("y", "x"): -1})
    request = operation.request_type(ideal=_ideal(alphabet, (commutator,)), degree=3)
    result = operation.run(request)
    assert result.hilbert_function == (1, 2, 3, 4)
