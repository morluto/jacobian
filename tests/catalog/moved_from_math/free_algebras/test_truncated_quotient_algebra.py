"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/free_algebras/test_truncated_quotient_algebra.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.free_algebras import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    TruncatedFreeAlgebraQuotient,
)
from jacobian.math.free_algebras._models import canonical_word_key


def _polynomial(
    alphabet: tuple[str, ...], values: dict[tuple[str, ...], Fraction | int]
) -> FreeAlgebraPolynomial:
    terms = tuple(
        FreeAlgebraTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
            word=word,
        )
        for word, coefficient in sorted(
            values.items(),
            key=lambda entry: canonical_word_key(alphabet, entry[0]),
            reverse=True,
        )
        if coefficient
    )
    return FreeAlgebraPolynomial(alphabet=alphabet, terms=terms)


def _ideal(
    alphabet: tuple[str, ...], generators: tuple[FreeAlgebraPolynomial, ...]
) -> FreeAlgebraIdeal:
    return FreeAlgebraIdeal(alphabet=alphabet, generators=generators, side="two-sided")


def _coordinates(value: FreeAlgebraPolynomial) -> dict[tuple[str, ...], Fraction]:
    return {term.word: term.coefficient.as_fraction() for term in value.terms}


def _multiply_coordinates(
    algebra: TruncatedFreeAlgebraQuotient,
    left: dict[tuple[str, ...], Fraction],
    right: dict[tuple[str, ...], Fraction],
) -> dict[tuple[str, ...], Fraction]:
    positions = {word: index for index, word in enumerate(algebra.basis_words)}
    result: dict[tuple[str, ...], Fraction] = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            value = algebra.multiplication[positions[left_word]][positions[right_word]]
            for term in value.terms:
                result[term.word] = result.get(term.word, Fraction(0)) + (
                    left_coefficient
                    * right_coefficient
                    * term.coefficient.as_fraction()
                )
    return {word: coefficient for word, coefficient in result.items() if coefficient}


def test_catalog_example_round_trips_and_is_discoverable() -> None:
    operation_id = "free_algebra.two_sided_quotient.truncated_algebra.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, TruncatedFreeAlgebraQuotient)
    assert len(result.basis_words) == 6
