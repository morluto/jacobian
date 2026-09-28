"""Exact degree components for homogeneous two-sided free-algebra ideals."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
import sympy

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import ideal_degree_component

OPERATION_ID = "free_algebra.two_sided_ideal.degree_component.compute"


def _poly(
    alphabet: tuple[str, ...], values: dict[tuple[str, ...], int | Fraction]
) -> FreeAlgebraPolynomial:
    ordered = sorted(
        values.items(),
        key=lambda item: canonical_word_key(alphabet, item[0]),
        reverse=True,
    )
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
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


def _matrix_basis(
    basis: tuple[FreeAlgebraPolynomial, ...], words: tuple[tuple[str, ...], ...]
) -> sympy.Matrix:
    return sympy.Matrix(
        [
            [
                next(
                    (
                        sympy.Rational(term.coefficient.as_fraction())
                        for term in polynomial.terms
                        if term.word == word
                    ),
                    sympy.Integer(0),
                )
                for word in words
            ]
            for polynomial in basis
        ]
    )


def test_commutator_ideal_degree_three_matches_independent_context_span() -> None:
    alphabet = ("x", "y")
    relation = _poly(alphabet, {("x", "y"): 1, ("y", "x"): -1})
    ideal = _ideal(alphabet, (relation,))
    result = ideal_degree_component(ideal, 3)

    words = tuple(product(alphabet, repeat=3))
    contexts = tuple(
        (left, right)
        for left_length in range(2)
        for left in product(alphabet, repeat=left_length)
        for right in product(alphabet, repeat=1 - left_length)
    )
    source_rows = []
    relation_map = {
        term.word: term.coefficient.as_fraction() for term in relation.terms
    }
    for left, right in contexts:
        source_rows.append(
            [
                sympy.Rational(
                    relation_map.get(
                        word[len(left) : len(word) - len(right) if right else None], 0
                    )
                )
                if word[: len(left)] == left
                and (not right or word[-len(right) :] == right)
                else sympy.Integer(0)
                for word in words
            ]
        )
    expected = sympy.Matrix(source_rows).rref()[0]
    actual = _matrix_basis(result.component_basis, words)

    assert result.ambient_dimension == 8
    assert result.ideal_dimension == 4
    assert actual == expected
    assert all(
        tuple(len(term.word) for term in polynomial.terms)
        == (3,) * len(polynomial.terms)
        for polynomial in result.component_basis
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_degree_component_covers_degree_zero_and_empty_alphabet() -> None:
    empty = _poly((), {})
    ideal = _ideal((), (empty,))

    degree_zero = ideal_degree_component(ideal, 0)
    degree_one = ideal_degree_component(ideal, 1)

    assert degree_zero.ambient_dimension == 1
    assert degree_zero.component_basis == ()
    assert degree_one.ambient_dimension == 0
    assert degree_one.component_basis == ()


def test_generators_of_different_degrees_contribute_distinct_context_families() -> None:
    alphabet = ("x", "y")
    ideal = _ideal(
        alphabet,
        (
            _poly(alphabet, {("x",): 1}),
            _poly(alphabet, {("x", "y"): 1, ("y", "x"): -1}),
        ),
    )

    result = ideal_degree_component(ideal, 2)

    assert result.ideal_dimension == 3
    assert {
        term.word for polynomial in result.component_basis for term in polynomial.terms
    } == {("x", "x"), ("x", "y"), ("y", "x")}


def test_repeated_large_denominators_are_canonicalized_before_growth_admission() -> (
    None
):
    alphabet = ("x", "y")
    denominator = 10**63
    generator = _poly(
        alphabet,
        {("x",): Fraction(1, denominator), ("y",): Fraction(1, denominator)},
    )

    result = ideal_degree_component(_ideal(alphabet, (generator,)), 1)

    assert result.ideal_dimension == 1
    assert {
        term.word: term.coefficient.as_fraction()
        for term in result.component_basis[0].terms
    } == {("y",): Fraction(1), ("x",): Fraction(1)}


def test_component_operation_is_published() -> None:
    operation = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    assert operation.request_type.__name__ == "FreeAlgebraIdealDegreeComponentRequest"
    assert operation.result_type.__name__ == "FreeAlgebraIdealDegreeComponentResult"


def test_degree_component_rejects_nonsided_and_nonhomogeneous_inputs() -> None:
    alphabet = ("x", "y")
    relation = _poly(alphabet, {("x", "y"): 1, ("y",): -1})
    with pytest.raises(OperationDomainValidationError, match="two-sided"):
        ideal_degree_component(
            FreeAlgebraIdeal(alphabet=alphabet, generators=(), side="left"), 1
        )
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        ideal_degree_component(_ideal(alphabet, (relation,)), 2)


def test_degree_component_rejects_oversized_ambient_word_axis_before_expansion() -> (
    None
):
    alphabet = ("x", "y")
    relation = _poly(alphabet, {("x",): 1})
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ideal_degree_component(_ideal(alphabet, (relation,)), 8)
    assert exc_info.value.errors()[0]["type"] == "free_algebra.component_word_axis"


def test_degree_component_admits_source_and_result_cells_together() -> None:
    alphabet = ("😀" * 64, "😁" * 64)
    generator = _poly(
        alphabet,
        dict.fromkeys(product(alphabet, repeat=4), 1),
    )
    ideal = _ideal(alphabet, (generator,) * 32)

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ideal_degree_component(ideal, 4)

    assert exc_info.value.errors()[0]["type"] == "free_algebra.component_cells"
