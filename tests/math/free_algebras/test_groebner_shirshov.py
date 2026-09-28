"""Boundary and critical-ambiguity evidence for bounded GS completion."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras.operations import (
    _compositions,
    groebner_shirshov_through_degree,
)


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
                coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                word=word,
            )
            for word, coefficient in ordered
            if coefficient
        ),
    )


def _maximum_degree(values: tuple[FreeAlgebraPolynomial, ...]) -> int:
    return max((len(term.word) for value in values for term in value.terms), default=0)


def test_completion_prefix_excludes_higher_homogeneous_generators_and_compositions() -> (
    None
):
    alphabet = ("x", "y")
    degree_one = _poly(alphabet, {("y",): 1, ("x",): -1})
    degree_three = _poly(alphabet, {("y", "x", "y"): 1, ("x", "x", "x"): -1})
    ideal = FreeAlgebraIdeal(
        alphabet=alphabet,
        generators=(degree_one, degree_three),
        side="two-sided",
    )

    prefix = groebner_shirshov_through_degree(ideal, 2)
    assert prefix.status == "COMPLETE_THROUGH_DEGREE"
    assert prefix.basis == (degree_one,)
    assert _maximum_degree(prefix.compositions) <= 2

    full = groebner_shirshov_through_degree(ideal, 3)
    assert _maximum_degree(full.basis) <= 3
    assert _maximum_degree(full.compositions) <= 3


def test_all_occurrences_of_an_inclusion_word_produce_compositions() -> None:
    alphabet = ("x", "y")
    # The leading word yxy contains the leading word y at positions 0 and 2.
    f = _poly(alphabet, {("y", "x", "y"): 1, ("x", "x", "x"): -1})
    g = _poly(alphabet, {("y",): 1, ("x",): -1})
    candidates = _compositions(f, g, 3)
    # Direct independent expansion gives f - g*xy = xxy - xxx and
    # f - yx*g = yxx - xxx. Both positions matter even though the later
    # normal-form pass reduces each candidate to zero.
    observed = {
        tuple(
            sorted(
                (term.word, term.coefficient.as_fraction()) for term in candidate.terms
            )
        )
        for candidate in candidates
    }
    assert observed == {
        (
            (("x", "x", "x"), Fraction(-1)),
            (("x", "x", "y"), Fraction(1)),
        ),
        (
            (("x", "x", "x"), Fraction(-1)),
            (("y", "x", "x"), Fraction(1)),
        ),
    }, observed


def test_completion_rejects_inhomogeneous_generators() -> None:
    alphabet = ("x", "y")
    relation = _poly(alphabet, {("y", "x"): 1, ("x",): -1})
    ideal = FreeAlgebraIdeal(
        alphabet=alphabet, generators=(relation,), side="two-sided"
    )

    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        groebner_shirshov_through_degree(ideal, 2)
