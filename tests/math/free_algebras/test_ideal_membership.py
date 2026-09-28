"""Bounded exact membership for homogeneous free-algebra ideals."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras import operations
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
    canonical_word_key,
)
from jacobian.math.free_algebras._tools import TOOLS
from jacobian.math.free_algebras.operations import ideal_membership

OPERATION_ID = "free_algebra.two_sided_ideal.membership.decide"


def _poly(
    alphabet: tuple[str, ...], terms: dict[tuple[str, ...], int | Fraction]
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


def _commutator_ideal() -> tuple[FreeAlgebraIdeal, FreeAlgebraPolynomial]:
    alphabet = ("x", "y")
    relation = _poly(alphabet, {("x", "y"): 1, ("y", "x"): -1})
    return (
        FreeAlgebraIdeal(alphabet=alphabet, generators=(relation,), side="two-sided"),
        relation,
    )


def test_membership_and_nonmembership_have_exact_reduced_forms() -> None:
    ideal, relation = _commutator_ideal()
    member = ideal_membership(ideal, relation)
    nonmember_polynomial = _poly(("x", "y"), {("x", "y"): 1, ("y", "x"): 1})
    nonmember = ideal_membership(ideal, nonmember_polynomial)

    assert member.status == "MEMBER"
    assert member.completion_degree == 2
    assert member.normal_form is not None and member.normal_form.terms == ()
    assert nonmember.status == "NOT_MEMBER"
    assert nonmember.completion_degree == 2
    assert nonmember.normal_form is not None
    assert {
        term.word: term.coefficient.as_fraction()
        for term in nonmember.normal_form.terms
    } == {("x", "y"): Fraction(2)}
    assert type(member).model_validate_json(member.model_dump_json()) == member


def test_membership_handles_context_multiples_and_mixed_degree_candidates() -> None:
    ideal, relation = _commutator_ideal()
    x_relation = _poly(("x", "y"), {("x", "x", "y"): 1, ("x", "y", "x"): -1})
    mixed = _poly(("x", "y"), {("x",): 1, ("y",): -1})

    context_result = ideal_membership(ideal, x_relation)
    mixed_result = ideal_membership(
        ideal,
        _poly(
            ("x", "y"),
            {term.word: term.coefficient.as_fraction() for term in relation.terms}
            | {term.word: term.coefficient.as_fraction() for term in mixed.terms},
        ),
    )

    assert context_result.status == "MEMBER"
    assert mixed_result.status == "NOT_MEMBER"
    assert mixed_result.normal_form is not None
    assert {term.word for term in mixed_result.normal_form.terms} == {("x",), ("y",)}


def test_nonhomogeneous_generators_are_rejected() -> None:
    alphabet = ("x", "y")
    nonhomogeneous = _poly(alphabet, {("x", "y"): 1, ("x",): -1})
    ideal = FreeAlgebraIdeal(
        alphabet=alphabet, generators=(nonhomogeneous,), side="two-sided"
    )
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        ideal_membership(ideal, _poly(alphabet, {("x",): 1}))


def test_resource_incompletion_returns_unknown_without_a_conclusion(
    monkeypatch,
) -> None:
    ideal, relation = _commutator_ideal()

    def incomplete(*args, **kwargs):
        raise OperationResourceAdmissionError(
            location=("degree",), code="free_algebra.gs_pair_budget", message="limit"
        )

    monkeypatch.setattr(operations, "groebner_shirshov_through_degree", incomplete)
    result = ideal_membership(ideal, relation)

    assert result.status == "UNKNOWN"
    assert result.normal_form is None
    assert result.completion_degree is None


def test_membership_operation_is_published_with_a_valid_example() -> None:
    operation = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    assert operation.request_type.__name__ == "FreeAlgebraIdealMembershipRequest"
    assert operation.result_type.__name__ == "FreeAlgebraIdealMembershipResult"
    assert len(operation.examples) == 1
