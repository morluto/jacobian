"""Exact finite-dimensional spans for a supplied polynomial Ga action."""

from fractions import Fraction
from typing import Any, NoReturn

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations import _stable_kernels
from jacobian.math.polynomials.derivations._models import (
    PolynomialDerivation,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._stable_kernels import (
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.derivations.operations import ga_action_from_derivation
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(variable: str, exponent: int | None) -> RationalPolynomial:
    terms = () if exponent is None else ((1, exponent),)
    return RationalPolynomial.model_validate(
        {
            "variables": [variable],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": c, "den": 1}, "exponents": [e]}
                    for c, e in terms
                ]
            },
        }
    )


def _translation_action() -> PolynomialGaAction:
    x = _poly("x", 1)
    return ga_action_from_derivation(
        PolynomialDerivation(variables=("x",), images=(_poly("x", 0),)),
        ((x, _poly("x", 0), _poly("x", None)),),
    )


def _scaled_translation_action(coefficient: int) -> PolynomialGaAction:
    image = RationalPolynomial.model_validate(
        {
            "variables": ["x", "t"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 1}, "exponents": [1, 0]},
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": [0, 1],
                    },
                ]
            },
        }
    )
    return PolynomialGaAction(
        source_variables=("x",), parameter="t", generator_images=(image,)
    )


def test_translation_has_expected_matrix_and_stable_basis_roundtrips() -> None:
    action = _translation_action()
    result = ga_stable_subrepresentation(action, (_poly("x", 0), _poly("x", 1)))
    assert {
        term.exponents: term.coefficient.as_fraction()
        for term in result.action_matrix[0][0].polynomial.terms
    } == {(0,): Fraction(1)}
    assert result.action_matrix[1][0].polynomial.terms == ()
    assert result.action_matrix[0][1].variables == ("t",)
    assert {
        term.exponents: term.coefficient.as_fraction()
        for term in result.action_matrix[0][1].polynomial.terms
    } == {(1,): Fraction(1)}
    assert {
        term.exponents: term.coefficient.as_fraction()
        for term in result.action_matrix[1][1].polynomial.terms
    } == {(0,): Fraction(1)}
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.basis == result.basis
    assert restored.action_matrix == result.action_matrix


def test_noninvariant_span_is_rejected() -> None:
    # span{x} fails since x+t contains the constant polynomial.
    with pytest.raises(OperationDomainValidationError) as exc_info:
        ga_stable_subrepresentation(_translation_action(), (_poly("x", 1),))
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.not_stable"
    )


def test_large_valid_translation_coefficient_preserves_constant_subspace() -> None:
    coefficient = 10**125 + 3
    action = _scaled_translation_action(coefficient)
    result = ga_stable_subrepresentation(action, (_poly("x", 0),))
    assert result.action_matrix[0][0] == _poly("t", 0)


def test_dependent_supplied_basis_is_rejected() -> None:
    x = _poly("x", 1)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        ga_stable_subrepresentation(_translation_action(), (x, x))
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.dependent_basis"
    )


def test_serialized_action_claim_must_satisfy_additive_law() -> None:
    action = _translation_action()
    malformed_image = RationalPolynomial.model_validate(
        {
            "variables": ["x", "t"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 1}, "exponents": [1, 0]},
                    {"coefficient": {"num": 1, "den": 1}, "exponents": [0, 2]},
                ]
            },
        }
    )
    forged = PolynomialGaAction.model_construct(
        source_variables=action.source_variables,
        parameter=action.parameter,
        generator_images=(malformed_image,),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        ga_stable_subrepresentation(forged, (_poly("x", 0),))
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.action_composition"
    )


def test_retained_action_cells_are_included_before_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _translation_action()
    # The envelope counts retained entries, so a basis term and the retained
    # action both contribute; this bound is below what they add together.
    monkeypatch.setattr(_stable_kernels, "MAX_GA_ACTION_OUTPUT_CELLS", 4)

    def expansion_must_not_start(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("substitution ran before combined output admission")

    monkeypatch.setattr(_stable_kernels, "_substitute_basis", expansion_must_not_start)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_stable_subrepresentation(action, (_poly("x", 0),))
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.output_bytes"
    )


def test_admitted_stable_subrepresentation_phases_observe_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission, basis substitution, and coordinate reconstruction must each
    reach a checkpoint so a cancelled request stops inside the computation
    rather than after the operation returns."""
    observed: list[str] = []
    monkeypatch.setattr(
        _stable_kernels, "request_checkpoint", lambda stage: observed.append(stage)
    )

    result = ga_stable_subrepresentation(
        _translation_action(), (_poly("x", 0), _poly("x", 1))
    )
    assert result.action_matrix

    assert any("action admission" in phase for phase in observed)
    assert any("basis substitution" in phase for phase in observed)
    assert any("coordinate reconstruction" in phase for phase in observed)
