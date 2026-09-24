"""Exact finite-dimensional spans for a supplied polynomial Ga action."""

from fractions import Fraction

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationMatchRequest,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations import _stable_operations
from jacobian.math.polynomials.derivations._models import (
    PolynomialDerivation,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._stable_models import (
    PolynomialGaStableSubrepresentationRequest,
)
from jacobian.math.polynomials.derivations._stable_operations import (
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.derivations._tools import TOOLS
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


def _translation_action():
    x = _poly("x", 1)
    return ga_action_from_derivation(
        PolynomialDerivation(variables=("x",), images=(_poly("x", 0),)),
        ((x, _poly("x", 0), _poly("x", None)),),
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
    with pytest.raises(
        OperationDomainValidationError, match="outside the supplied span"
    ):
        ga_stable_subrepresentation(_translation_action(), (_poly("x", 1),))


def test_dependent_supplied_basis_is_rejected() -> None:
    x = _poly("x", 1)
    with pytest.raises(OperationDomainValidationError, match="linearly independent"):
        ga_stable_subrepresentation(_translation_action(), (x, x))


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
    with pytest.raises(OperationDomainValidationError, match="composition law"):
        ga_stable_subrepresentation(forged, (_poly("x", 0),))


def test_retained_action_bytes_are_included_before_substitution(monkeypatch) -> None:
    action = _translation_action()
    # Without the retained action this one-vector result estimate is 1,408
    # bytes. Including the serialized action takes the total above 2,000.
    monkeypatch.setattr(_stable_operations, "MAX_GA_ACTION_OUTPUT_BYTES", 2_000)

    def expansion_must_not_start(*_args, **_kwargs):
        raise AssertionError("substitution ran before combined output admission")

    monkeypatch.setattr(
        _stable_operations, "_substitute_basis", expansion_must_not_start
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="serialized-output budget"
    ):
        ga_stable_subrepresentation(action, (_poly("x", 0),))


def test_request_preserves_explicit_axis_and_catalog_example() -> None:
    request = PolynomialGaStableSubrepresentationRequest(
        action=_translation_action(), basis=(_poly("x", 0), _poly("x", 1))
    )
    assert request.basis[1].variables == ("x",)
    assert any(
        tool.operation_id == "algebraic_group.ga.stable_subrepresentation.compute"
        for tool in TOOLS
    )
    assert isinstance(TOOLS[-1], MathTool)
    matches = Catalog.open().match(
        OperationMatchRequest(need="Ga-stable finite polynomial span")
    )
    assert any(
        result.operation_id == "algebraic_group.ga.stable_subrepresentation.compute"
        for result in matches.matches
    )
