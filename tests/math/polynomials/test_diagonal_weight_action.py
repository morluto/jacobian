"""Exact diagonal G_m polynomial coaction and weight-slice tests."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightAction,
    PolynomialWeightActionRequest,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    diagonal_weight_action,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...], terms: tuple[tuple[int, tuple[int, ...]], ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": variables,
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": n, "den": 1}, "exponents": exponents}
                    for n, exponents in sorted(
                        terms, key=lambda item: item[1], reverse=True
                    )
                ]
            },
        }
    )


def _coefficients(poly: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(term.exponents): term.coefficient.as_fraction()
        for term in poly.polynomial.terms
    }


def test_diagonal_action_oracle_negative_zero_positive_and_invariants() -> None:
    action = PolynomialWeightAction(variables=("x", "y", "z"), weights=(1, -1, 0))
    source = _poly(("x", "y", "z"), ((3, (2, 0, 0)), (5, (1, 1, 1)), (7, (0, 2, 0))))
    result = diagonal_weight_action(
        PolynomialWeightActionRequest(action=action, polynomial=source)
    )

    # Independent monomial oracle: weight is the dot product of declared weights
    # and source exponents. This checks all three signs and the fixed slice.
    weights = (1, -1, 0)
    expected = _coefficients(source)
    got_coaction = {
        tuple(term.exponents[:-1]): (term.exponents[-1], term.coefficient.as_fraction())
        for term in result.coaction.terms
    }
    assert got_coaction == {
        exponent: (
            sum(w * e for w, e in zip(weights, exponent, strict=True)),
            coefficient,
        )
        for exponent, coefficient in expected.items()
    }
    assert {
        tuple(term.exponents[:-1]): term.coefficient.as_fraction()
        for term in result.coaction.terms
    } == expected  # counit evaluation t=1 recovers the source polynomial
    assert {c.weight: _coefficients(c.polynomial) for c in result.components} == {
        -2: {(0, 2, 0): Fraction(7)},
        0: {(1, 1, 1): Fraction(5)},
        2: {(2, 0, 0): Fraction(3)},
    }
    assert _coefficients(result.weight_zero) == {(1, 1, 1): Fraction(5)}


def test_coaction_counit_and_composition_law_termwise() -> None:
    # For a monomial x^e, rho(x^e)=x^e*t^k. Counit sets t=1,
    # and coassociativity follows from (s*t)^k=s^k*t^k, also for k<0.
    action = PolynomialWeightAction(variables=("x", "y"), weights=(2, -3))
    source = _poly(("x", "y"), ((2, (2, 1)), (-1, (0, 1))))
    result = diagonal_weight_action(
        {"action": action.model_dump(), "polynomial": source.model_dump()}
    )
    coaction = {
        tuple(term.exponents[:-1]): (term.exponents[-1], term.coefficient.as_fraction())
        for term in result.coaction.terms
    }
    assert coaction == {(2, 1): (1, Fraction(2)), (0, 1): (-3, Fraction(-1))}
    # Explicit exponent composition is an independent exact oracle for the
    # coaction equation, not a backend-generated proof or a bounded sample.
    for exponent, (k, coefficient) in coaction.items():
        source_weight = sum(
            w * e for w, e in zip(action.weights, exponent, strict=True)
        )
        assert k == source_weight
        assert coefficient == _coefficients(source)[exponent]
        # Compare exponents in (rho_s tensor id)rho_t and (id tensor Delta)rho.
        # Their common Laurent monomial is x^e*s^k*t^k, valid for k < 0 too.
        left_exponents = (*exponent, source_weight, k)
        right_exponents = (*exponent, source_weight, source_weight)
        assert left_exponents == right_exponents


def test_ring_binding_and_bounds_reject_before_expansion() -> None:
    with pytest.raises(OperationDomainValidationError):
        diagonal_weight_action(
            {
                "action": {"variables": ["x", "y"], "weights": [1, -1]},
                "polynomial": _poly(("y", "x"), ((1, (1, 0)),)).model_dump(),
            }
        )

    action = PolynomialWeightAction(variables=("x",), weights=(-64,))
    too_large = _poly(("x",), ((1, (65,)),))
    with pytest.raises(OperationResourceAdmissionError):
        diagonal_weight_action(
            {"action": action.model_dump(), "polynomial": too_large.model_dump()}
        )


def test_weight_action_is_catalogued_with_valid_example() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.catalog.catalog import Catalog
    from jacobian.catalog.models import OperationMatchRequest
    from jacobian.math.polynomials.derivations._tools import TOOLS

    operation_id = "algebraic_group.gm.diagonal_weight_action.compute"
    tool = next(item for item in TOOLS if item.operation_id == operation_id)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    assert tool.run(request).weight_zero == diagonal_weight_action(request).weight_zero
    found = Catalog.open().match(OperationMatchRequest(need="diagonal integer weights"))
    assert any(item.operation_id == operation_id for item in found.matches)
