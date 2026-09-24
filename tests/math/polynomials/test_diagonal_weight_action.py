"""Exact diagonal G_m polynomial coaction and weight-slice tests."""

from fractions import Fraction
from typing import Any

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightAction,
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
    result = diagonal_weight_action(action, source)

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
    result = diagonal_weight_action(action.model_dump(), source.model_dump())
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
            {"variables": ["x", "y"], "weights": [1, -1]},
            _poly(("y", "x"), ((1, (1, 0)),)).model_dump(),
        )

    action = PolynomialWeightAction(variables=("x",), weights=(-64,))
    too_large = _poly(("x",), ((1, (65,)),))
    with pytest.raises(OperationResourceAdmissionError):
        diagonal_weight_action(action.model_dump(), too_large.model_dump())


def test_eight_variable_action_is_rejected_before_result_construction() -> None:
    # The Laurent coaction carrier has eight axes and the parameter owns one,
    # so an otherwise valid eight-variable request must surface as a domain
    # error at the request contract instead of leaking a Pydantic failure from
    # result construction.
    variables = tuple(f"x{i}" for i in range(8))
    action = PolynomialWeightAction(variables=variables, weights=(1,) * 8)
    source = _poly(variables, ((1, (1, 0, 0, 0, 0, 0, 0, 0)),))
    with pytest.raises(OperationDomainValidationError) as error:
        diagonal_weight_action(action.model_dump(), source.model_dump())
    assert error.value.errors()[0]["type"] == "polynomial_weight_action.request_shape"


def test_seven_variable_action_fills_the_carrier_with_its_parameter() -> None:
    variables = tuple(f"x{i}" for i in range(7))
    action = PolynomialWeightAction(
        variables=variables, weights=(1, 2, -1, 3, -2, 4, -3)
    )
    monomials = ((2, (2, 0, 0, 0, 0, 0, 0)), (1, (1, 0, 0, 0, 0, 0, 0)))
    source = _poly(variables, monomials)
    result = diagonal_weight_action(action, source)
    assert result.coaction.variables == (*variables, "t")
    # Independent oracle: exponents 2 and 1 on x0 carry weights 2*1 and 1*1.
    assert {
        tuple(term.exponents[:-1]): term.exponents[-1] for term in result.coaction.terms
    } == {
        (2, 0, 0, 0, 0, 0, 0): 2,
        (1, 0, 0, 0, 0, 0, 0): 1,
    }
    assert tuple(component.weight for component in result.components) == (1, 2)


def test_native_weight_helpers_take_canonical_values() -> None:
    from jacobian.math.polynomials.derivations import (
        diagonal_weight_action as package_diagonal,
    )
    from jacobian.math.polynomials.derivations import (
        gm_invariants_through_degree as package_invariants,
    )

    action = PolynomialWeightAction(variables=("x", "y"), weights=(1, -1))
    invariant = _poly(("x", "y"), ((1, (1, 1)),))
    result = package_diagonal(action, invariant)
    assert result.coaction.variables == ("x", "y", "t")
    assert result.weight_zero == invariant
    assert package_invariants(action, 2).dimension == 2


@pytest.mark.parametrize("bad_parameter", ["bad-name", "", "t" * 40, 1, None])
def test_malformed_parameter_is_rejected_before_expansion(bad_parameter: Any) -> None:
    action = PolynomialWeightAction(variables=("x",), weights=(1,))
    source = _poly(("x",), ((1, (64,)),))
    with pytest.raises(OperationDomainValidationError) as error:
        diagonal_weight_action(action, source, bad_parameter)
    assert error.value.errors()[0]["type"] == "polynomial_weight_action.request_shape"


def test_native_surface_excludes_wire_request_envelopes() -> None:
    import jacobian.math.polynomials.derivations as surface

    assert "PolynomialWeightInvariantRequest" not in surface.__all__
    assert not hasattr(surface, "PolynomialWeightInvariantRequest")
    assert "PolynomialWeightActionRequest" not in surface.__all__


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
    assert (
        tool.run(request).weight_zero
        == diagonal_weight_action(
            request.action, request.polynomial, request.parameter
        ).weight_zero
    )
    found = Catalog.open().match(OperationMatchRequest(need="diagonal integer weights"))
    assert any(item.operation_id == operation_id for item in found.matches)
