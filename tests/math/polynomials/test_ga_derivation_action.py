"""Exact additive-group actions from locally nilpotent QQ derivations."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.polynomials.derivations._models import (
    GaActionRequest,
    PolynomialDerivation,
)
from jacobian.math.polynomials.derivations._tools import TOOLS
from jacobian.math.polynomials.derivations.operations import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
    ga_action_from_derivation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...],
    terms: tuple[tuple[Fraction, tuple[int, ...]], ...],
) -> RationalPolynomial:
    ordered = sorted(terms, key=lambda term: term[1], reverse=True)
    return RationalPolynomial.model_validate_json(
        json.dumps(
            {
                "domain": "QQ",
                "variables": list(variables),
                "polynomial": {
                    "terms": [
                        {
                            "coefficient": {
                                "num": str(coefficient.numerator),
                                "den": str(coefficient.denominator),
                            },
                            "exponents": list(exponents),
                        }
                        for coefficient, exponents in ordered
                    ]
                },
            },
        ),
        strict=True,
    )


def _terms(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(term.exponents): Fraction(term.coefficient.num, term.coefficient.den)
        for term in polynomial.polynomial.terms
    }


def _derivation() -> tuple[
    PolynomialDerivation, tuple[tuple[RationalPolynomial, ...], ...]
]:
    variables = ("x", "y", "z")
    zero = _poly(variables, ())
    x = _poly(variables, ((Fraction(1), (1, 0, 0)),))
    y = _poly(variables, ((Fraction(1), (0, 1, 0)),))
    z = _poly(variables, ((Fraction(1), (0, 0, 1)),))
    return (
        PolynomialDerivation(variables=variables, images=(zero, x, y)),
        ((x, zero), (y, x, zero), (z, y, x, zero)),
    )


def _add(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result = left.copy()
    for powers, coefficient in right.items():
        result[powers] = result.get(powers, Fraction(0)) + coefficient
        if result[powers] == 0:
            del result[powers]
    return result


def _multiply(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_powers, left_value in left.items():
        for right_powers, right_value in right.items():
            powers = tuple(
                a + b for a, b in zip(left_powers, right_powers, strict=True)
            )
            result[powers] = result.get(powers, Fraction(0)) + left_value * right_value
    return {powers: value for powers, value in result.items() if value}


def _power(
    value: dict[tuple[int, ...], Fraction], exponent: int, axes: int
) -> dict[tuple[int, ...], Fraction]:
    result = {(0,) * axes: Fraction(1)}
    for _ in range(exponent):
        result = _multiply(result, value)
    return result


def _independent_coaction_oracle(
    action_images: tuple[RationalPolynomial, ...],
) -> None:
    """Compare (rho_s tensor id)rho_t with x |-> rho_(s+t)(x) directly."""

    # The tested action uses x,y,z,t in that order; its s-lift is written
    # independently from the expected triangular formulas.
    axes = 5
    one = Fraction(1)
    x_s = {(1, 0, 0, 0, 0): one}
    y_s = {(0, 1, 0, 0, 0): one, (1, 0, 0, 1, 0): one}
    z_s = {
        (0, 0, 1, 0, 0): one,
        (0, 1, 0, 1, 0): one,
        (1, 0, 0, 2, 0): Fraction(1, 2),
    }
    source_lifts = (x_s, y_s, z_s)
    parameter_t = {(0, 0, 0, 0, 1): one}
    parameter_s_plus_t = {
        (0, 0, 0, 1, 0): one,
        (0, 0, 0, 0, 1): one,
    }

    for image in action_images:
        left: dict[tuple[int, ...], Fraction] = {}
        right: dict[tuple[int, ...], Fraction] = {}
        for powers, coefficient in _terms(image).items():
            substituted = {(0, 0, 0, 0, 0): coefficient}
            for axis, exponent in enumerate(powers[:-1]):
                substituted = _multiply(
                    substituted, _power(source_lifts[axis], exponent, axes)
                )
            substituted = _multiply(substituted, _power(parameter_t, powers[-1], axes))
            left = _add(left, substituted)

            unreplaced = {(*powers[:-1], 0, 0): coefficient}
            right = _add(
                right,
                _multiply(
                    unreplaced,
                    _power(parameter_s_plus_t, powers[-1], axes),
                ),
            )
        assert left == right


def test_triangular_locally_nilpotent_derivation_exponentiates_exactly() -> None:
    derivation, chains = _derivation()
    action = ga_action_from_derivation(derivation, chains)
    assert action.source_variables == ("x", "y", "z")
    assert action.parameter == "t"
    assert tuple(_terms(image) for image in action.generator_images) == (
        {(1, 0, 0, 0): Fraction(1)},
        {(0, 1, 0, 0): Fraction(1), (1, 0, 0, 1): Fraction(1)},
        {
            (0, 0, 1, 0): Fraction(1),
            (0, 1, 0, 1): Fraction(1),
            (1, 0, 0, 2): Fraction(1, 2),
        },
    )
    _independent_coaction_oracle(action.generator_images)


def test_action_checks_every_generator_chain() -> None:
    derivation, chains = _derivation()
    invalid = (chains[0], (chains[1][0], _poly(("x", "y", "z"), ())), chains[2])
    with pytest.raises(OperationDomainValidationError) as error:
        ga_action_from_derivation(derivation, invalid)
    assert (
        error.value.errors()[0]["type"] == "polynomial_derivation.certificate_mismatch"
    )


def test_action_parameter_axis_has_a_closed_variable_boundary() -> None:
    variables = tuple(f"x{i}" for i in range(8))
    zero = _poly(variables, ())
    images = (zero,) * 8
    derivation = PolynomialDerivation(variables=variables, images=images)
    chains = tuple(
        (
            _poly(
                variables, ((Fraction(1), tuple(1 if i == j else 0 for i in range(8))),)
            ),
            zero,
        )
        for j in range(8)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        ga_action_from_derivation(derivation, chains)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial_derivation.action_variable_budget"
    )


def test_action_accepts_full_carrier_with_a_fresh_parameter_axis() -> None:
    variables = tuple(f"x{i}" for i in range(7))
    zero = _poly(variables, ())
    derivation = PolynomialDerivation(variables=variables, images=(zero,) * 7)
    chains = tuple(
        (
            _poly(
                variables,
                ((Fraction(1), tuple(1 if i == j else 0 for i in range(7))),),
            ),
            zero,
        )
        for j in range(7)
    )
    result = ga_action_from_derivation(derivation, chains)
    assert result.parameter not in variables
    assert len(result.generator_images[0].variables) == 8
    assert _terms(result.generator_images[0]) == {(1, 0, 0, 0, 0, 0, 0, 0): Fraction(1)}


def test_action_chooses_a_fresh_parameter_when_t_is_a_source_axis() -> None:
    variables = ("t",)
    zero = _poly(variables, ())
    generator = _poly(variables, ((Fraction(1), (1,)),))
    derivation = PolynomialDerivation(variables=variables, images=(zero,))
    result = ga_action_from_derivation(derivation, ((generator, zero),))
    assert result.parameter == "action_0"
    assert result.generator_images[0].variables == ("t", "action_0")


def test_action_output_term_limit_is_checked_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials.derivations import operations

    derivation, chains = _derivation()
    monkeypatch.setattr(operations, "MAX_GA_ACTION_OUTPUT_TERMS", 1)
    with pytest.raises(OperationResourceAdmissionError) as error:
        ga_action_from_derivation(derivation, chains)
    assert (
        error.value.errors()[0]["type"] == "polynomial_derivation.action_output_budget"
    )


def test_ga_action_catalog_example_and_request_are_publishable() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "algebraic_group.ga.action_from_derivation.compute"
    )
    request = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.generator_images[0].variables == ("x", "y", "t")
    assert GaActionRequest.model_validate(request.model_dump()) == request

    assert result.parameter == "t"
    assert [
        (term.coefficient.num, term.coefficient.den, term.exponents)
        for term in result.generator_images[0].polynomial.terms
    ] == [(1, 1, (1, 0, 0)), (1, 1, (0, 1, 1))]


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        5,
        (None, None, None),
        (5, 5, 5),
    ],
)
def test_native_action_rejects_malformed_chains_as_domain_errors(
    invalid: Any,
) -> None:
    derivation, _ = _derivation()
    with pytest.raises(OperationDomainValidationError) as error:
        ga_action_from_derivation(derivation, invalid)
    assert error.value.errors()[0]["type"] == "polynomial_derivation.certificate_shape"


def test_native_action_rejects_lazy_chain_iterables_without_materializing() -> None:
    derivation, _ = _derivation()
    pulled = 0

    def lazy():
        nonlocal pulled
        while pulled < 100_000:
            pulled += 1
            yield ()

    with pytest.raises(OperationDomainValidationError):
        ga_action_from_derivation(derivation, lazy())
    assert pulled == 0


def test_action_input_chain_shape_is_bounded() -> None:
    with pytest.raises(ValidationError):
        GaActionRequest.model_validate(
            {
                "derivation": _derivation()[0],
                "chains": [[], [], []],
            }
        )
