"""Contract tests for exact rational-polynomial arc energy."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.unit_circle import (
    UnitCircleArcEnergyRequest,
    UnitCircleArcEnergyResult,
    unit_circle_arc_energy,
    verify_unit_circle_arc_energy,
)
from jacobian.math.polynomials.unit_circle._tools import TOOLS
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def polynomial(coefficients: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(coefficient=q(value), exponents=(degree,))
                for degree, value in reversed(tuple(enumerate(coefficients)))
                if value
            )
        ),
    )


def energy(
    coefficients: tuple[int, ...], start: Fraction, end: Fraction
) -> UnitCircleArcEnergyResult:
    return unit_circle_arc_energy(
        UnitCircleArcEnergyRequest(
            polynomial=polynomial(coefficients),
            start_turn=q(start),
            end_turn=q(end),
        )
    )


def rational_value(result: UnitCircleArcEnergyResult) -> Fraction:
    binding = result.pi_inverse_coefficient
    assert binding.element.presentation.degree == 1
    return binding.element.coefficients_ascending[0].as_fraction()


def test_three_exact_arc_energy_examples() -> None:
    right = energy((1, 1), Fraction(-1, 4), Fraction(1, 4))
    assert right.rational_part.as_fraction() == 1
    assert rational_value(right) == 2

    left = energy((1, 1), Fraction(1, 4), Fraction(3, 4))
    assert left.rational_part.as_fraction() == 1
    assert rational_value(left) == -2

    fourth_power_filter = energy((1, 2, 3, 2, 1), Fraction(1, 4), Fraction(3, 4))
    assert fourth_power_filter.rational_part.as_fraction() == Fraction(19, 2)
    assert rational_value(fourth_power_filter) == Fraction(-88, 3)


def test_zero_full_circle_monomial_and_crossing_zero() -> None:
    zero = energy((0,), Fraction(0), Fraction(0))
    assert zero.rational_part.as_fraction() == 0
    assert rational_value(zero) == 0

    full = energy((1, 2), Fraction(-1, 4), Fraction(3, 4))
    assert full.rational_part.as_fraction() == 5
    assert rational_value(full) == 0

    monomial = energy((0, 0, 3), Fraction(0), Fraction(1, 1))
    assert monomial.rational_part.as_fraction() == 9
    assert rational_value(monomial) == 0

    crossing = energy((1, 1), Fraction(3, 4), Fraction(5, 4))
    assert crossing.rational_part.as_fraction() == 1
    assert rational_value(crossing) == 2


def test_additivity_complement_and_nonrational_cyclotomic_coefficient() -> None:
    whole = energy((1, 1), Fraction(0), Fraction(1))
    left = energy((1, 1), Fraction(0), Fraction(1, 3))
    right = energy((1, 1), Fraction(1, 3), Fraction(1))
    assert (
        left.rational_part.as_fraction() + right.rational_part.as_fraction()
        == whole.rational_part.as_fraction()
    )
    assert left.pi_inverse_coefficient.element.presentation.degree == 2
    assert (
        right.pi_inverse_coefficient.element.presentation
        == left.pi_inverse_coefficient.element.presentation
    )
    assert verify_unit_circle_arc_energy(
        UnitCircleArcEnergyResult.model_validate_json(
            left.model_dump_json(), strict=True
        )
    )


def test_arc_admission_rejects_excessive_conductor() -> None:
    with pytest.raises(OperationDomainValidationError, match="conductor"):
        energy((1, 1), Fraction(0), Fraction(1, 33))


def test_native_and_mcp_paths_share_serialized_result() -> None:
    request = UnitCircleArcEnergyRequest(
        polynomial=polynomial((1, 1)),
        start_turn=q(Fraction(-1, 4)),
        end_turn=q(Fraction(1, 4)),
    )
    native = unit_circle_arc_energy(request)
    decoded = UnitCircleArcEnergyResult.model_validate_json(
        native.model_dump_json(), strict=True
    )
    assert decoded == native
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.unit_circle.arc_energy.compute"
    )
    public = invoke_operation(
        operation.operation_id, request.model_dump(mode="json"), Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")


def test_serialized_claim_forgery_is_structural_but_fails_verification() -> None:
    native = energy((1, 1), Fraction(-1, 4), Fraction(1, 4))
    payload = json.loads(native.model_dump_json())
    payload["rational_part"] = {"num": "2", "den": "1"}
    forged = UnitCircleArcEnergyResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert forged.rational_part.as_fraction() == 2
    assert not verify_unit_circle_arc_energy(forged)
