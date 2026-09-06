"""Contract tests for exact rational-polynomial arc energy."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from sympy import ZZ, Poly, Symbol

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.unit_circle import (
    UnitCircleArcEnergyResult,
    unit_circle_arc_energy,
    verify_unit_circle_arc_energy,
)
from jacobian.math.polynomials.unit_circle._models import UnitCircleArcEnergyRequest
from jacobian.math.polynomials.unit_circle.operations import (
    _REAL_CYCLOTOMIC_POLYNOMIALS,
    _real_cyclotomic_record,
)
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


def rational_polynomial(coefficients: tuple[Fraction, ...]) -> RationalPolynomial:
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
    return unit_circle_arc_energy(polynomial(coefficients), q(start), q(end))


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


def test_standard_real_cyclotomic_field_and_embedding_are_fixed_by_conductor() -> None:
    result = energy((1, 1), Fraction(0), Fraction(1, 8))
    binding = result.pi_inverse_coefficient
    assert result.cyclotomic_conductor == 8
    assert binding.element.presentation.coefficients_descending == ("1", "0", "-2")
    assert tuple(
        value.as_fraction() for value in binding.element.coefficients_ascending
    ) == (Fraction(0), Fraction(1, 2))
    assert binding.embedding_record.embedding.root.real_root_index == 1
    assert binding.embedding_record.isolating_interval.lower.as_fraction() == 1
    assert binding.embedding_record.isolating_interval.upper.as_fraction() == 2
    assert verify_unit_circle_arc_energy(result)


@pytest.mark.parametrize("conductor", (4, 8, 12, 16, 20, 24, 28, 32))
def test_every_admitted_standard_embedding_interval_is_exact(conductor: int) -> None:
    presentation, record = _real_cyclotomic_record(conductor)
    x = Symbol("x")
    defining = Poly.from_list(
        [int(value) for value in _REAL_CYCLOTOMIC_POLYNOMIALS[conductor]],
        gens=x,
        domain=ZZ,
    )
    assert defining.is_irreducible
    assert presentation.degree == defining.degree()
    interval = record.isolating_interval
    assert (
        defining.count_roots(interval.lower.as_fraction(), interval.upper.as_fraction())
        == 1
    )
    assert record.embedding.root.real_root_index == presentation.degree - 1


def test_dense_degree_32_and_large_exact_integer_are_admitted() -> None:
    dense = energy((1,) * 33, Fraction(0), Fraction(1, 32))
    assert dense.cyclotomic_conductor == 32
    assert dense.pi_inverse_coefficient.element.presentation.degree == 8
    assert verify_unit_circle_arc_energy(dense)

    large = unit_circle_arc_energy(rational_polynomial((Fraction(10**47),)), q(0), q(1))
    assert large.rational_part.as_fraction() == 10**94
    assert verify_unit_circle_arc_energy(large)


def test_large_unwrapped_turn_is_reduced_before_cyclotomic_recurrence() -> None:
    integer_part = 10**1000
    large = energy(
        (1, 1),
        Fraction(integer_part, 3),
        Fraction(integer_part + 1, 3),
    )
    reduced = energy((1, 1), Fraction(1, 3), Fraction(2, 3))
    assert large.rational_part == reduced.rational_part
    assert large.pi_inverse_coefficient == reduced.pi_inverse_coefficient
    assert verify_unit_circle_arc_energy(large)


def test_arc_admission_rejects_excessive_conductor() -> None:
    with pytest.raises(OperationDomainValidationError, match="conductor"):
        energy((1, 1), Fraction(0), Fraction(1, 33))

    # Endpoint denominators alone fit, but adjoining i requires conductor 36.
    with pytest.raises(OperationDomainValidationError, match="conductor"):
        energy((1, 1), Fraction(0), Fraction(1, 18))


def test_arc_admission_bounds_exact_coefficient_growth_before_expansion() -> None:
    oversized = UnitCircleArcEnergyRequest(
        polynomial=rational_polynomial((Fraction(10**48),)),
        start_turn=q(0),
        end_turn=q(1),
    )
    with pytest.raises(OperationDomainValidationError, match="exact-growth"):
        unit_circle_arc_energy(
            oversized.polynomial, oversized.start_turn, oversized.end_turn
        )

    many_denominators = UnitCircleArcEnergyRequest(
        polynomial=rational_polynomial(
            tuple(Fraction(1, 10**12 + offset) for offset in range(5))
        ),
        start_turn=q(0),
        end_turn=q(Fraction(1, 4)),
    )
    with pytest.raises(OperationDomainValidationError, match="denominators"):
        unit_circle_arc_energy(
            many_denominators.polynomial,
            many_denominators.start_turn,
            many_denominators.end_turn,
        )


def test_serialized_claim_forgery_is_structural_but_fails_verification() -> None:
    native = energy((1, 1), Fraction(-1, 4), Fraction(1, 4))
    assert not verify_unit_circle_arc_energy(
        native.model_copy(update={"pi_inverse_coefficient": None})  # type: ignore[arg-type]
    )
    payload = json.loads(native.model_dump_json())
    payload["rational_part"] = {"num": "2", "den": "1"}
    forged = UnitCircleArcEnergyResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert forged.rational_part.as_fraction() == 2
    assert not verify_unit_circle_arc_energy(forged)

    payload = json.loads(native.model_dump_json())
    payload["cyclotomic_conductor"] = 8
    forged_conductor = UnitCircleArcEnergyResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_unit_circle_arc_energy(forged_conductor)

    irrational = energy((1, 1), Fraction(0), Fraction(1, 8))
    payload = json.loads(irrational.model_dump_json())
    payload["pi_inverse_coefficient"]["element"]["coefficients_ascending"][1] = {
        "num": "1",
        "den": "3",
    }
    forged_coordinate = UnitCircleArcEnergyResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_unit_circle_arc_energy(forged_coordinate)

    payload = json.loads(irrational.model_dump_json())
    payload["pi_inverse_coefficient"]["embedding_record"]["isolating_interval"] = {
        "lower": {"num": "3", "den": "1"},
        "upper": {"num": "4", "den": "1"},
        "interval_type": "OPEN",
    }
    forged_embedding = UnitCircleArcEnergyResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_unit_circle_arc_energy(forged_embedding)
