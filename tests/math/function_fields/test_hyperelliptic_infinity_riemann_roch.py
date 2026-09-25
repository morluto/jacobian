from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    HyperellipticInfinityPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.hyperelliptic_infinity_riemann_roch import (
    HyperellipticInfinityRiemannRochRequest,
    hyperelliptic_infinity_riemann_roch_space,
)
from jacobian.math.function_fields.operations import (
    function_field_hyperelliptic_infinity_valuation,
)


def _hyperelliptic_field(
    characteristic: int, branch_coefficients: tuple[int, ...]
) -> FiniteFunctionField:
    zero = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(0,)
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(1,)
        ),
    )
    one = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(1,)
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(1,)
        ),
    )
    negative_branch = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=characteristic,
            coefficients=tuple(
                (-coefficient) % characteristic for coefficient in branch_coefficients
            ),
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(1,)
        ),
    )
    return FiniteFunctionField(
        characteristic=characteristic,
        variable="x",
        generator="y",
        defining_polynomial=(negative_branch, zero, one),
    )


def _place(field: FiniteFunctionField) -> HyperellipticInfinityPlace:
    return HyperellipticInfinityPlace(
        field=field,
        residue_field=FiniteFieldPresentation(
            characteristic=field.characteristic,
            modulus_coefficients=(0, 1),
            generator="z",
        ),
    )


def test_genus_one_basis_has_expected_valuations_and_rr_dimensions() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))  # y^2=x^3-x
    place = _place(field)
    expected_dimensions = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4}

    for multiplicity, expected_dimension in expected_dimensions.items():
        result = hyperelliptic_infinity_riemann_roch_space(place, multiplicity)
        assert result.genus == 1
        assert result.dimension == expected_dimension
        assert len(result.basis) == expected_dimension
        for basis_element in result.basis:
            valuation_result = function_field_hyperelliptic_infinity_valuation(
                place, basis_element
            )
            assert valuation_result.valuation.kind == "FINITE"
            assert valuation_result.valuation.value + multiplicity >= 0
        assert type(result).model_validate_json(result.model_dump_json()) == result


def test_genus_two_special_and_large_degree_spaces_match_closed_form() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 0, 0, 1))  # y^2=x^5-x
    place = _place(field)
    expected_dimensions = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5}

    for multiplicity, expected_dimension in expected_dimensions.items():
        result = hyperelliptic_infinity_riemann_roch_space(place, multiplicity)
        assert result.genus == 2
        assert result.dimension == expected_dimension
        assert len(result.basis) == expected_dimension
        # For m >= 2g-1, the independent Riemann-Roch theorem gives l(mP)=m+1-g.
        if multiplicity >= 3:
            assert result.dimension == multiplicity + 1 - result.genus
        valuations = []
        for element in result.basis:
            value = function_field_hyperelliptic_infinity_valuation(
                place, element
            ).valuation
            assert value.kind == "FINITE"
            valuations.append(value.value)
        if multiplicity == 5:
            assert valuations == [0, -2, -4, -5]
            assert [
                (
                    element.coordinates[0].numerator.coefficients,
                    element.coordinates[1].numerator.coefficients,
                )
                for element in result.basis
            ] == [((1,), (0,)), ((0, 1), (0,)), ((0, 0, 1), (0,)), ((0,), (1,))]
        else:
            assert all(value >= -multiplicity for value in valuations)


def test_negative_infinity_divisor_has_zero_space() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))

    result = hyperelliptic_infinity_riemann_roch_space(_place(field), -1)

    assert result.dimension == 0
    assert result.basis == ()


def test_basis_growth_is_rejected_before_construction() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))

    boundary = hyperelliptic_infinity_riemann_roch_space(_place(field), 13)
    assert boundary.dimension == 13

    genus_two = _hyperelliptic_field(5, (0, 4, 0, 0, 0, 1))
    assert (
        hyperelliptic_infinity_riemann_roch_space(_place(genus_two), 14).dimension == 13
    )
    with pytest.raises(OperationResourceAdmissionError, match="basis exceeds"):
        hyperelliptic_infinity_riemann_roch_space(_place(genus_two), 15)
    with pytest.raises(OperationResourceAdmissionError, match="basis exceeds"):
        hyperelliptic_infinity_riemann_roch_space(_place(field), 100_000)


def test_request_keeps_signed_divisor_coefficient() -> None:
    field = _hyperelliptic_field(5, (0, 4, 0, 1))
    request = HyperellipticInfinityRiemannRochRequest(
        place=_place(field), multiplicity=-3
    )

    assert request.multiplicity == -3
