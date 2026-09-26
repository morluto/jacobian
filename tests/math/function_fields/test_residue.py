"""Residue maps for rational function fields, checked by quotient arithmetic."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_place_residue,
    function_field_place_valuation,
)


def _rf(
    numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=numerator),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=denominator),
    )


def _field() -> FiniteFunctionField:
    return FiniteFunctionField(characteristic=5, defining_polynomial=(_rf((1,)),))


def _finite_place(coefficients: tuple[int, ...]) -> FunctionFieldPlace:
    return FunctionFieldPlace(
        field=_field(),
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(
            characteristic=5, coefficients=coefficients
        ),
        degree=len(coefficients) - 1,
    )


def _element(numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)):
    field = _field()
    return FiniteFunctionFieldElement(
        field=field, coordinates=(_rf(numerator, denominator),)
    )


def test_finite_residue_is_exact_reduction_in_irreducible_quotient() -> None:
    place = _finite_place((1, 1, 1))
    result = function_field_place_residue(place, _element((0, 0, 1)))

    assert result.residue.presentation == FiniteFieldPresentation.model_construct(
        characteristic=5, modulus_coefficients=(1, 1, 1), generator="z"
    )
    # z^2 = -z - 1 in GF(5)[z]/(z^2+z+1).
    assert result.residue.coordinates == (4, 4)


def test_cubic_place_residue_uses_exact_extension_field_coordinates() -> None:
    # x^3 + x + 1 is irreducible over GF(2). In its quotient, z^3 = z + 1,
    # hence z^5 = z^2 + z + 1. The residue carrier must retain all three
    # power-basis coordinates and the exact place modulus.
    field = FiniteFunctionField(
        characteristic=2,
        defining_polynomial=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
            ),
        ),
    )
    place = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(
            characteristic=2, coefficients=(1, 1, 0, 1)
        ),
        degree=3,
    )
    x_to_fifth = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(
                    characteristic=2, coefficients=(0, 0, 0, 0, 0, 1)
                ),
                denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
            ),
        ),
    )

    assert function_field_place_valuation(place, x_to_fifth) == 0
    result = function_field_place_residue(place, x_to_fifth)

    assert result.residue.presentation == FiniteFieldPresentation.model_construct(
        characteristic=2, modulus_coefficients=(1, 1, 0, 1), generator="z"
    )
    assert result.residue.coordinates == (1, 1, 1)


def test_residue_cancels_a_common_factor_before_reduction() -> None:
    # (x^2 - 1)/(x - 1) is regular at x=1 and has residue 2.
    result = function_field_place_residue(
        _finite_place((4, 1)), _element((4, 0, 1), (4, 1))
    )
    assert result.residue.coordinates == (2,)


def test_infinite_residue_of_regular_functions() -> None:
    infinity = FunctionFieldPlace(field=_field(), kind="INFINITE", degree=1)
    assert function_field_place_residue(
        infinity, _element((0, 1), (1, 0, 1))
    ).residue.coordinates == (0,)
    assert function_field_place_residue(
        infinity, _element((3,))
    ).residue.coordinates == (3,)


def test_zero_element_has_zero_residue_at_every_place() -> None:
    zero = _element((0,))
    at_x = function_field_place_residue(_finite_place((0, 1)), zero)
    assert at_x.residue.coordinates == (0,)
    at_shifted = function_field_place_residue(_finite_place((2, 1)), zero)
    assert at_shifted.residue.coordinates == (0,)
    at_infinity = function_field_place_residue(
        FunctionFieldPlace(field=_field(), kind="INFINITE", degree=1), zero
    )
    assert at_infinity.residue.coordinates == (0,)


def test_residue_detects_zero_before_the_valuation_routine(monkeypatch) -> None:
    # The order-division routine cannot express the valuation of zero.  The
    # residue operation must detect the zero rational function structurally
    # rather than relying on the routine to mishandle it.
    import jacobian.math.function_fields.operations as operations

    def forbidden(value: object, place: object) -> int:
        raise AssertionError("zero element reached the valuation routine")

    monkeypatch.setattr(operations, "_rf_valuation", forbidden)
    result = function_field_place_residue(_finite_place((0, 1)), _element((0,)))
    assert result.residue.coordinates == (0,)


def test_residue_rejects_a_pole() -> None:
    infinity = FunctionFieldPlace(field=_field(), kind="INFINITE", degree=1)
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_place_residue(infinity, _element((0, 1)))
    assert (
        error.value.errors()[0]["type"]
        == "function_field.residue_requires_integral_element"
    )
