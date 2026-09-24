import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import function_field_genus


def _rational_field(characteristic: int) -> FiniteFunctionField:
    one = PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(1,)
        ),
        denominator=PrimeFieldPolynomial(
            characteristic=characteristic, coefficients=(1,)
        ),
    )
    return FiniteFunctionField(
        characteristic=characteristic,
        variable="t",
        defining_polynomial=(one,),
    )


@pytest.mark.parametrize("characteristic", (2, 3, 5, 257))
def test_rational_function_field_has_genus_zero(characteristic: int) -> None:
    field = _rational_field(characteristic)

    result = function_field_genus(field)

    assert result.field == field
    assert result.genus == 0


def test_genus_rejects_a_nontrivial_algebraic_extension() -> None:
    def rational_function(
        numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
    ) -> PrimeFieldRationalFunction:
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(characteristic=2, coefficients=numerator),
            denominator=PrimeFieldPolynomial(
                characteristic=2, coefficients=denominator
            ),
        )

    elliptic_function_field = FiniteFunctionField(
        characteristic=2,
        defining_polynomial=(
            rational_function((0, 1)),
            rational_function((1,)),
            rational_function((1,)),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_genus(elliptic_function_field)

    assert error.value.errors()[0]["type"] == (
        "function_field.genus_requires_rational_field"
    )


def test_genus_rejects_composite_characteristic_before_classifying_field() -> None:
    field = FiniteFunctionField(
        characteristic=4,
        defining_polynomial=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=4, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=4, coefficients=(1,)),
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_genus(field)

    assert error.value.errors()[0]["type"] == (
        "function_field.characteristic_not_prime"
    )
