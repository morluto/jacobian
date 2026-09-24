import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_element_multiply,
    function_field_genus,
)


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
        "function_field.genus_requires_supported_model"
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


def _hyperelliptic_field(
    branch: tuple[int, ...],
    *,
    characteristic: int = 5,
    denominator: tuple[int, ...] = (1,),
) -> FiniteFunctionField:
    def rational_function(
        coefficients: tuple[int, ...], denominator_coefficients: tuple[int, ...] = (1,)
    ) -> PrimeFieldRationalFunction:
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(
                characteristic=characteristic, coefficients=coefficients
            ),
            denominator=PrimeFieldPolynomial(
                characteristic=characteristic,
                coefficients=denominator_coefficients,
            ),
        )

    return FiniteFunctionField(
        characteristic=characteristic,
        variable="x",
        generator="y",
        # The field equation is y^2 - f(x) = 0.
        defining_polynomial=(
            rational_function(
                tuple((-coefficient) % characteristic for coefficient in branch),
                denominator,
            ),
            rational_function((0,)),
            rational_function((1,)),
        ),
    )


@pytest.mark.parametrize(
    ("branch", "genus"),
    [
        ((0, 4, 0, 1), 1),
        ((4, 0, 0, 0, 1), 1),
        ((1, 1, 0, 0, 0, 1), 2),
        ((0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 5),
    ],
)
def test_hyperelliptic_function_field_genus_and_parent_composition(
    branch: tuple[int, ...], genus: int
) -> None:
    field = _hyperelliptic_field(branch)

    result = function_field_genus(field)

    assert result.field == field
    assert result.genus == genus
    y = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(0,)),
                denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
            ),
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
            ),
        ),
    )
    square = function_field_element_multiply(y, y)
    assert square.product.field == field
    assert square.product.coordinates[0].numerator.coefficients == branch


@pytest.mark.parametrize(
    "field",
    [
        _hyperelliptic_field((0, 0, 0, 1)),  # repeated root
        _hyperelliptic_field((1, 4, 4, 1)),  # (x - 1)^2 (x + 1)
        _hyperelliptic_field((1, 1, 0, 1), characteristic=2),
        _hyperelliptic_field((2,)),  # constant branch polynomial
        _hyperelliptic_field((0, 4, 0, 1), denominator=(1, 1)),
    ],
)
def test_genus_rejects_unsupported_quadratic_curve_models(
    field: FiniteFunctionField,
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_genus(field)

    assert error.value.errors()[0]["type"] == (
        "function_field.genus_requires_supported_model"
    )
