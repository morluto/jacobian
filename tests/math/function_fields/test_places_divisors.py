import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
    FunctionFieldPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_divisor_degree,
    function_field_place_valuation,
    function_field_principal_divisor,
)


def _rf(
    num: tuple[int, ...], den: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=num),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=den),
    )


def _field() -> FiniteFunctionField:
    return FiniteFunctionField(characteristic=5, defining_polynomial=(_rf((1,)),))


def test_rational_places_and_principal_degree_zero() -> None:
    field = _field()
    x = FiniteFunctionFieldElement(field=field, coordinates=(_rf((0, 1)),))
    place = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
        degree=1,
    )
    assert function_field_place_valuation(place, x) == 1
    result = function_field_principal_divisor(field, x)
    assert result.degree == 0
    assert sorted(term.multiplicity for term in result.divisor.terms) == [-1, 1]


def test_reducible_finite_place_is_rejected_and_irreducible_one_is_accepted() -> None:
    field = _field()
    element = FiniteFunctionFieldElement(field=field, coordinates=(_rf((1,)),))
    reducible = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(1, 0, 1)),
        degree=2,
    )
    irreducible = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(2, 0, 1)),
        degree=2,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_place_valuation(reducible, element)
    assert error.value.errors()[0]["type"] == "function_field.place_not_prime"
    assert function_field_place_valuation(irreducible, element) == 0


def test_quadratic_numerator_over_gf5_uses_stable_factorization() -> None:
    field = _field()
    value = FiniteFunctionFieldElement(field=field, coordinates=(_rf((1, 0, 1)),))

    result = function_field_principal_divisor(field, value)

    finite = {
        term.place.prime_polynomial.coefficients: term.multiplicity
        for term in result.divisor.terms
        if term.place.prime_polynomial is not None
    }
    assert finite == {(2, 1): 1, (3, 1): 1}
    assert result.degree == 0


def test_constant_has_empty_principal_divisor() -> None:
    field = _field()
    one = FiniteFunctionFieldElement(field=field, coordinates=(_rf((1,)),))
    result = function_field_principal_divisor(field, one)
    assert result.divisor.terms == () and result.degree == 0


def test_native_divisor_consumers_reject_missing_authored_fields() -> None:
    forged_element = FiniteFunctionFieldElement.model_construct(field=_field())
    with pytest.raises(OperationDomainValidationError):
        function_field_principal_divisor(_field(), forged_element)

    divisor = FunctionFieldDivisor.model_construct(terms=())
    with pytest.raises(OperationDomainValidationError):
        function_field_divisor_degree(divisor)


def test_divisor_degree_rejects_a_forged_place_parent() -> None:
    field = _field()
    other = FiniteFunctionField(
        characteristic=7,
        defining_polynomial=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=7, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=7, coefficients=(1,)),
            ),
        ),
    )
    forged_place = FunctionFieldPlace.model_construct(
        field=other,
        kind="INFINITE",
        degree=1,
    )
    divisor = FunctionFieldDivisor.model_construct(
        field=field,
        terms=(
            FunctionFieldDivisorTerm.model_construct(
                place=forged_place,
                multiplicity=1,
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_divisor_degree(divisor)
    assert error.value.errors()[0]["type"] == "function_field.divisor_parent"


def test_principal_divisor_cancels_and_canonicalizes_rational_presentations() -> None:
    field = _field()
    x_over_x = FiniteFunctionFieldElement(
        field=field, coordinates=(_rf((0, 1), (0, 1)),)
    )
    cancelled = function_field_principal_divisor(field, x_over_x)
    assert cancelled.divisor.terms == ()
    x_squared_over_x = FiniteFunctionFieldElement(
        field=field, coordinates=(_rf((0, 0, 1), (0, 1)),)
    )
    reduced = function_field_principal_divisor(field, x_squared_over_x)
    assert sorted(term.multiplicity for term in reduced.divisor.terms) == [-1, 1]
    assert reduced.element.coordinates[0].denominator.is_one()
