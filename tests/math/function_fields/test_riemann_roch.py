import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
    FunctionFieldPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_place_valuation,
    function_field_riemann_roch_space,
)


def _field(characteristic: int = 5) -> FiniteFunctionField:
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
        variable="x",
        defining_polynomial=(one,),
    )


def _finite_place(
    field: FiniteFunctionField, coefficients: tuple[int, ...]
) -> FunctionFieldPlace:
    return FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(
            characteristic=field.characteristic, coefficients=coefficients
        ),
        degree=len(coefficients) - 1,
    )


def _infinity(field: FiniteFunctionField) -> FunctionFieldPlace:
    return FunctionFieldPlace(field=field, kind="INFINITE", degree=1)


def _divisor(
    field: FiniteFunctionField,
    terms: tuple[tuple[FunctionFieldPlace, int], ...],
) -> FunctionFieldDivisor:
    return FunctionFieldDivisor(
        field=field,
        terms=tuple(
            FunctionFieldDivisorTerm(place=place, multiplicity=multiplicity)
            for place, multiplicity in terms
        ),
    )


def test_rational_riemann_roch_basis_has_exact_dimension_and_membership() -> None:
    field = _field()
    x_plus_one = _finite_place(field, (1, 1))
    x_place = _finite_place(field, (0, 1))
    infinity = _infinity(field)
    divisor = _divisor(
        field,
        ((x_plus_one, 2), (x_place, -1), (infinity, 1)),
    )

    space = function_field_riemann_roch_space(divisor)

    assert space.divisor.degree == 2
    assert space.dimension == 3 == len(space.basis)
    assert [
        element.coordinates[0].numerator.coefficients for element in space.basis
    ] == [
        (0, 1),
        (0, 0, 1),
        (0, 0, 0, 1),
    ]
    assert all(
        element.coordinates[0].denominator.coefficients == (1, 2, 1)
        for element in space.basis
    )
    for element in space.basis:
        for place, multiplicity in ((x_plus_one, 2), (x_place, -1), (infinity, 1)):
            assert function_field_place_valuation(place, element) + multiplicity >= 0
    # The only denominator is (x+1)^2, supported where D has coefficient 2;
    # each numerator is x^(i+1), so there are no further finite poles.


def test_riemann_roch_zero_and_negative_degree_spaces() -> None:
    field = _field()
    infinity = _infinity(field)

    zero_divisor_space = function_field_riemann_roch_space(_divisor(field, ()))
    assert zero_divisor_space.dimension == 1
    assert zero_divisor_space.basis[0].coordinates[0].numerator.is_one()

    negative_divisor = _divisor(field, ((infinity, -1),))
    empty_space = function_field_riemann_roch_space(negative_divisor)
    assert empty_space.dimension == 0
    assert empty_space.basis == ()


def test_riemann_roch_basis_capacity_boundary_is_accepted() -> None:
    field = _field()
    divisor = _divisor(field, ((_infinity(field), 12),))

    space = function_field_riemann_roch_space(divisor)

    assert space.dimension == 13
    assert len(space.basis) == 13
    assert [
        len(element.coordinates[0].numerator.coefficients) - 1
        for element in space.basis
    ] == list(range(13))


def test_riemann_roch_rejects_basis_or_coefficient_growth_before_place_factorization() -> (
    None
):
    field = _field()
    too_large_basis = _divisor(field, ((_infinity(field), 13),))
    with pytest.raises(OperationResourceAdmissionError) as basis_error:
        function_field_riemann_roch_space(too_large_basis)
    assert basis_error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_basis_exceeds_envelope"
    )

    reducible_quadratic = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(1, 0, 1)),
        degree=2,
    )
    huge_basis_over_invalid_place = _divisor(field, ((reducible_quadratic, 7),))
    with pytest.raises(OperationResourceAdmissionError) as output_error:
        function_field_riemann_roch_space(huge_basis_over_invalid_place)
    assert output_error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_basis_exceeds_envelope"
    )

    x_plus_one = _finite_place(field, (1, 1))
    output_degree_overflow = _divisor(
        field, ((x_plus_one, -12), (_infinity(field), 13))
    )
    with pytest.raises(OperationResourceAdmissionError) as coefficient_error:
        function_field_riemann_roch_space(output_degree_overflow)
    assert coefficient_error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_output_exceeds_envelope"
    )


def test_riemann_roch_multiplicity_bound_is_checked_before_place_admission() -> None:
    field = _field()
    forged = FunctionFieldDivisor.model_construct(
        field=field,
        terms=(
            FunctionFieldDivisorTerm.model_construct(
                place=None, multiplicity=1 << 4096
            ),
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_riemann_roch_space(forged)

    assert error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_multiplicity_exceeds_envelope"
    )


def test_riemann_roch_accepts_nonmonic_associate_of_x_place() -> None:
    field = _field()
    two_x_place = _finite_place(field, (0, 2))
    divisor = _divisor(field, ((two_x_place, 1),))

    space = function_field_riemann_roch_space(divisor)

    assert space.divisor.terms[0].place.prime_polynomial.coefficients == (0, 1)
    assert space.dimension == 2
    assert len(space.basis) == 2
    for element in space.basis:
        assert function_field_place_valuation(two_x_place, element) + 1 >= 0


def test_riemann_roch_rejects_algebraic_extension_fields() -> None:
    def rf(numerator: tuple[int, ...]) -> PrimeFieldRationalFunction:
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(characteristic=2, coefficients=numerator),
            denominator=PrimeFieldPolynomial(characteristic=2, coefficients=(1,)),
        )

    extension = FiniteFunctionField(
        characteristic=2,
        defining_polynomial=(rf((0, 1)), rf((1,)), rf((1,))),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_riemann_roch_space(
            FunctionFieldDivisor(field=extension, terms=())
        )

    assert error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_requires_rational_field"
    )
