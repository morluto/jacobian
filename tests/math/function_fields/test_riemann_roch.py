import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
    FunctionFieldPlace,
    FunctionFieldRiemannRochMembership,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_place_valuation,
    function_field_riemann_roch_membership,
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


def _element(
    field: FiniteFunctionField,
    numerator: tuple[int, ...],
    denominator: tuple[int, ...] = (1,),
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(
                    characteristic=field.characteristic, coefficients=numerator
                ),
                denominator=PrimeFieldPolynomial(
                    characteristic=field.characteristic, coefficients=denominator
                ),
            ),
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
            valuation = function_field_place_valuation(place, element)
            assert valuation is not None
            assert valuation + multiplicity >= 0
    # The only denominator is (x+1)^2, supported where D has coefficient 2;
    # each numerator is x^(i+1), so there are no further finite poles.


def test_riemann_roch_membership_returns_complete_exact_support_profile() -> None:
    field = _field()
    x_plus_one = _finite_place(field, (1, 1))
    x_place = _finite_place(field, (0, 1))
    infinity = _infinity(field)
    divisor = _divisor(field, ((x_plus_one, 2), (x_place, -1), (infinity, 1)))

    # Independent hand calculation: x/(x+1)^2 has orders 1, -2, and 1 at
    # x, x+1, and infinity respectively.
    element = _element(field, (0, 1), (1, 2, 1))
    result = function_field_riemann_roch_membership(element, divisor)

    assert result.status == "IN_SPACE"
    assert [
        (
            row.place.kind,
            row.place.prime_polynomial.coefficients
            if row.place.prime_polynomial
            else None,
            row.element_valuation,
            row.divisor_multiplicity,
            row.sum,
        )
        for row in result.profile
    ] == [
        ("FINITE", (0, 1), 1, -1, 0),
        ("FINITE", (1, 1), -2, 2, 0),
        ("INFINITE", None, 1, 1, 2),
    ]
    restored = FunctionFieldRiemannRochMembership.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_riemann_roch_membership_reports_a_concrete_nonmembership_place() -> None:
    field = _field()
    x_plus_one = _finite_place(field, (1, 1))
    x_place = _finite_place(field, (0, 1))
    infinity = _infinity(field)
    divisor = _divisor(field, ((x_plus_one, 2), (x_place, -1), (infinity, 1)))

    # 1/(x+1)^2 has valuation zero at x, so it violates D(x)=-1.
    result = function_field_riemann_roch_membership(
        _element(field, (1,), (1, 2, 1)), divisor
    )

    assert result.status == "NOT_IN_SPACE"
    obstruction_rows = [row for row in result.profile if row.sum < 0]
    assert len(obstruction_rows) == 1
    obstruction = obstruction_rows[0]
    assert obstruction.place == x_place
    assert (
        obstruction.element_valuation,
        obstruction.divisor_multiplicity,
        obstruction.sum,
    ) == (
        0,
        -1,
        -1,
    )


def test_riemann_roch_membership_uses_zero_and_constant_structural_cases() -> None:
    field = _field()
    infinity = _infinity(field)
    negative_divisor = _divisor(field, ((infinity, -2),))
    zero = _element(field, (0,))

    zero_result = function_field_riemann_roch_membership(zero, negative_divisor)
    constant_result = function_field_riemann_roch_membership(
        _element(field, (1,)), negative_divisor
    )

    assert zero_result.status == "IN_SPACE"
    assert zero_result.profile == ()
    assert constant_result.status == "NOT_IN_SPACE"
    assert len(constant_result.profile) == 1
    assert (
        constant_result.profile[0].element_valuation,
        constant_result.profile[0].divisor_multiplicity,
        constant_result.profile[0].sum,
    ) == (0, -2, -2)


def test_riemann_roch_membership_includes_function_only_places() -> None:
    field = _field()
    infinity = _infinity(field)
    divisor = _divisor(field, ((infinity, 2),))

    # x+1 has a zero at the finite place x+1, which is outside D's support.
    result = function_field_riemann_roch_membership(_element(field, (1, 1)), divisor)

    assert result.status == "IN_SPACE"
    finite_zero = next(row for row in result.profile if row.place.kind == "FINITE")
    assert finite_zero.place.prime_polynomial == PrimeFieldPolynomial(
        characteristic=5, coefficients=(1, 1)
    )
    assert (
        finite_zero.element_valuation,
        finite_zero.divisor_multiplicity,
        finite_zero.sum,
    ) == (1, 0, 1)


def test_riemann_roch_membership_admits_exact_multiplicity_boundary() -> None:
    field = _field()
    infinity = _infinity(field)
    divisor = _divisor(field, ((infinity, (1 << 4096) - 1),))

    result = function_field_riemann_roch_membership(_element(field, (1,)), divisor)

    assert result.status == "IN_SPACE"
    assert result.profile[0].sum == (1 << 4096) - 1

    above_bound = _divisor(field, ((infinity, 1 << 4096),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_riemann_roch_membership(_element(field, (1,)), above_bound)
    assert error.value.errors()[0]["type"] == (
        "function_field.riemann_roch_multiplicity_exceeds_envelope"
    )


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


def test_riemann_roch_rejects_unsupported_extension_fields() -> None:
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
        "function_field.riemann_roch_requires_supported_model"
    )
