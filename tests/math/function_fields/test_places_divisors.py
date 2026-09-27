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
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_divisor_add,
    function_field_divisor_degree,
    function_field_divisor_effective_parts,
    function_field_divisor_negate,
    function_field_divisor_scale,
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


def test_divisor_add_negate_scale_have_exact_canonical_support() -> None:
    field = _field()
    x_place = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
        degree=1,
    )
    infinity = FunctionFieldPlace(field=field, kind="INFINITE", degree=1)
    left = FunctionFieldDivisor(
        field=field,
        terms=(
            FunctionFieldDivisorTerm(place=x_place, multiplicity=2),
            FunctionFieldDivisorTerm(place=infinity, multiplicity=-1),
        ),
    )
    right = FunctionFieldDivisor(
        field=field,
        terms=(
            FunctionFieldDivisorTerm(place=x_place, multiplicity=-1),
            FunctionFieldDivisorTerm(place=infinity, multiplicity=3),
        ),
    )

    total = function_field_divisor_add(left, right)
    assert {(t.place.kind, t.multiplicity) for t in total.terms} == {
        ("FINITE", 1),
        ("INFINITE", 2),
    }
    assert total.degree == 3
    assert (
        function_field_divisor_add(left, function_field_divisor_negate(left)).terms
        == ()
    )
    scaled = function_field_divisor_scale(left, -2)
    assert {t.place.kind: t.multiplicity for t in scaled.terms} == {
        "FINITE": -4,
        "INFINITE": 2,
    }
    assert function_field_divisor_scale(left, 0).terms == ()


def test_divisor_add_rejects_different_function_field_parents() -> None:
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
    left = FunctionFieldDivisor(field=field, terms=())
    right = FunctionFieldDivisor(field=other, terms=())
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_divisor_add(left, right)
    assert error.value.errors()[0]["type"] == "function_field.parent_mismatch"


def test_divisor_multiplicity_bound_is_checked_before_prime_factorization() -> None:
    field = _field()
    reducible = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(1, 0, 1)),
        degree=2,
    )
    divisor = FunctionFieldDivisor.model_construct(
        field=field,
        terms=(
            FunctionFieldDivisorTerm.model_construct(
                place=reducible, multiplicity=1 << 4096
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_divisor_negate(divisor)
    assert (
        error.value.errors()[0]["type"]
        == "function_field.divisor_multiplicity_exceeds_envelope"
    )


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


def test_effective_parts_reconstruct_and_have_disjoint_effective_support() -> None:
    field = _field()
    x_place = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
        degree=1,
    )
    infinity = FunctionFieldPlace(field=field, kind="INFINITE", degree=1)
    x_plus_one = FunctionFieldPlace(
        field=field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(1, 1)),
        degree=1,
    )
    divisor = FunctionFieldDivisor(
        field=field,
        terms=(
            FunctionFieldDivisorTerm(place=infinity, multiplicity=-3),
            FunctionFieldDivisorTerm(place=x_place, multiplicity=2),
            FunctionFieldDivisorTerm(place=x_plus_one, multiplicity=-1),
        ),
    )

    result = function_field_divisor_effective_parts(divisor)

    def coefficients(value: FunctionFieldDivisor) -> dict[str, int]:
        return {term.place.model_dump_json(): term.multiplicity for term in value.terms}

    original = coefficients(result.divisor)
    positive = coefficients(result.positive_part)
    negative = coefficients(result.negative_part)
    assert {
        place: positive.get(place, 0) - negative.get(place, 0) for place in original
    } == original
    assert all(multiplicity > 0 for multiplicity in positive.values())
    assert all(multiplicity > 0 for multiplicity in negative.values())
    assert positive.keys().isdisjoint(negative)
    assert result.divisor.degree == (
        result.positive_part.degree - result.negative_part.degree
    )
    assert len(result.positive_part.terms) <= len(result.divisor.terms)
    assert len(result.negative_part.terms) <= len(result.divisor.terms)


def test_effective_parts_is_presentation_order_independent_and_handles_zero() -> None:
    field = _field()
    places = (
        FunctionFieldPlace(
            field=field,
            kind="FINITE",
            prime_polynomial=PrimeFieldPolynomial(
                characteristic=5, coefficients=(0, 1)
            ),
            degree=1,
        ),
        FunctionFieldPlace(field=field, kind="INFINITE", degree=1),
    )
    terms = tuple(
        FunctionFieldDivisorTerm(place=place, multiplicity=sign)
        for place, sign in zip(places, (1, -1), strict=True)
    )
    left = function_field_divisor_effective_parts(
        FunctionFieldDivisor(field=field, terms=terms)
    )
    right = function_field_divisor_effective_parts(
        FunctionFieldDivisor(field=field, terms=terms[::-1])
    )
    assert left.positive_part == right.positive_part
    assert left.negative_part == right.negative_part
    empty = function_field_divisor_effective_parts(
        FunctionFieldDivisor(field=field, terms=())
    )
    assert empty.positive_part.terms == empty.negative_part.terms == ()


def test_effective_parts_exhaustively_replays_small_coefficient_profiles() -> None:
    from itertools import product

    field = _field()
    places = (
        FunctionFieldPlace(
            field=field,
            kind="FINITE",
            prime_polynomial=PrimeFieldPolynomial(
                characteristic=5, coefficients=(0, 1)
            ),
            degree=1,
        ),
        FunctionFieldPlace(
            field=field,
            kind="FINITE",
            prime_polynomial=PrimeFieldPolynomial(
                characteristic=5, coefficients=(1, 1)
            ),
            degree=1,
        ),
        FunctionFieldPlace(field=field, kind="INFINITE", degree=1),
    )
    for profile in product((-2, -1, 0, 1, 2), repeat=len(places)):
        terms = tuple(
            FunctionFieldDivisorTerm(place=place, multiplicity=value)
            for place, value in zip(places, profile, strict=True)
            if value
        )
        result = function_field_divisor_effective_parts(
            FunctionFieldDivisor(field=field, terms=terms)
        )
        source = {
            place.model_dump_json(): value
            for place, value in zip(places, profile, strict=True)
        }
        plus = {
            term.place.model_dump_json(): term.multiplicity
            for term in result.positive_part.terms
        }
        minus = {
            term.place.model_dump_json(): term.multiplicity
            for term in result.negative_part.terms
        }
        assert all(
            source[key] == plus.get(key, 0) - minus.get(key, 0) for key in source
        )


def test_effective_parts_bounds_scalar_output_before_place_admission() -> None:
    field = _field()
    place = FunctionFieldPlace(field=field, kind="INFINITE", degree=1)
    largest = (1 << 4096) - 1
    divisor = FunctionFieldDivisor(
        field=field,
        terms=(FunctionFieldDivisorTerm(place=place, multiplicity=largest),),
    )
    result = function_field_divisor_effective_parts(divisor)
    assert result.positive_part.terms[0].multiplicity == largest

    oversized = FunctionFieldDivisor.model_construct(
        field=field,
        terms=(
            FunctionFieldDivisorTerm.model_construct(
                place=None, multiplicity=1 << 4096
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_divisor_effective_parts(oversized)
    assert (
        error.value.errors()[0]["type"]
        == "function_field.divisor_multiplicity_exceeds_envelope"
    )


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
