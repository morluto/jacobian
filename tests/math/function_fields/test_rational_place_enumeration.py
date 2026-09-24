from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.function_fields._models import (
    FunctionFieldPlaceEnumerationRequest,
    PrimeFieldPolynomial,
)
from jacobian.math.function_fields.operations import (
    function_field_rational_places_degree_bounded,
)


def _field(prime: int):
    one = PrimeFieldPolynomial(characteristic=prime, coefficients=(1,))
    from jacobian.math.function_fields._models import (
        FiniteFunctionField,
        PrimeFieldRationalFunction,
    )

    return FiniteFunctionField(
        characteristic=prime,
        defining_polynomial=(
            PrimeFieldRationalFunction(numerator=one, denominator=one),
        ),
    )


def _divides(dividend: tuple[int, ...], divisor: tuple[int, ...], p: int) -> bool:
    remainder = list(dividend)
    inverse_lead = pow(divisor[-1], -1, p)
    while len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        factor = remainder[-1] * inverse_lead % p
        for i, coefficient in enumerate(divisor):
            remainder[shift + i] = (remainder[shift + i] - factor * coefficient) % p
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return not remainder


def _independently_irreducible(poly: tuple[int, ...], p: int) -> bool:
    degree = len(poly) - 1
    for proper_degree in range(1, degree // 2 + 1):
        for lower in product(range(p), repeat=proper_degree):
            divisor = (*lower, 1)
            if _divides(poly, divisor, p):
                return False
    return True


@pytest.mark.parametrize(
    ("prime", "bound", "expected_count"),
    ((2, 5, 15), (3, 3, 15), (5, 2, 16)),
)
def test_complete_rational_place_enumeration_has_independent_factor_checks(
    prime: int, bound: int, expected_count: int
) -> None:
    field = _field(prime)
    result = function_field_rational_places_degree_bounded(
        FunctionFieldPlaceEnumerationRequest(field=field, maximum_degree=bound)
    )

    assert result.field == field
    assert len(result.places) == expected_count
    assert sum(place.kind == "INFINITE" for place in result.places) == 1
    finite = [place for place in result.places if place.kind == "FINITE"]
    assert all(place.field == field for place in result.places)
    assert all(place.degree <= bound for place in result.places)
    keys = [
        (
            place.degree,
            place.kind,
            ()
            if place.prime_polynomial is None
            else place.prime_polynomial.coefficients,
        )
        for place in result.places
    ]
    assert keys == sorted(keys)
    for place in finite:
        assert place.prime_polynomial is not None
        assert place.prime_polynomial.coefficients[-1] == 1
        assert _independently_irreducible(place.prime_polynomial.coefficients, prime)


def test_rational_place_enumeration_contains_known_degree_two_polynomials() -> None:
    result = function_field_rational_places_degree_bounded(
        FunctionFieldPlaceEnumerationRequest(field=_field(2), maximum_degree=2)
    )

    degree_two = {
        place.prime_polynomial.coefficients
        for place in result.places
        if place.prime_polynomial is not None and place.degree == 2
    }
    assert degree_two == {(1, 1, 1)}


def test_candidate_envelope_rejects_before_enumeration() -> None:
    # 257 + 257**2 candidates exceed the bound even though the degree field is
    # structurally valid and the request itself is tiny.
    request = FunctionFieldPlaceEnumerationRequest(field=_field(257), maximum_degree=2)

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_rational_places_degree_bounded(request)

    assert error.value.errors()[0]["type"] == (
        "function_field.place_enumeration_candidates_exceed_envelope"
    )


def test_degree_one_candidate_boundary_is_accepted_completely() -> None:
    result = function_field_rational_places_degree_bounded(
        FunctionFieldPlaceEnumerationRequest(field=_field(257), maximum_degree=1)
    )

    assert len(result.places) == 258
    assert sum(place.kind == "FINITE" for place in result.places) == 257
    assert sum(place.kind == "INFINITE" for place in result.places) == 1


def test_extension_field_is_rejected_instead_of_partially_enumerated() -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.function_fields._models import (
        FiniteFunctionField,
        PrimeFieldRationalFunction,
    )

    p = 2

    def rational(coefficients: tuple[int, ...]) -> PrimeFieldRationalFunction:
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(characteristic=p, coefficients=coefficients),
            denominator=PrimeFieldPolynomial(characteristic=p, coefficients=(1,)),
        )

    extension = FiniteFunctionField(
        characteristic=p,
        defining_polynomial=(rational((0, 1)), rational((1,)), rational((1,))),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_rational_places_degree_bounded(
            FunctionFieldPlaceEnumerationRequest(field=extension, maximum_degree=1)
        )

    assert error.value.errors()[0]["type"] == (
        "function_field.place_extension_unsupported"
    )
