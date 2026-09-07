import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic_functions._models import (
    DirichletConvolutionRequest,
    DirichletInverseRequest,
    MobiusTransformRequest,
    SummatoryFunctionRequest,
)
from jacobian.math.number_theory.arithmetic_functions._tools import (
    compute_dirichlet_convolution,
    compute_dirichlet_inverse,
    compute_mobius_transform,
    compute_summatory_function,
)


def _rational(num: int, den: int = 1) -> dict[str, int]:
    return {"num": num, "den": den}


def test_summatory_rejects_cross_denominator_growth() -> None:
    power = 32_768
    request = SummatoryFunctionRequest.model_validate(
        {
            "values": [
                _rational(1, 2**power),
                _rational(1, 5**power),
            ]
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_summatory_function(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


def test_convolution_accounts_for_numerator_products() -> None:
    large = 10**20_000
    request = DirichletConvolutionRequest.model_validate(
        {"f": [_rational(large)], "g": [_rational(large)]}
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_dirichlet_convolution(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


def test_mobius_transform_accounts_for_signed_sums() -> None:
    power = 32_768
    request = MobiusTransformRequest.model_validate(
        {
            "values": [
                _rational(1, 2**power),
                _rational(1, 5**power),
            ]
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_mobius_transform(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


def test_dirichlet_inverse_propagates_its_recurrence() -> None:
    denominator = 10**20_000
    request = DirichletInverseRequest.model_validate(
        {"values": [_rational(1, denominator), _rational(1)]}
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_dirichlet_inverse(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


@pytest.mark.parametrize(
    "parsed_request",
    [
        SummatoryFunctionRequest.model_validate({"values": [_rational(1, 2)]}),
        MobiusTransformRequest.model_validate({"values": [_rational(1, 2)]}),
        DirichletInverseRequest.model_validate({"values": [_rational(1, 2)]}),
        DirichletConvolutionRequest.model_validate(
            {"f": [_rational(1, 2)], "g": [_rational(1, 3)]}
        ),
    ],
)
def test_ordinary_requests_remain_admitted(parsed_request: object) -> None:
    assert parsed_request is not None


@pytest.mark.parametrize("length", [900, 1000])
def test_summatory_shared_denominator_prefixes(length: int) -> None:
    from fractions import Fraction

    request = SummatoryFunctionRequest.model_validate(
        {"values": [_rational(1, 10**32)] * length}
    )
    result = compute_summatory_function(request)
    assert tuple(v.as_fraction() for v in result.values) == tuple(
        Fraction(k, 10**32) for k in range(1, length + 1)
    )


def test_summatory_alternating_simplex_moments() -> None:
    from fractions import Fraction
    from itertools import accumulate
    from math import factorial

    # Independent repeated integer convolution defines the ninth power.
    order = 257
    factorials = [factorial(j) for j in range(order)]
    coefficients = [1] + [0] * (order - 1)
    for _ in range(9):
        coefficients = [
            sum(coefficients[i] * factorials[k - i] for i in range(k + 1))
            for k in range(order)
        ]
    values = [
        Fraction((-1) ** j * coefficients[j], factorial(j + 8)) for j in range(order)
    ]
    request = SummatoryFunctionRequest.model_validate(
        {"values": [_rational(v.numerator, v.denominator) for v in values]}
    )
    result = compute_summatory_function(request)
    assert [v.as_fraction() for v in result.values] == list(accumulate(values))


def test_one_digit_inverse_can_have_multidigit_coefficients() -> None:
    from fractions import Fraction

    values = [_rational(2)] + [_rational(9)] * 127
    request = DirichletInverseRequest.model_validate({"values": values})
    result = compute_dirichlet_inverse(request)
    assert any(abs(value.num) >= 10 for value in result.values)
    convolution = compute_dirichlet_convolution(
        DirichletConvolutionRequest(f=request.values, g=result.values)
    )
    assert (
        tuple(value.as_fraction() for value in convolution.values)
        == (Fraction(1),) + (Fraction(0),) * 127
    )
