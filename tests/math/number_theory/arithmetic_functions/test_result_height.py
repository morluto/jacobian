import pytest

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic_functions._models import (
    DirichletConvolutionRequest,
    DirichletInverseRequest,
    MobiusTransformRequest,
    SummatoryFunctionRequest,
)
from jacobian.math.number_theory.arithmetic_functions._operations import (
    compute_dirichlet_convolution,
    compute_dirichlet_inverse,
    compute_mobius_transform,
    compute_summatory_function,
)


def _rational(num: str, den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


def test_summatory_rejects_cross_denominator_growth() -> None:
    power = 32_768
    request = SummatoryFunctionRequest.model_validate(
        {
            "values": [
                _rational("1", format_canonical_integer(2**power)),
                _rational("1", format_canonical_integer(5**power)),
            ]
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_summatory_function(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


def test_convolution_accounts_for_numerator_products() -> None:
    large = "1" + "0" * 20_000
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
                _rational("1", format_canonical_integer(2**power)),
                _rational("1", format_canonical_integer(5**power)),
            ]
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_mobius_transform(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


def test_dirichlet_inverse_propagates_its_recurrence() -> None:
    denominator = "1" + "0" * 20_000
    request = DirichletInverseRequest.model_validate(
        {"values": [_rational("1", denominator), _rational("1")]}
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_dirichlet_inverse(request)
    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic_functions.result_height_exceeded"
    )


@pytest.mark.parametrize(
    "parsed_request",
    [
        SummatoryFunctionRequest.model_validate({"values": [_rational("1", "2")]}),
        MobiusTransformRequest.model_validate({"values": [_rational("1", "2")]}),
        DirichletInverseRequest.model_validate({"values": [_rational("1", "2")]}),
        DirichletConvolutionRequest.model_validate(
            {"f": [_rational("1", "2")], "g": [_rational("1", "3")]}
        ),
    ],
)
def test_ordinary_requests_remain_admitted(parsed_request: object) -> None:
    assert parsed_request is not None
