import pytest
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.math.arithmetic_functions._models import (
    DirichletConvolutionRequest,
    DirichletInverseRequest,
    MobiusTransformRequest,
    SummatoryFunctionRequest,
)


def _rational(num: str, den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


def test_summatory_rejects_cross_denominator_growth() -> None:
    power = 32_768
    with pytest.raises(ValidationError, match="summatory function rational height"):
        SummatoryFunctionRequest.model_validate(
            {
                "values": [
                    _rational("1", format_canonical_integer(2**power)),
                    _rational("1", format_canonical_integer(5**power)),
                ]
            }
        )


def test_convolution_accounts_for_numerator_products() -> None:
    large = "1" + "0" * 20_000
    with pytest.raises(ValidationError, match="Dirichlet convolution rational height"):
        DirichletConvolutionRequest.model_validate(
            {"f": [_rational(large)], "g": [_rational(large)]}
        )


def test_mobius_transform_accounts_for_signed_sums() -> None:
    power = 32_768
    with pytest.raises(ValidationError, match="Möbius transform rational height"):
        MobiusTransformRequest.model_validate(
            {
                "values": [
                    _rational("1", format_canonical_integer(2**power)),
                    _rational("1", format_canonical_integer(5**power)),
                ]
            }
        )


def test_dirichlet_inverse_propagates_its_recurrence() -> None:
    denominator = "1" + "0" * 20_000
    with pytest.raises(ValidationError, match="Dirichlet inverse rational height"):
        DirichletInverseRequest.model_validate(
            {"values": [_rational("1", denominator), _rational("1")]}
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
