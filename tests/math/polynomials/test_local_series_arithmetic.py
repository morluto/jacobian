import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.local_series import TruncatedLaurentWindow
from jacobian.math.polynomials.local_series.arithmetic import principal_part
from jacobian.math.polynomials.local_series.arithmetic_models import (
    LaurentDeramifyResult,
)
from jacobian.math.polynomials.local_series.operations import (
    deramify,
    derivative,
    integral,
    inverse,
    multiply,
    power,
    ramify,
    residue,
)


def _s() -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        center=CanonicalRational(num=0, den=1),
        valuation_lower=-1,
        precision=2,
        coefficients=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=2, den=1),
            CanonicalRational(num=3, den=1),
        ),
    )


def test_laurent_arithmetic_preserves_exact_residual_prefixes() -> None:
    source = _s()
    inverse_prefix = inverse(source)
    product = multiply(source, inverse_prefix, output_precision=2)
    assert product.coefficients[0].as_fraction() == 1
    assert residue(source).residue.num == 1
    recovered = derivative(integral(source).laurent_part)
    assert recovered.coefficients[1].as_fraction() == 2


def test_ramification_multiplies_exponents() -> None:
    result = ramify(_s(), 2)
    assert result.valuation_lower == -2
    assert result.precision == 4


def test_principal_and_regular_parts_are_disjoint() -> None:
    result = principal_part(_s())
    assert (result.principal_part.valuation_lower, result.principal_part.precision) == (
        -1,
        0,
    )
    assert (result.regular_part.valuation_lower, result.regular_part.precision) == (
        0,
        2,
    )
    assert result.regular_part.coefficients[0].as_fraction() == 2

    regular_source = TruncatedLaurentWindow(
        valuation_lower=0,
        precision=2,
        coefficients=(CanonicalRational(num=1, den=1), CanonicalRational(num=2, den=1)),
    )
    regular_split = principal_part(regular_source)
    assert regular_split.principal_part.coefficients == ()
    assert regular_split.regular_part == regular_source


def test_deramify_does_not_invent_unknown_lower_tail() -> None:
    source = TruncatedLaurentWindow(
        valuation_lower=-1,
        precision=2,
        coefficients=(
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=5, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    result = deramify(source, 2)
    assert result.status == "IN_IMAGE_OF_RAMIFICATION"
    assert result.result is not None
    assert (result.result.valuation_lower, result.result.precision) == (0, 1)
    assert result.result.coefficients[0].as_fraction() == 5


def test_power_one_preserves_the_known_source_window() -> None:
    assert power(_s(), 1) == _s()


def test_deramify_native_admission_rejects_zero() -> None:
    with pytest.raises(OperationDomainValidationError):
        deramify(_s(), 0)


def test_contradictory_deramification_state_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LaurentDeramifyResult(status="IN_IMAGE_OF_RAMIFICATION")
    with pytest.raises(ValidationError):
        LaurentDeramifyResult.model_validate_json(
            '{"status":"IN_IMAGE_OF_RAMIFICATION","result":null}'
        )
