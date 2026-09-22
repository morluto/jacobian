import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.local_series import TruncatedLaurentWindow
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
    principal_part,
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


def test_principal_and_regular_windows_are_disjoint() -> None:
    source = TruncatedLaurentWindow(
        valuation_lower=0,
        precision=2,
        coefficients=(CanonicalRational(num=7, den=1), CanonicalRational(num=3, den=1)),
    )
    result = principal_part(source)
    assert result.principal_part.valuation_lower == 0
    assert result.principal_part.precision == 0
    assert result.principal_part.coefficients == ()
    assert result.regular_part == source


def test_principal_split_handles_a_pure_pole() -> None:
    source = TruncatedLaurentWindow(
        valuation_lower=-4,
        precision=-1,
        coefficients=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=2, den=1),
            CanonicalRational(num=3, den=1),
        ),
    )
    result = principal_part(source)
    assert result.principal_part == source
    assert result.regular_part.valuation_lower == 0
    assert result.regular_part.precision == 0
    assert result.regular_part.coefficients == ()


def test_deramify_does_not_invent_unknown_negative_exponents() -> None:
    result = deramify(
        TruncatedLaurentWindow(
            valuation_lower=-1,
            precision=2,
            coefficients=(CanonicalRational(num=0, den=1),) * 3,
        ),
        2,
    )
    assert result.status == "IN_IMAGE_OF_RAMIFICATION"
    assert result.result is not None
    assert result.result.valuation_lower == 0
    assert result.result.precision == 1


def test_power_one_preserves_the_known_source_prefix() -> None:
    assert power(_s(), 1) == _s()


def test_native_admission_rejects_invalid_ramification() -> None:
    with pytest.raises(OperationDomainValidationError):
        deramify(_s(), 0)


def test_deramify_result_rejects_contradictory_payload() -> None:
    with pytest.raises(ValueError):
        LaurentDeramifyResult(status="IN_IMAGE_OF_RAMIFICATION")
