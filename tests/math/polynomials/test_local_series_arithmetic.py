import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series import TruncatedLaurentWindow
from jacobian.math.polynomials.local_series.arithmetic_models import (
    LaurentDeramifyResult,
)
from jacobian.math.polynomials.local_series.operations import (
    add,
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
    integrated = integral(source)
    assert integrated.laurent_part is not None
    recovered = derivative(integrated.laurent_part)
    assert recovered.coefficients[1].as_fraction() == 2


def test_ramification_multiplies_exponents() -> None:
    result = ramify(_s(), 2)
    assert result.valuation_lower == -2
    assert result.precision == 4


def test_principal_split_does_not_double_count_regular_constant() -> None:
    regular = TruncatedLaurentWindow(
        variable="t",
        center=CanonicalRational(num=0, den=1),
        valuation_lower=0,
        precision=2,
        coefficients=(CanonicalRational(num=7, den=1), CanonicalRational(num=3, den=1)),
    )
    split = principal_part(regular)
    assert split.principal_part is not None
    assert split.principal_part.coefficients[0].num == 0
    assert split.regular_part is not None
    assert split.regular_part == regular


def test_deramify_rejects_zero_and_does_not_read_unknown_lower_tail() -> None:
    with pytest.raises(OperationDomainValidationError):
        deramify(_s(), 0)
    result = deramify(_s(), 2)
    assert result.status == "NOT_IN_IMAGE_OF_RAMIFICATION"
    assert result.offending_exponents == (-1, 1)


def test_power_one_preserves_the_known_window() -> None:
    assert power(_s(), 1) == _s()


def test_large_exact_coefficients_are_refused_before_materialization() -> None:
    coefficient = CanonicalRational(num=10**4095, den=1)
    series = TruncatedLaurentWindow(
        valuation_lower=0, precision=1, coefficients=(coefficient,)
    )
    with pytest.raises(OperationResourceAdmissionError):
        add(series, series)


def test_negative_only_window_does_not_fabricate_regular_prefix() -> None:
    series = TruncatedLaurentWindow(
        valuation_lower=-1,
        precision=0,
        coefficients=(CanonicalRational(num=1, den=1),),
    )
    assert principal_part(series).regular_part is None
    assert integral(series).laurent_part is None


def test_deramify_success_state_requires_result() -> None:
    with pytest.raises(ValidationError):
        LaurentDeramifyResult(status="IN_IMAGE_OF_RAMIFICATION")
