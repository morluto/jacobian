from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.local_series import TruncatedLaurentWindow
from jacobian.math.polynomials.local_series.operations import (
    derivative, integral, inverse, multiply, ramify, residue,
)

def _s() -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t", center=CanonicalRational(num=0, den=1),
        valuation_lower=-1, precision=2,
        coefficients=(CanonicalRational(num=1, den=1), CanonicalRational(num=2, den=1), CanonicalRational(num=3, den=1)),
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
