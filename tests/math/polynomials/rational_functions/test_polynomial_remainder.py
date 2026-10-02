"""Remainder-only consumers do not inherit an unused primitive's envelope."""

from collections.abc import Callable

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.rational_functions.operations import (
    formal_antiderivative,
    hermite_reduction,
    logarithmic_differential,
    partial_fractions,
    rational_primitive,
    verify_partial_fractions,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(axis: str, degree: int, digits: int, sign: int = 1) -> RationalFunction:
    return RationalFunction(
        variables=(axis,),
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=sign, den=10**digits - 1),
                    exponents=(degree,),
                ),
            )
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(0,)
                ),
            )
        ),
    )


@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("axis", ("x", "t"))
@pytest.mark.parametrize("degree", (1, 63))
@pytest.mark.parametrize("digits", (127, 128))
def test_polynomial_remainder_retains_source_without_unused_primitive(
    axis: str, degree: int, digits: int, sign: int
) -> None:
    source = _polynomial(axis, degree, digits, sign)
    profile = partial_fractions(source)
    assert profile.polynomial_part.variables == source.variables
    assert profile.polynomial_part.polynomial == source.numerator
    assert profile.factors == ()
    assert profile.terms == ()
    assert profile.reconstructed == source
    assert profile.hermite_agreement
    assert profile.hermite_remainder.variables == source.variables
    assert not profile.hermite_remainder.numerator.terms
    decoded = type(profile).model_validate_json(profile.model_dump_json())
    assert decoded == profile
    assert verify_partial_fractions(decoded)

    logarithmic = logarithmic_differential(source)
    assert logarithmic.source == source
    assert logarithmic.terms == ()
    assert logarithmic.reconstructed == profile.hermite_remainder
    assert (
        type(logarithmic).model_validate_json(logarithmic.model_dump_json())
        == logarithmic
    )


@pytest.mark.parametrize(
    "operation", (hermite_reduction, rational_primitive, formal_antiderivative)
)
def test_primitive_producers_keep_genuine_denominator_refusal(
    operation: Callable[[RationalFunction], object],
) -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        operation(_polynomial("x", 1, 128))
    assert error.value.errors()[0]["type"] == "polynomial.hermite_reduction_budget"


@pytest.mark.parametrize("operation", (partial_fractions, logarithmic_differential))
@pytest.mark.parametrize("degree,digits", ((64, 128), (1, 129)))
def test_remainder_only_path_keeps_its_source_admission(
    operation: Callable[[RationalFunction], object], degree: int, digits: int
) -> None:
    source = _polynomial("x", degree, min(digits, 128))
    if digits > 128:
        term = source.numerator.terms[0].model_copy(
            update={"coefficient": CanonicalRational(num=1, den=10**digits - 1)}
        )
        source = source.model_copy(
            update={"numerator": SparseRationalPolynomial(terms=(term,))}
        )
    with pytest.raises(OperationResourceAdmissionError) as error:
        operation(source)
    assert error.value.errors()[0]["type"] == "polynomial.partial_fraction_budget"
