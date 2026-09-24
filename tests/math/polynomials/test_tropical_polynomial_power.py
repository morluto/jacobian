from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.tropical._models import PolynomialPowerRequest
from jacobian.math.polynomials.tropical._tools import TOOLS
from jacobian.math.polynomials.tropical.operations import (
    tropical_polynomial_evaluate,
    tropical_polynomial_power,
)
from jacobian.math.polynomials.tropical.values import (
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    TropicalVector,
)


def _polynomial(convention: str, exponents: tuple[int, ...]) -> TropicalPolynomial:
    semiring = TropicalSemiring(convention=convention, base="ZZ")  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(exponent,),
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(0, 1),
                ),
            )
            for exponent in exponents
        ),
    )


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_formal_cube_has_exact_binomial_support_and_evaluates_compatibly(
    convention: str,
) -> None:
    polynomial = _polynomial(convention, (0, 1))
    result = tropical_polynomial_power(polynomial, 3)
    assert tuple(term.exponents for term in result.terms) == ((0,), (1,), (2,), (3,))
    assert tuple(term.coefficient.value.num for term in result.terms) == (0, 0, 0, 0)

    point_value = 2
    point = TropicalVector(
        semiring=polynomial.semiring,
        axis=("x",),
        entries=(
            TropicalScalar(
                semiring=polynomial.semiring,
                kind="FINITE",
                value=CanonicalRational.from_integer_ratio(point_value, 1),
            ),
        ),
    )
    actual, active = tropical_polynomial_evaluate(result, point)
    expected = 0 if convention == "MIN_PLUS" else 6
    assert actual.value is not None and actual.value.num == expected
    assert active == (((0,),) if convention == "MIN_PLUS" else ((3,),))


def test_zero_power_is_the_polynomial_multiplicative_identity() -> None:
    polynomial = _polynomial("MAX_PLUS", (2,))
    result = tropical_polynomial_power(polynomial, 0)
    assert result.semiring == polynomial.semiring
    assert result.variables == polynomial.variables
    assert len(result.terms) == 1
    assert result.terms[0].exponents == (0,)
    assert result.terms[0].coefficient.value is not None
    assert result.terms[0].coefficient.value.num == 0


def test_power_preflights_exponent_and_intermediate_term_growth() -> None:
    with pytest.raises(OperationResourceAdmissionError) as exponent_error:
        tropical_polynomial_power(_polynomial("MIN_PLUS", (100,)), 16)
    assert (
        exponent_error.value.errors()[0]["type"]
        == "tropical.polynomial_power_exponent_bound"
    )

    dense = _polynomial("MIN_PLUS", tuple(range(512)))
    with pytest.raises(OperationResourceAdmissionError) as term_error:
        tropical_polynomial_power(dense, 2)
    assert (
        term_error.value.errors()[0]["type"] == "tropical.polynomial_power_term_bound"
    )


def test_power_preflights_dimension_weighted_convolution_work():
    semiring = TropicalSemiring(convention="MIN_PLUS", base="ZZ")
    variables = tuple(f"x{index}" for index in range(128))
    coefficient = TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(0, 1),
    )
    dense_high_dimensional = TropicalPolynomial(
        semiring=semiring,
        variables=variables,
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(index, *(0 for _ in range(127))),
                coefficient=coefficient,
            )
            for index in range(512)
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_power(dense_high_dimensional, 2)

    assert error.value.errors()[0]["type"] == "tropical.polynomial_power_work_bound"


def test_request_enforces_the_published_power_bound() -> None:
    with pytest.raises(ValueError):
        PolynomialPowerRequest(polynomial=_polynomial("MIN_PLUS", (0,)), exponent=17)


def test_catalogued_power_example_executes() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "tropical.polynomial.power.compute"
    )
    result = invoke_operation(tool.operation_id, tool.examples[0].input, Catalog.open())
    terms = result.output["result"]["terms"]
    assert tuple(tuple(term["exponents"]) for term in terms) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )
    assert tuple(term["coefficient"]["value"] for term in terms) == tuple(
        {"num": "0", "den": "1"} for _ in range(4)
    )
