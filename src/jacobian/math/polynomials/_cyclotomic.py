"""Exact cyclotomic polynomials through python-flint."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS

MAX_CYCLOTOMIC_INDEX = 100_000


class CyclotomicRequest(StrictModel):
    index: StrictInt = Field(ge=1, le=MAX_CYCLOTOMIC_INDEX)


class CyclotomicResult(StrictModel):
    source_index: StrictInt = Field(ge=1, le=MAX_CYCLOTOMIC_INDEX)
    totient: StrictInt = Field(ge=1, le=MAX_POLYNOMIAL_TERMS - 1)
    polynomial: IntegerPolynomial


def cyclotomic(request: CyclotomicRequest) -> CyclotomicResult:
    from flint import fmpz, fmpz_poly

    degree = int(fmpz(request.index).euler_phi())
    if degree + 1 > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.output_degree_bound",
            message=(
                f"cyclotomic degree {degree} exceeds the output degree bound "
                f"{MAX_POLYNOMIAL_TERMS - 1}"
            ),
        )
    polynomial = fmpz_poly.cyclotomic(request.index)
    return CyclotomicResult(
        source_index=request.index,
        totient=degree,
        polynomial=IntegerPolynomial(
            coefficients=tuple(int(value) for value in reversed(polynomial.coeffs()))
        ),
    )
