"""Catalog boundary tests for exact unit-circle operations."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.unit_circle import (
    HermitianLaurentPolynomial,
    HermitianLaurentTerm,
    UnitCircleArcEnergyRequest,
    real_symmetric_degree_one_fejer_riesz_factor,
    unit_circle_arc_energy,
)
from jacobian.math.polynomials.unit_circle._tools import TOOLS
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def test_arc_energy_native_and_catalog_paths_share_serialized_result() -> None:
    request = UnitCircleArcEnergyRequest(
        polynomial=RationalPolynomial(
            variables=("z",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(coefficient=_q(1), exponents=(1,)),
                    RationalPolynomialTerm(coefficient=_q(1), exponents=(0,)),
                )
            ),
        ),
        start_turn=_q(Fraction(-1, 4)),
        end_turn=_q(Fraction(1, 4)),
    )
    native = unit_circle_arc_energy(
        request.polynomial, request.start_turn, request.end_turn
    )
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.unit_circle.arc_energy.compute"
    )
    public = invoke_operation(
        operation.operation_id, request.model_dump(mode="json"), Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")


def test_fejer_riesz_native_and_catalog_paths_share_serialized_result() -> None:
    source = HermitianLaurentPolynomial(
        terms=(
            HermitianLaurentTerm(exponent=-1, coefficient=_q(-1)),
            HermitianLaurentTerm(exponent=0, coefficient=_q(2)),
            HermitianLaurentTerm(exponent=1, coefficient=_q(-1)),
        )
    )
    native = real_symmetric_degree_one_fejer_riesz_factor(source)
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "polynomial.unit_circle.real_symmetric_degree_one_fejer_riesz_factor.compute"
    )
    public = invoke_operation(
        operation.operation_id, source.model_dump(mode="json"), Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")
