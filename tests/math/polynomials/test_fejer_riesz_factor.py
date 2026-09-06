"""Contract tests for the bounded exact scalar Fejer-Riesz operation."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.unit_circle import (
    FejerRieszFactorResult,
    HermitianLaurentPolynomial,
    HermitianLaurentTerm,
    fejer_riesz_factor,
    verify_fejer_riesz_factor,
)
from jacobian.math.polynomials.unit_circle._tools import TOOLS


def q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def laurent(c0: int, c1: int) -> HermitianLaurentPolynomial:
    return HermitianLaurentPolynomial(
        terms=tuple(
            HermitianLaurentTerm(exponent=exponent, coefficient=q(coefficient))
            for exponent, coefficient in ((-1, c1), (0, c0), (1, c1))
            if coefficient or exponent == 0
        )
    )


def test_boundary_zero_factor_is_exact_and_source_bound() -> None:
    result = fejer_riesz_factor(laurent(2, -1))
    assert result.zero_input is False
    assert result.field_degree == 1
    assert len(result.factor_coefficients) == 2
    assert tuple(
        coefficient.element.coefficients_ascending[0].as_fraction()
        for coefficient in result.factor_coefficients
    ) == (Fraction(1), Fraction(-1))
    assert verify_fejer_riesz_factor(
        FejerRieszFactorResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
    )


def test_nonrational_factor_uses_one_embedded_exact_field() -> None:
    result = fejer_riesz_factor(laurent(3, -1))
    first, second = result.factor_coefficients
    assert result.field_degree == 2
    assert first.element.presentation == second.element.presentation
    assert first.embedding_record == second.embedding_record
    assert first.element.coefficients_ascending == (q(0), q(1))
    assert second.element.coefficients_ascending == (q(1), q(-1))


def test_zero_and_constant_inputs() -> None:
    zero = fejer_riesz_factor(HermitianLaurentPolynomial(terms=()))
    assert zero.zero_input is True
    assert len(zero.factor_coefficients) == 1
    assert zero.factor_coefficients[0].element.coefficients_ascending == (q(0),)

    constant = fejer_riesz_factor(laurent(4, 0))
    assert constant.zero_input is False
    assert tuple(
        coefficient.element.coefficients_ascending[0].as_fraction()
        for coefficient in constant.factor_coefficients
    ) == (Fraction(2), Fraction(0))


def test_admission_rejects_negative_and_nonhermitian_inputs() -> None:
    with pytest.raises(OperationDomainValidationError, match="negative"):
        fejer_riesz_factor(laurent(-1, 0))
    with pytest.raises(OperationDomainValidationError, match="Hermitian"):
        fejer_riesz_factor(
            HermitianLaurentPolynomial(
                terms=(HermitianLaurentTerm(exponent=1, coefficient=q(1)),)
            )
        )


def test_serialized_forgery_is_structural_but_fails_verification() -> None:
    result = fejer_riesz_factor(laurent(3, -1))
    payload = json.loads(result.model_dump_json())
    payload["field_degree"] = 1
    forged = FejerRieszFactorResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert forged.field_degree == 1
    assert not verify_fejer_riesz_factor(forged)


def test_native_and_mcp_paths_share_serialized_result() -> None:
    source = laurent(2, -1)
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "polynomial.unit_circle.fejer_riesz_factor.compute"
    )
    native = fejer_riesz_factor(source)
    public = invoke_operation(
        operation.operation_id, source.model_dump(mode="json"), Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")
