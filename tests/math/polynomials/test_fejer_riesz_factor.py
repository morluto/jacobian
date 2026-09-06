"""Contract tests for exact rational degree-one Fejer-Riesz conclusions."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.unit_circle import (
    FejerRieszFactored,
    FejerRieszFactorResult,
    FejerRieszNegative,
    FejerRieszZero,
    HermitianLaurentPolynomial,
    HermitianLaurentTerm,
    real_symmetric_degree_one_fejer_riesz_factor,
    verify_real_symmetric_degree_one_fejer_riesz_factor,
)
from jacobian.math.polynomials.unit_circle._tools import TOOLS


def q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def laurent(c0: int, c1: int) -> HermitianLaurentPolynomial:
    return HermitianLaurentPolynomial(
        terms=tuple(
            HermitianLaurentTerm(exponent=exponent, coefficient=q(coefficient))
            for exponent, coefficient in ((-1, c1), (0, c0), (1, c1))
            if coefficient
        )
    )


def factored(c0: int, c1: int) -> FejerRieszFactored:
    conclusion = real_symmetric_degree_one_fejer_riesz_factor(
        laurent(c0, c1)
    ).conclusion
    assert isinstance(conclusion, FejerRieszFactored)
    return conclusion


def test_boundary_factor_is_exact_normalized_and_source_bound() -> None:
    source = laurent(2, -1)
    result = real_symmetric_degree_one_fejer_riesz_factor(source)
    assert result.source == source
    assert isinstance(result.conclusion, FejerRieszFactored)
    factor = result.conclusion.factor
    assert tuple(
        coefficient.coefficients_ascending[0].as_fraction()
        for coefficient in factor.coefficients_ascending
    ) == (Fraction(1), Fraction(-1))
    decoded = FejerRieszFactorResult.model_validate_json(
        result.model_dump_json(), strict=True
    )
    assert verify_real_symmetric_degree_one_fejer_riesz_factor(decoded)


def test_nonrational_factor_uses_one_embedded_exact_field() -> None:
    factor = factored(3, -1).factor
    first, second = factor.coefficients_ascending
    assert factor.embedding_record.embedding.presentation.degree == 2
    assert first.presentation == second.presentation
    assert first.presentation == factor.embedding_record.embedding.presentation
    assert first.coefficients_ascending == (q(0), q(1))
    assert second.coefficients_ascending == (q(1), q(-1))


def test_zero_constant_and_negative_are_distinct_exact_conclusions() -> None:
    zero = real_symmetric_degree_one_fejer_riesz_factor(
        HermitianLaurentPolynomial(terms=())
    )
    assert isinstance(zero.conclusion, FejerRieszZero)

    assert isinstance(
        real_symmetric_degree_one_fejer_riesz_factor(laurent(4, 0)).conclusion,
        FejerRieszFactored,
    )

    negative = real_symmetric_degree_one_fejer_riesz_factor(laurent(1, -1))
    assert isinstance(negative.conclusion, FejerRieszNegative)
    assert negative.conclusion.cosine_witness.as_fraction() == 1
    assert verify_real_symmetric_degree_one_fejer_riesz_factor(negative)


def test_source_carrier_requires_canonical_nonzero_ordered_terms() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        HermitianLaurentPolynomial(
            terms=(
                HermitianLaurentTerm(exponent=1, coefficient=q(1)),
                HermitianLaurentTerm(exponent=-1, coefficient=q(1)),
            )
        )
    with pytest.raises(ValidationError, match="zero Laurent"):
        HermitianLaurentPolynomial(
            terms=(HermitianLaurentTerm(exponent=0, coefficient=q(0)),)
        )


def test_operation_admits_only_real_symmetric_sources() -> None:
    with pytest.raises(OperationDomainValidationError, match="Hermitian"):
        real_symmetric_degree_one_fejer_riesz_factor(
            HermitianLaurentPolynomial(
                terms=(HermitianLaurentTerm(exponent=1, coefficient=q(1)),)
            )
        )


def test_verifier_rejects_reciprocal_and_source_forgeries() -> None:
    result = real_symmetric_degree_one_fejer_riesz_factor(laurent(3, -1))
    payload = json.loads(result.model_dump_json())
    coefficients = payload["conclusion"]["factor"]["coefficients_ascending"]
    coefficients.reverse()
    reciprocal = FejerRieszFactorResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_real_symmetric_degree_one_fejer_riesz_factor(reciprocal)

    payload = json.loads(result.model_dump_json())
    payload["source"] = laurent(4, -1).model_dump(mode="json")
    wrong_source = FejerRieszFactorResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_real_symmetric_degree_one_fejer_riesz_factor(wrong_source)


def test_verifier_rejects_false_negative_witness() -> None:
    result = real_symmetric_degree_one_fejer_riesz_factor(laurent(1, -1))
    payload = json.loads(result.model_dump_json())
    payload["conclusion"]["cosine_witness"] = {"num": "-1", "den": "1"}
    forged = FejerRieszFactorResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_real_symmetric_degree_one_fejer_riesz_factor(forged)


def test_verifier_recognizes_the_exact_embedding_record() -> None:
    result = real_symmetric_degree_one_fejer_riesz_factor(laurent(3, -1))
    payload = json.loads(result.model_dump_json())
    interval = payload["conclusion"]["factor"]["embedding_record"]["isolating_interval"]
    interval["lower"] = {"num": "8", "den": "5"}
    interval["upper"] = {"num": "17", "den": "10"}
    forged = FejerRieszFactorResult.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert not verify_real_symmetric_degree_one_fejer_riesz_factor(forged)


def test_component_growth_is_rejected_before_algebraic_work() -> None:
    huge = 10**64
    source = laurent(2 * huge, -huge)
    with pytest.raises(OperationResourceAdmissionError, match="64-digit"):
        real_symmetric_degree_one_fejer_riesz_factor(source)


def test_native_and_mcp_paths_share_serialized_result() -> None:
    source = laurent(2, -1)
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "polynomial.unit_circle.real_symmetric_degree_one_fejer_riesz_factor.compute"
    )
    native = real_symmetric_degree_one_fejer_riesz_factor(source)
    public = invoke_operation(
        operation.operation_id, source.model_dump(mode="json"), Catalog.open()
    )
    assert public.output == native.model_dump(mode="json")
