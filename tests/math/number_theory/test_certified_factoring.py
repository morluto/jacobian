"""Tests for subexponential certified factoring and Pratt certificates."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from tests.math.number_theory._validation import expect_validation

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._certification_models import (
    CertifiedFactor,
    CertifiedFactorizationRequest,
    CertifiedFactorizationResult,
    PrattCertificateFactor,
    PrattCertificateNode,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.math.number_theory._factorization_kernels import (
    compute_pratt_certificate,
    factorize_certified,
    verify_certified_factorization,
    verify_pratt_certificate,
    verify_primality_certificate,
)

# ---------------------------------------------------------------------------
# Certified factorization
# ---------------------------------------------------------------------------


def test_semiprime_factors_completely_with_certificates() -> None:
    result = factorize_certified(CertifiedFactorizationRequest(value="10403"))
    assert result.status == "COMPLETE"
    assert result.value == "10403"
    product = math.prod(
        parse_canonical_integer(f.prime) ** f.exponent for f in result.factors
    )
    assert product == 10403
    assert verify_certified_factorization(result)


def test_large_semiprime_factors_with_subexponential_methods() -> None:
    value = str(10000000019 * 10000000033)
    result = factorize_certified(CertifiedFactorizationRequest(value=value))
    assert result.status == "COMPLETE"
    product = math.prod(
        parse_canonical_integer(f.prime) ** f.exponent for f in result.factors
    )
    assert product == int(value)
    assert len(result.factors) == 2
    assert verify_certified_factorization(result)


def test_perfect_power_factors_with_certificates() -> None:
    result = factorize_certified(CertifiedFactorizationRequest(value=str(2**20)))
    assert result.status == "COMPLETE"
    assert len(result.factors) == 1
    assert result.factors[0].prime == "2"
    assert result.factors[0].exponent == 20
    assert verify_certified_factorization(result)


def test_digit_bound_rejects_oversized_input() -> None:
    with pytest.raises(ValidationError):
        CertifiedFactorizationRequest(value="1" + "0" * 80)


def test_rejects_value_below_two() -> None:
    with expect_validation("number_theory."):
        CertifiedFactorizationRequest(value="1")
    with expect_validation("number_theory."):
        CertifiedFactorizationRequest(value="0")


def test_result_binds_product_and_ordering() -> None:
    result = factorize_certified(CertifiedFactorizationRequest(value="360"))
    assert result.status == "COMPLETE"
    primes = [int(f.prime) for f in result.factors]
    assert primes == sorted(primes)
    assert primes == [2, 3, 5]


@pytest.mark.parametrize("value", ["6", "-2", "1"])
def test_complete_worker_result_remains_an_unverified_claim(value: str) -> None:
    certificate = PrattCertificateNode(prime="2")
    claim = CertifiedFactorizationResult.model_validate(
        {
            "status": "COMPLETE",
            "value": value,
            "factors": [
                CertifiedFactor(prime="2", exponent=1, certificate=certificate)
            ],
        }
    )
    assert not verify_certified_factorization(claim)


# ---------------------------------------------------------------------------
# Pratt primality certificate
# ---------------------------------------------------------------------------


def test_pratt_certificate_for_known_prime() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="101"))
    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert result.certificate.prime == "101"
    assert verify_primality_certificate(result)


def test_pratt_certificate_for_base_case_prime_two() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="2"))
    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert result.certificate.prime == "2"
    assert result.certificate.witness is None
    assert result.certificate.factors == ()


def test_pratt_certificate_for_large_prime() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="1000000007"))
    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert verify_primality_certificate(result)


def test_pratt_certificate_reports_composite() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="9"))
    assert result.status == "COMPOSITE"
    assert result.certificate is None


def test_pratt_certificate_rejects_value_below_two() -> None:
    with expect_validation("number_theory."):
        PrimalityCertificateRequest(value="1")
    with expect_validation("number_theory."):
        PrimalityCertificateRequest(value="0")


def test_pratt_certificate_rejects_oversized_input() -> None:
    with pytest.raises(ValidationError):
        PrimalityCertificateRequest(value="1" + "0" * 80)


def test_certified_result_remains_an_unverified_claim() -> None:
    valid = compute_pratt_certificate(PrimalityCertificateRequest(value="101"))
    assert valid.certificate is not None
    claim = PrimalityCertificateResult(
        status="CERTIFIED",
        value="103",
        certificate=valid.certificate,
    )
    assert not verify_primality_certificate(claim)


def test_certified_result_without_certificate_remains_an_unverified_claim() -> None:
    assert not verify_primality_certificate(
        PrimalityCertificateResult(status="CERTIFIED", value="101")
    )


def test_composite_result_with_certificate_remains_an_unverified_claim() -> None:
    claim = PrimalityCertificateResult(
        status="COMPOSITE",
        value="9",
        certificate=PrattCertificateNode(prime="2"),
    )
    assert not verify_primality_certificate(claim)


def test_unwitnessed_composite_status_does_not_reenter_a_primality_backend() -> None:
    """The owner kernel, rather than result validation, establishes COMPOSITE."""

    result = PrimalityCertificateResult(status="COMPOSITE", value="101")

    assert result.certificate is None


# ---------------------------------------------------------------------------
# math.find and math.run discovery / execution
# ---------------------------------------------------------------------------


def test_operations_are_discoverable_via_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {t.operation_id for t in BUILTIN_TOOLS}
    assert "integer.factor.certified_compute" in ids
    assert "integer.primality.certificate.compute" in ids


def test_serialized_pratt_node_is_a_claim_not_a_primality_check() -> None:
    candidate = PrattCertificateNode.model_validate_json(
        '{"prime":"9","witness":"2","factors":[{"prime":"2","exponent":1,"certificate":{"prime":"2"}}]}'
    )
    assert not verify_pratt_certificate(candidate)
    malformed_factorization = PrattCertificateNode(
        prime="7",
        witness="3",
        factors=(
            PrattCertificateFactor(
                prime="2", exponent=1, certificate=PrattCertificateNode(prime="2")
            ),
        ),
    )
    assert not verify_pratt_certificate(malformed_factorization)
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="101"))
    assert result.certificate is not None
    decoded = PrimalityCertificateResult.model_validate_json(result.model_dump_json())
    assert decoded.certificate is not None
    assert verify_primality_certificate(decoded)
