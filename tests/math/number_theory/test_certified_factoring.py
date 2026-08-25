"""Tests for subexponential certified factoring and Pratt certificates."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from tests.math.number_theory._validation import expect_validation

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._factorization_kernels import (
    compute_pratt_certificate,
    factorize_certified,
)
from jacobian.math.number_theory._models import (
    CertifiedFactorizationRequest,
    PrattCertificateNode,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)

# ---------------------------------------------------------------------------
# Independent Pratt certificate verifier (mirror of the mathematical invariant)
# ---------------------------------------------------------------------------


def verify_pratt(node: PrattCertificateNode) -> bool:
    """Recursively verify one Pratt certificate node without trusting SymPy."""
    prime = parse_canonical_integer(node.prime)
    if prime == 2:
        return node.witness is None and not node.sub_certificates
    if prime < 2 or node.witness is None:
        return False
    witness = parse_canonical_integer(node.witness)
    if pow(witness, prime - 1, prime) != 1:
        return False
    sub_primes = {parse_canonical_integer(sub.prime) for sub in node.sub_certificates}
    expected_factors = set()
    for sub in node.sub_certificates:
        q = parse_canonical_integer(sub.prime)
        expected_factors.add(q)
        if pow(witness, (prime - 1) // q, prime) == 1:
            return False
        if not verify_pratt(sub):
            return False
    from sympy import factorint

    return set(factorint(prime - 1).keys()) == sub_primes


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
    assert all(verify_pratt(f.certificate) for f in result.factors)


def test_large_semiprime_factors_with_subexponential_methods() -> None:
    value = str(10000000019 * 10000000033)
    result = factorize_certified(CertifiedFactorizationRequest(value=value))
    assert result.status == "COMPLETE"
    product = math.prod(
        parse_canonical_integer(f.prime) ** f.exponent for f in result.factors
    )
    assert product == int(value)
    assert len(result.factors) == 2
    assert all(verify_pratt(f.certificate) for f in result.factors)


def test_perfect_power_factors_with_certificates() -> None:
    result = factorize_certified(CertifiedFactorizationRequest(value=str(2**20)))
    assert result.status == "COMPLETE"
    assert len(result.factors) == 1
    assert result.factors[0].prime == "2"
    assert result.factors[0].exponent == 20
    assert verify_pratt(result.factors[0].certificate)


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


# ---------------------------------------------------------------------------
# Pratt primality certificate
# ---------------------------------------------------------------------------


def test_pratt_certificate_for_known_prime() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="101"))
    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert result.certificate.prime == "101"
    assert verify_pratt(result.certificate)


def test_pratt_certificate_for_base_case_prime_two() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="2"))
    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert result.certificate.prime == "2"
    assert result.certificate.witness is None
    assert result.certificate.sub_certificates == ()


def test_pratt_certificate_for_large_prime() -> None:
    result = compute_pratt_certificate(PrimalityCertificateRequest(value="1000000007"))
    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert verify_pratt(result.certificate)


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


def test_certified_result_rejects_non_matching_certificate() -> None:
    with expect_validation("number_theory."):
        PrimalityCertificateResult(
            status="CERTIFIED",
            value="101",
            certificate=PrattCertificateNode(
                prime="103",
                witness="5",
                sub_certificates=(
                    PrattCertificateNode(prime="2"),
                    PrattCertificateNode(
                        prime="3",
                        witness="2",
                        sub_certificates=(PrattCertificateNode(prime="2"),),
                    ),
                    PrattCertificateNode(
                        prime="17",
                        witness="3",
                        sub_certificates=(PrattCertificateNode(prime="2"),),
                    ),
                ),
            ),
        )


def test_certified_result_rejects_certificate_without_prime() -> None:
    with expect_validation("number_theory."):
        PrimalityCertificateResult(status="CERTIFIED", value="101")


def test_composite_result_rejects_certificate() -> None:
    with expect_validation("number_theory."):
        PrimalityCertificateResult(
            status="COMPOSITE",
            value="9",
            certificate=PrattCertificateNode(prime="2"),
        )


# ---------------------------------------------------------------------------
# math.find and math.run discovery / execution
# ---------------------------------------------------------------------------


def test_operations_are_discoverable_via_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {t.operation_id for t in BUILTIN_TOOLS}
    assert "integer.factor.certified_compute" in ids
    assert "integer.primality.certificate.compute" in ids
