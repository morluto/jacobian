"""Tests for certified integer factoring with Pratt primality certificates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sympy import isprime, nextprime

from jacobian.contracts.certified_factoring import (
    CertifiedFactorRequest,
    CertifiedFactorResult,
    PrattCertificateNode,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.domains.certified_factoring.operations import (
    compute_certified_factor,
    compute_primality_certificate,
)
from jacobian.math.certified_factoring import (
    build_pratt_certificate,
    verify_pratt_certificate,
)
from jacobian.math.certified_factoring.operations import PrattCertificate


# ---------------------------------------------------------------------------
# Pratt certificate construction and verification (math kernel)
# ---------------------------------------------------------------------------


class TestPrattCertificateConstruction:
    """Pratt certificate construction and verification for various primes."""

    @pytest.mark.parametrize(
        "p",
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61],
    )
    def test_small_primes_verify(self, p: int) -> None:
        cert = build_pratt_certificate(p)
        assert verify_pratt_certificate(cert) is True

    def test_base_case_p2(self) -> None:
        cert = build_pratt_certificate(2)
        assert cert.prime == 2
        assert cert.witness is None
        assert cert.cofactor_factors == ()
        assert cert.cofactor_certificates == ()

    def test_prime_3(self) -> None:
        cert = build_pratt_certificate(3)
        assert cert.prime == 3
        assert cert.witness is not None
        assert cert.cofactor_factors == ((2, 1),)
        assert len(cert.cofactor_certificates) == 1
        assert cert.cofactor_certificates[0].prime == 2

    def test_prime_5(self) -> None:
        # 5 - 1 = 4 = 2^2
        cert = build_pratt_certificate(5)
        assert cert.prime == 5
        assert cert.witness is not None
        assert cert.cofactor_factors == ((2, 2),)
        assert len(cert.cofactor_certificates) == 1

    def test_prime_17(self) -> None:
        # 17 - 1 = 16 = 2^4
        cert = build_pratt_certificate(17)
        assert cert.prime == 17
        assert cert.witness is not None
        assert cert.cofactor_factors == ((2, 4),)
        assert len(cert.cofactor_certificates) == 1

    def test_prime_997(self) -> None:
        # 997 - 1 = 996 = 2^2 * 3 * 83
        cert = build_pratt_certificate(997)
        assert cert.prime == 997
        assert cert.witness is not None
        # cofactor_factors should list 2^2, 3^1, 83^1
        factor_dict = dict(cert.cofactor_factors)
        assert factor_dict == {2: 2, 3: 1, 83: 1}
        assert len(cert.cofactor_certificates) == 3
        sub_primes = {c.prime for c in cert.cofactor_certificates}
        assert sub_primes == {2, 3, 83}

    def test_large_prime_verifies(self) -> None:
        p = int(nextprime(10**50))
        cert = build_pratt_certificate(p)
        assert verify_pratt_certificate(cert) is True

    def test_semiprime_factor_prime(self) -> None:
        """Factor a semiprime's prime factor and verify its certificate."""
        # 1009 * 1013 = 1022117
        cert = build_pratt_certificate(1009)
        assert verify_pratt_certificate(cert) is True
        cert2 = build_pratt_certificate(1013)
        assert verify_pratt_certificate(cert2) is True


class TestPrattCertificateVerificationFailures:
    """Test that corrupted certificates are rejected."""

    def test_non_prime_rejected_by_construct(self) -> None:
        with pytest.raises(ValueError, match="cannot build Pratt certificate"):
            build_pratt_certificate(1)
        with pytest.raises(ValueError, match="cannot build Pratt certificate"):
            build_pratt_certificate(0)
        with pytest.raises(ValueError, match="cannot build Pratt certificate"):
            build_pratt_certificate(-5)

    def test_corrupted_witness_rejected(self) -> None:
        """If we tamper with the witness, verification should fail."""
        cert = build_pratt_certificate(997)
        bad_cert = PrattCertificate(
            prime=997,
            witness=1,  # Not a primitive root
            cofactor_factors=cert.cofactor_factors,
            cofactor_certificates=cert.cofactor_certificates,
        )
        assert verify_pratt_certificate(bad_cert) is False

    def test_corrupted_factors_rejected(self) -> None:
        """If cofactor_factors don't multiply to p-1, verification should fail."""
        cert = build_pratt_certificate(997)
        bad_cert = PrattCertificate(
            prime=997,
            witness=cert.witness,
            cofactor_factors=((2, 1), (3, 1)),  # Wrong factorization
            cofactor_certificates=cert.cofactor_certificates,
        )
        assert verify_pratt_certificate(bad_cert) is False

    def test_wrong_prime_in_subcert_rejected(self) -> None:
        """If a sub-certificate certifies the wrong prime, verification should fail."""
        cert = build_pratt_certificate(997)
        # Tamper with the first sub-certificate's prime
        bad_sub = PrattCertificate(
            prime=7,  # Wrong prime
            witness=cert.cofactor_certificates[0].witness,
            cofactor_factors=cert.cofactor_certificates[0].cofactor_factors,
            cofactor_certificates=cert.cofactor_certificates[0].cofactor_certificates,
        )
        bad_cert = PrattCertificate(
            prime=997,
            witness=cert.witness,
            cofactor_factors=cert.cofactor_factors,
            cofactor_certificates=(bad_sub,) + cert.cofactor_certificates[1:],
        )
        assert verify_pratt_certificate(bad_cert) is False


# ---------------------------------------------------------------------------
# Certified factorization operation (domain adapter)
# ---------------------------------------------------------------------------


class TestCertifiedFactorization:
    """Tests for the integer.factor.certified_compute operation."""

    def test_factor_60(self) -> None:
        result = compute_certified_factor(CertifiedFactorRequest(n="60"))
        assert isinstance(result, CertifiedFactorResult)
        assert result.method == "SYMPY_FACTORINT_WITH_PRATT"
        factors = {f.prime: f.exponent for f in result.factors}
        assert factors == {"2": 2, "3": 1, "5": 1}

    def test_factor_prime(self) -> None:
        result = compute_certified_factor(CertifiedFactorRequest(n="97"))
        assert len(result.factors) == 1
        assert result.factors[0].prime == "97"
        assert result.factors[0].exponent == 1

    def test_factor_prime_power(self) -> None:
        result = compute_certified_factor(CertifiedFactorRequest(n="1024"))
        assert len(result.factors) == 1
        assert result.factors[0].prime == "2"
        assert result.factors[0].exponent == 10

    def test_factor_one(self) -> None:
        """Factorint(1) returns empty dict; factors list should be empty."""
        with pytest.raises(ValidationError, match="greater than 1"):
            CertifiedFactorRequest(n="1")

    def test_factor_semiprime(self) -> None:
        """Factor a product of two large primes."""
        p = 999999999989  # prime
        q = 1000000000039  # prime
        n = p * q
        result = compute_certified_factor(CertifiedFactorRequest(n=str(n)))
        factors = {f.prime: f.exponent for f in result.factors}
        assert factors == {str(p): 1, str(q): 1}

    def test_factor_perfect_power(self) -> None:
        """Factor a perfect power: 6^5 = 7776."""
        result = compute_certified_factor(CertifiedFactorRequest(n="7776"))
        factors = {f.prime: f.exponent for f in result.factors}
        # 6^5 = (2*3)^5 = 2^5 * 3^5
        assert factors == {"2": 5, "3": 5}

    def test_factor_includes_pratt_certificates(self) -> None:
        """Each factor should include a Pratt primality certificate."""
        result = compute_certified_factor(CertifiedFactorRequest(n="360"))
        for factor in result.factors:
            cert = factor.certificate
            assert isinstance(cert, PrattCertificateNode)
            assert int(cert.prime) == int(factor.prime)
            # Certificate should verify
            math_cert = PrattCertificate(
                prime=int(cert.prime),
                witness=int(cert.witness) if cert.witness is not None else None,
                cofactor_factors=tuple(
                    (int(f.prime), f.exponent) for f in cert.cofactor_factors
                ),
                cofactor_certificates=tuple(
                    PrattCertificate(
                        prime=int(sub.prime),
                        witness=int(sub.witness) if sub.witness is not None else None,
                        cofactor_factors=tuple(
                            (int(f.prime), f.exponent) for f in sub.cofactor_factors
                        ),
                        cofactor_certificates=(),
                    )
                    for sub in cert.cofactor_certificates
                ),
            )
            assert verify_pratt_certificate(math_cert) is True

    def test_factor_large_prime(self) -> None:
        """Factor a number involving a large prime."""
        # 2 * 999999999989
        n = 2 * 999999999989
        result = compute_certified_factor(CertifiedFactorRequest(n=str(n)))
        factors = {f.prime: f.exponent for f in result.factors}
        assert factors == {"2": 1, "999999999989": 1}


# ---------------------------------------------------------------------------
# Primality certificate operation (domain adapter)
# ---------------------------------------------------------------------------


class TestPrimalityCertificate:
    """Tests for the integer.primality.certificate.compute operation."""

    def test_certificate_for_2(self) -> None:
        result = compute_primality_certificate(PrimalityCertificateRequest(p="2"))
        assert isinstance(result, PrimalityCertificateResult)
        assert result.status == "PRIME"
        assert result.method == "PRATT_CERTIFICATE"
        assert result.prime == "2"
        assert result.certificate.prime == "2"
        assert result.certificate.witness is None

    def test_certificate_for_17(self) -> None:
        result = compute_primality_certificate(PrimalityCertificateRequest(p="17"))
        assert result.prime == "17"
        assert result.certificate.prime == "17"
        assert result.certificate.witness is not None
        # 17 - 1 = 16 = 2^4
        assert len(result.certificate.cofactor_factors) == 1
        assert result.certificate.cofactor_factors[0].prime == "2"
        assert result.certificate.cofactor_factors[0].exponent == 4

    def test_certificate_for_997(self) -> None:
        result = compute_primality_certificate(PrimalityCertificateRequest(p="997"))
        assert result.prime == "997"
        # 997 - 1 = 996 = 2^2 * 3 * 83
        factor_dict = {
            f.prime: f.exponent for f in result.certificate.cofactor_factors
        }
        assert factor_dict == {"2": 2, "3": 1, "83": 1}

    def test_certificate_for_large_prime(self) -> None:
        p = int(nextprime(10**30))
        result = compute_primality_certificate(PrimalityCertificateRequest(p=str(p)))
        assert result.prime == str(p)
        assert result.certificate.prime == str(p)

    def test_non_prime_raises(self) -> None:
        with pytest.raises(ValueError, match="not prime"):
            compute_primality_certificate(PrimalityCertificateRequest(p="4"))
        with pytest.raises(ValueError, match="not prime"):
            compute_primality_certificate(PrimalityCertificateRequest(p="15"))

    def test_certificate_round_trip_verification(self) -> None:
        """Convert the contract certificate back to a PrattCertificate and verify."""
        result = compute_primality_certificate(PrimalityCertificateRequest(p="7919"))
        cert = result.certificate

        def to_math(node: PrattCertificateNode) -> PrattCertificate:
            return PrattCertificate(
                prime=int(node.prime),
                witness=int(node.witness) if node.witness is not None else None,
                cofactor_factors=tuple(
                    (int(f.prime), f.exponent) for f in node.cofactor_factors
                ),
                cofactor_certificates=tuple(
                    to_math(sub) for sub in node.cofactor_certificates
                ),
            )

        math_cert = to_math(cert)
        assert verify_pratt_certificate(math_cert) is True


# ---------------------------------------------------------------------------
# Contract validation (fail-closed)
# ---------------------------------------------------------------------------


class TestContractValidation:
    """Test that contracts reject invalid inputs."""

    def test_factor_request_rejects_zero(self) -> None:
        with pytest.raises(ValidationError, match="greater than 1"):
            CertifiedFactorRequest(n="0")

    def test_factor_request_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="greater than 1"):
            CertifiedFactorRequest(n="-5")

    def test_factor_request_rejects_one(self) -> None:
        with pytest.raises(ValidationError, match="greater than 1"):
            CertifiedFactorRequest(n="1")

    def test_factor_request_rejects_non_numeric(self) -> None:
        with pytest.raises(ValidationError):
            CertifiedFactorRequest(n="abc")

    def test_primality_request_rejects_zero(self) -> None:
        with pytest.raises(ValidationError, match="at least 2"):
            PrimalityCertificateRequest(p="0")

    def test_primality_request_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="at least 2"):
            PrimalityCertificateRequest(p="-1")

    def test_pratt_certificate_base_case_consistency(self) -> None:
        """PrattCertificateNode for p=2 must have empty cofactor fields."""
        with pytest.raises(ValidationError, match="p=2"):
            PrattCertificateNode(
                prime="2",
                witness="1",
                cofactor_factors=[],
                cofactor_certificates=[],
            )

    def test_pratt_certificate_non_base_requires_witness(self) -> None:
        """PrattCertificateNode for p>2 must have a witness."""
        with pytest.raises(ValidationError, match="primitive root witness"):
            PrattCertificateNode(
                prime="5",
                witness=None,
                cofactor_factors=[],
                cofactor_certificates=[],
            )


# ---------------------------------------------------------------------------
# Cross-consistency: factorization product check
# ---------------------------------------------------------------------------


class TestFactorizationConsistency:
    """Verify that factorization results reconstruct the original integer."""

    @pytest.mark.parametrize(
        "n", ["2", "6", "12", "60", "360", "1000", "99991", "123456789"]
    )
    def test_product_of_factors(self, n: str) -> None:
        result = compute_certified_factor(CertifiedFactorRequest(n=n))
        product = 1
        for factor in result.factors:
            product *= int(factor.prime) ** factor.exponent
        assert product == int(n)
