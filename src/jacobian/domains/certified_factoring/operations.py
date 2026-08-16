"""Domain adapter for certified factoring operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.certified_factoring import (
    CertifiedFactor,
    CertifiedFactorRequest,
    CertifiedFactorResult,
    PrattCertificateNode,
    PrattFactor,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.math.certified_factoring import (
    build_pratt_certificate,
    verify_pratt_certificate,
)
from jacobian.math.certified_factoring.operations import PrattCertificate


def _pratt_node_to_contract(node: PrattCertificate) -> PrattCertificateNode:
    """Convert a PrattCertificate dataclass to a PrattCertificateNode contract."""
    return PrattCertificateNode(
        prime=format_canonical_integer(node.prime),
        witness=format_canonical_integer(node.witness) if node.witness is not None else None,
        cofactor_factors=tuple(
            PrattFactor(
                prime=format_canonical_integer(base),
                exponent=exp,
            )
            for base, exp in node.cofactor_factors
        ),
        cofactor_certificates=tuple(
            _pratt_node_to_contract(sub) for sub in node.cofactor_certificates
        ),
    )


def _pratt_node_from_contract(node: PrattCertificateNode) -> PrattCertificate:
    """Convert a PrattCertificateNode contract back to a PrattCertificate dataclass."""
    return PrattCertificate(
        prime=parse_canonical_integer(node.prime),
        witness=parse_canonical_integer(node.witness) if node.witness is not None else None,
        cofactor_factors=tuple(
            (parse_canonical_integer(f.prime), f.exponent)
            for f in node.cofactor_factors
        ),
        cofactor_certificates=tuple(
            _pratt_node_from_contract(sub) for sub in node.cofactor_certificates
        ),
    )


def compute_certified_factor(request: CertifiedFactorRequest) -> CertifiedFactorResult:
    """Factor n using SymPy's factorint and attach Pratt certificates to each factor."""
    from sympy import factorint

    n = parse_canonical_integer(request.n)
    factors = factorint(n)

    certified_factors: list[CertifiedFactor] = []
    for base, exp in sorted(factors.items()):
        base_int = int(base)
        cert = build_pratt_certificate(base_int)
        certified_factors.append(
            CertifiedFactor(
                prime=format_canonical_integer(base_int),
                exponent=int(exp),
                certificate=_pratt_node_to_contract(cert),
            )
        )

    return CertifiedFactorResult(factors=tuple(certified_factors))


def compute_primality_certificate(
    request: PrimalityCertificateRequest,
) -> PrimalityCertificateResult:
    """Build a Pratt primality certificate for a declared prime.

    Raises ValueError if the candidate is not actually prime.
    """
    from sympy import isprime

    p = parse_canonical_integer(request.p)
    if not isprime(p):
        raise ValueError(f"{p} is not prime; cannot build a Pratt certificate")

    cert = build_pratt_certificate(p)
    return PrimalityCertificateResult(
        prime=format_canonical_integer(p),
        certificate=_pratt_node_to_contract(cert),
    )
