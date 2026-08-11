"""Installation bundle for exact arithmetic.

The arithmetic domain owns integer absolute value, sign, decimal digit
sum/count, base expansion, integer nth root, and rational arithmetic/order.
Number-theory capabilities (gcd, lcm, divisors, primes, modular arithmetic,
integer predicates) are owned by the number-theory domain (p3) and are
intentionally excluded from this bundle.
"""

from __future__ import annotations

import platform

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.arithmetic.checkers import ARITHMETIC_EXACT_REPLAY_CHECKERS
from jacobian.domains.arithmetic.integers import INTEGER_CAPABILITIES
from jacobian.domains.arithmetic.rationals import RATIONAL_CAPABILITIES
from jacobian.operations import (
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_arithmetic_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="arithmetic",
        schema_namespace="jacobian.arithmetic",
        semantics=DomainSemantics(
            name="jacobian.exact-arithmetic",
            version="1",
            definition={
                "description": (
                    "exact integer absolute value, sign, decimal digit sum/count, "
                    "base expansion, integer nth root, and rational arithmetic/order "
                    "over canonical integer and rational strings"
                ),
                "integer_encoding": "canonical decimal string",
                "rational_encoding": "canonical reduced num/den with positive denominator",
                "arithmetic": "exact via stdlib and maintained SymPy APIs",
                "assurance": (
                    "producers are computed; core binary rational arithmetic "
                    "supports operator-authorized independent replay"
                ),
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.sympy",
            features=("exact-integer-arithmetic", "exact-rational-arithmetic"),
            configuration={"sympy_version": SYMPY_VERSION},
        ),
        backend_version=f"python-{platform.python_version()};sympy-{SYMPY_VERSION}",
        capabilities=(
            *INTEGER_CAPABILITIES,
            *RATIONAL_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_ARITHMETIC_REQUEST",
                stage="arithmetic_input_validation",
                message="Input does not satisfy the exact arithmetic contract.",
                hint=(
                    "Use canonical integer/rational strings and bounded values; "
                    "inspect the operation's request schema."
                ),
            )
        ),
        scope_description="the complete supplied bounded exact arithmetic input",
        completeness_basis=(
            "deterministic exact computation covered the supplied input; core "
            "binary rational arithmetic can additionally be independently replayed"
        ),
        assurance_basis=(
            "deterministic exact arithmetic from the pinned stdlib/SymPy runtime; "
            "VERIFIED only after an operator-authorized independent checker accepts"
        ),
        checker_declarations=ARITHMETIC_EXACT_REPLAY_CHECKERS,
    )
