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
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.arithmetic.integers import INTEGER_CAPABILITIES
from jacobian.domains.arithmetic.rationals import RATIONAL_CAPABILITIES
from jacobian.domains.arithmetic.real_quadratic import (
    REAL_QUADRATIC_CAPABILITIES,
    REAL_QUADRATIC_CHECKERS,
)
from jacobian.operations import (
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
            *REAL_QUADRATIC_CAPABILITIES,
        ),
        checker_declarations=REAL_QUADRATIC_CHECKERS,
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
    )
