"""Installation bundle for exact arithmetic.

The arithmetic domain owns integer absolute value, sign, decimal digit
sum/count, base expansion, integer nth root, and rational arithmetic/order.
Number-theory operations (gcd, lcm, divisors, primes, modular arithmetic,
integer predicates) are owned by the number-theory domain (p3) and are
intentionally excluded from this bundle.
"""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.arithmetic.integers import INTEGER_OPERATIONS
from jacobian.domains.arithmetic.rationals import RATIONAL_OPERATIONS
from jacobian.domains.arithmetic.real_quadratic import (
    REAL_QUADRATIC_CHECKERS,
    REAL_QUADRATIC_OPERATIONS,
)
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)


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
        operations=(
            *INTEGER_OPERATIONS,
            *RATIONAL_OPERATIONS,
            *REAL_QUADRATIC_OPERATIONS,
        ),
        checker_declarations=REAL_QUADRATIC_CHECKERS,
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
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
