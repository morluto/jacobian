"""Exact arithmetic operation declarations.

The arithmetic domain owns integer absolute value, sign, decimal digit
sum/count, base expansion, integer nth root, and rational arithmetic/order.
Number-theory operations (gcd, lcm, divisors, primes, modular arithmetic,
integer predicates) are owned by the number-theory domain (p3) and are
intentionally excluded from this bundle.
"""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.arithmetic.integers import INTEGER_OPERATIONS
from jacobian.domains.arithmetic.rationals import RATIONAL_OPERATIONS
from jacobian.domains.arithmetic.real_quadratic import (
    REAL_QUADRATIC_CHECKERS,
    REAL_QUADRATIC_OPERATIONS,
)
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def arithmetic_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (*INTEGER_OPERATIONS, *RATIONAL_OPERATIONS, *REAL_QUADRATIC_OPERATIONS),
        OperationDiagnostic(
            code="INVALID_ARITHMETIC_REQUEST",
            stage="arithmetic_input_validation",
            message="Input does not satisfy the exact arithmetic contract.",
            hint=(
                "Use canonical integer/rational strings and bounded values; inspect "
                "the operation's request schema."
            ),
        ),
    )


AUTHORIZED_CHECKERS = REAL_QUADRATIC_CHECKERS
