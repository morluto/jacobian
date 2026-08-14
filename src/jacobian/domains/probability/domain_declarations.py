"""Finite-probability operation declarations."""

from dataclasses import replace
from typing import Any

from jacobian.domains.probability.checkers import PROBABILITY_EXACT_REPLAY_CHECKERS
from jacobian.domains.probability.gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.domains.probability.mutual_information import (
    MUTUAL_INFORMATION_OPERATION,
)
from jacobian.domains.probability.operations import FINITE_PROBABILITY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def _operation_with_canonical_gaussian_input(operation: Any) -> Any:
    if operation.operation_id != "probability.gaussian_polynomial.moment.compute":
        return operation
    return replace(
        operation,
        request_type=CanonicalGaussianPolynomialMomentRequest,
    )


def finite_probability_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        MUTUAL_INFORMATION_OPERATION,
        *(
            _operation_with_canonical_gaussian_input(operation)
            for operation in FINITE_PROBABILITY_OPERATIONS
        ),
    )


__all__ = ["finite_probability_operations"]

CHECKER_DECLARATIONS = PROBABILITY_EXACT_REPLAY_CHECKERS
