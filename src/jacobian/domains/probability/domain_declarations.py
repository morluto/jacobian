"""Finite-probability operation declarations."""

from dataclasses import replace
from typing import Any

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.probability.checkers import PROBABILITY_AUTHORIZED_CHECKERS
from jacobian.domains.probability.gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.domains.probability.mutual_information import (
    MUTUAL_INFORMATION_OPERATION,
)
from jacobian.domains.probability.operations import FINITE_PROBABILITY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def _operation_with_canonical_gaussian_input(operation: Any) -> Any:
    if operation.operation_id != "probability.gaussian_polynomial.moment.compute":
        return operation
    return replace(
        operation,
        request_type=CanonicalGaussianPolynomialMomentRequest,
    )


def finite_probability_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            MUTUAL_INFORMATION_OPERATION,
            *(
                _operation_with_canonical_gaussian_input(operation)
                for operation in FINITE_PROBABILITY_OPERATIONS
            ),
        ),
        OperationDiagnostic(
            code="INVALID_FINITE_PROBABILITY_REQUEST",
            stage="finite_probability_input_validation",
            message="Input does not satisfy the bounded exact-probability contract.",
            hint=(
                "Use a bounded normalized finite distribution or joint table, a "
                "bounded Gaussian polynomial request, or a fully weighted small graph."
            ),
        ),
    )


__all__ = ["finite_probability_operations"]

AUTHORIZED_CHECKERS = PROBABILITY_AUTHORIZED_CHECKERS
