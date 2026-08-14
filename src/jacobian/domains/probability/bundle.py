"""Finite-probability domain bundle."""

from dataclasses import replace
from typing import Any

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.probability.checkers import PROBABILITY_EXACT_REPLAY_CHECKERS
from jacobian.domains.probability.gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.domains.probability.mutual_information import (
    MUTUAL_INFORMATION_OPERATION,
)
from jacobian.domains.probability.operations import FINITE_PROBABILITY_OPERATIONS
from jacobian.operations import DomainDiagnostics, DomainSemantics


def _operation_with_canonical_gaussian_input(operation: Any) -> Any:
    if operation.operation_id != "probability.gaussian_polynomial.moment.compute":
        return operation
    return replace(
        operation,
        request_type=CanonicalGaussianPolynomialMomentRequest,
    )


def build_finite_probability_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="probability",
        schema_namespace="jacobian.validated-analysis",
        semantics=DomainSemantics(
            name="jacobian.probability",
            version="5",
            definition={
                "description": "bounded exact rational probability operations",
                "scope": (
                    "raw moments, explicit event mass and conditioning, total "
                    "pushforwards, independent finite convolutions, exact mutual "
                    "information for bounded rational joint tables, one fixed-order "
                    "moment of a sparse complex-rational polynomial in independent "
                    "standard real Gaussian variables, and exact small-graph terminal "
                    "connection reliability"
                ),
                "failure": "invalid bounded probability inputs fail before computation",
            },
        ),
        operations=(
            MUTUAL_INFORMATION_OPERATION,
            *(
                _operation_with_canonical_gaussian_input(operation)
                for operation in FINITE_PROBABILITY_OPERATIONS
            ),
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_FINITE_PROBABILITY_REQUEST",
                stage="finite_probability_input_validation",
                message="Input does not satisfy the bounded exact-probability contract.",
                hint=(
                    "Use a bounded normalized finite distribution or joint table, "
                    "a bounded Gaussian polynomial request, or a fully weighted "
                    "small graph."
                ),
            )
        ),
        checker_declarations=PROBABILITY_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_finite_probability_bundle"]
