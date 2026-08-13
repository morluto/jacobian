"""Finite-probability domain bundle."""

from dataclasses import replace
from typing import Any

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.probability.checkers import PROBABILITY_EXACT_REPLAY_CHECKERS
from jacobian.domains.probability.gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.domains.probability.mutual_information import (
    MUTUAL_INFORMATION_CAPABILITY,
)
from jacobian.domains.probability.operations import FINITE_PROBABILITY_CAPABILITIES
from jacobian.operations import DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import PYTHON_FLINT_VERSION
from jacobian.providers.flint_runtime import python_flint_probability_provider_runtime


def _operation_with_canonical_gaussian_input(operation: Any) -> Any:
    if operation.spec.operation_id != "probability.gaussian_polynomial.moment.compute":
        return operation
    return replace(
        operation,
        spec=replace(
            operation.spec,
            request_type=CanonicalGaussianPolynomialMomentRequest,
        ),
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
        provider_runtime=python_flint_probability_provider_runtime(),
        backend_version=f"python-flint-{PYTHON_FLINT_VERSION}",
        capabilities=(
            MUTUAL_INFORMATION_CAPABILITY,
            *(
                _operation_with_canonical_gaussian_input(operation)
                for operation in FINITE_PROBABILITY_CAPABILITIES
            ),
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
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
