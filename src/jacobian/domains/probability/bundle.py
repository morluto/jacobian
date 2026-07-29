"""Finite-probability domain bundle."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.probability.operations import FINITE_PROBABILITY_CAPABILITIES
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import (
    PYTHON_FLINT_VERSION,
    python_flint_probability_provider_runtime,
)

FINITE_PROBABILITY_BUNDLE = DomainBundle(
    domain_id="probability",
    schema_namespace="jacobian.validated-analysis",
    semantics=DomainSemantics(
        name="jacobian.probability",
        version="2",
        definition={
            "description": "bounded exact finite rational probability operations",
            "scope": (
                "raw moments, explicit event mass and conditioning, total "
                "pushforwards, and independent finite convolutions"
            ),
            "failure": "invalid distributions fail before computation",
        },
    ),
    provider_runtime=python_flint_probability_provider_runtime(),
    backend_version=f"python-flint-{PYTHON_FLINT_VERSION}",
    capabilities=FINITE_PROBABILITY_CAPABILITIES,
    diagnostics=DomainDiagnostics(
        invalid_request=CapabilityDiagnostic(
            code="INVALID_FINITE_PROBABILITY_REQUEST",
            stage="finite_probability_input_validation",
            message="Input does not satisfy the finite-probability contract.",
            hint="Use bounded rational atoms whose probabilities sum exactly to one.",
        )
    ),
    scope_description="one bounded exact finite-rational probability operation",
    completeness_basis=(
        "Python-FLINT produced every selected atom, source-map contribution, "
        "or bounded product-measure pair required by the request"
    ),
    assurance_basis="pinned maintained-backend exact rational computation",
)

__all__ = ["FINITE_PROBABILITY_BUNDLE"]
