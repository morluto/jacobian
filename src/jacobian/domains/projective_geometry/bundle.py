"""Installation bundle for exact rational projective geometry."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.projective_geometry.arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_CAPABILITY,
)
from jacobian.domains.projective_geometry.checkers import (
    PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS,
)
from jacobian.operations import DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_projective_geometry_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="projective_geometry",
        schema_namespace="jacobian.projective-geometry",
        semantics=DomainSemantics(
            name="jacobian.exact-rational-projective-geometry",
            version="1",
            definition={
                "description": (
                    "Labelled projective lines and points in P^2(Q), represented by "
                    "primitive integer homogeneous coordinates with first nonzero "
                    "coordinate positive"
                ),
                "line_equation": "a*x + b*y + c*z = 0",
                "flat_incidence": "exact homogeneous dot product equals zero",
                "completion": "all distinct line pairs are grouped by exact cross product",
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.sympy",
            features=("exact-rational-projective-incidence",),
        ),
        backend_version=SYMPY_VERSION,
        capabilities=(PROJECTIVE_LINE_ARRANGEMENT_CAPABILITY,),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_PROJECTIVE_ARRANGEMENT_REQUEST",
                stage="projective_arrangement_input_validation",
                message=(
                    "Input does not satisfy the bounded labelled rational projective "
                    "line-arrangement contract."
                ),
                hint=(
                    "Use unique labels and distinct nonzero rational coefficient "
                    "triples; scalar multiples denote the same line."
                ),
            )
        ),
        checker_declarations=PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_projective_geometry_bundle"]
