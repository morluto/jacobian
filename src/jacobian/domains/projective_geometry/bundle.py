"""Installation bundle for exact rational projective geometry."""

from __future__ import annotations

import sympy

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.projective_geometry.arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_CAPABILITY,
)
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import known_provider_runtime

PROJECTIVE_GEOMETRY_BUNDLE = DomainBundle(
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
            "assurance": "computed; complete finite materialization",
        },
    ),
    provider_runtime=known_provider_runtime(
        "jacobian.sympy",
        features=("exact-rational-projective-incidence",),
    ),
    backend_version=sympy.__version__,
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
    scope_description="the complete supplied finite labelled line arrangement",
    completeness_basis=(
        "every unordered line pair was exactly intersected and every coincident "
        "point was grouped with all incident supplied lines"
    ),
    assurance_basis=(
        "exact rational-to-primitive-integer incidence materialization; no "
        "independent checker invoked"
    ),
)

__all__ = ["PROJECTIVE_GEOMETRY_BUNDLE"]
