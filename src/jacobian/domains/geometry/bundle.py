"""Installation bundle for exact rational planar geometry."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.geometry.checkers import GEOMETRY_EXACT_REPLAY_CHECKERS
from jacobian.domains.geometry.lines import LINE_CAPABILITIES
from jacobian.domains.geometry.points import POINT_CAPABILITIES
from jacobian.domains.geometry.polygons import POLYGON_CAPABILITIES
from jacobian.domains.geometry.segments import SEGMENT_CAPABILITIES
from jacobian.domains.geometry.triangles import TRIANGLE_CAPABILITIES
from jacobian.operations import (
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime


def build_geometry_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="geometry",
        schema_namespace="jacobian.geometry",
        semantics=DomainSemantics(
            name="jacobian.exact-rational-plane-geometry",
            version="3",
            definition={
                "description": (
                    "Euclidean plane geometry over exact rational coordinates, "
                    "including bounded exact rational-weight convex triangulation"
                ),
                "degeneracy": "operation-specific and fail-closed",
                "assurance": (
                    "computed producers; selected results admit separately "
                    "authorized independent exact replay"
                ),
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.sympy",
            features=("exact-rational-geometry",),
        ),
        backend_version=SYMPY_VERSION,
        capabilities=(
            *POINT_CAPABILITIES,
            *SEGMENT_CAPABILITIES,
            *LINE_CAPABILITIES,
            *TRIANGLE_CAPABILITIES,
            *POLYGON_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_GEOMETRY_REQUEST",
                stage="geometry_input_validation",
                message="Input does not satisfy the exact planar-geometry contract.",
                hint="Use canonical rationals and inspect the operation's point/line schema.",
            )
        ),
        scope_description="the complete supplied exact rational geometry input",
        completeness_basis=(
            "exact symbolic computation covered the supplied finite input; "
            "not independently verified"
        ),
        assurance_basis="exact SymPy rational geometry; no independent checker invoked",
        checker_declarations=GEOMETRY_EXACT_REPLAY_CHECKERS,
    )
