"""Installation bundle for exact rational planar geometry."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.geometry.checkers import GEOMETRY_EXACT_REPLAY_CHECKERS
from jacobian.domains.geometry.lines import LINE_OPERATIONS
from jacobian.domains.geometry.points import POINT_OPERATIONS
from jacobian.domains.geometry.polygons import POLYGON_OPERATIONS
from jacobian.domains.geometry.segments import SEGMENT_OPERATIONS
from jacobian.domains.geometry.triangles import TRIANGLE_OPERATIONS
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)


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
            },
        ),
        operations=(
            *POINT_OPERATIONS,
            *SEGMENT_OPERATIONS,
            *LINE_OPERATIONS,
            *TRIANGLE_OPERATIONS,
            *POLYGON_OPERATIONS,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_GEOMETRY_REQUEST",
                stage="geometry_input_validation",
                message="Input does not satisfy the exact planar-geometry contract.",
                hint="Use canonical rationals and inspect the operation's point/line schema.",
            )
        ),
        checker_declarations=GEOMETRY_EXACT_REPLAY_CHECKERS,
    )
