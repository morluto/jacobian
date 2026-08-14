"""Exact rational planar-geometry operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.geometry.lines import LINE_OPERATIONS
from jacobian.domains.geometry.points import POINT_OPERATIONS
from jacobian.domains.geometry.polygons import POLYGON_OPERATIONS
from jacobian.domains.geometry.segments import SEGMENT_OPERATIONS
from jacobian.domains.geometry.triangles import TRIANGLE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def geometry_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *POINT_OPERATIONS,
            *SEGMENT_OPERATIONS,
            *LINE_OPERATIONS,
            *TRIANGLE_OPERATIONS,
            *POLYGON_OPERATIONS,
        ),
        OperationDiagnostic(
            code="INVALID_GEOMETRY_REQUEST",
            stage="geometry_input_validation",
            message="Input does not satisfy the exact planar-geometry contract.",
            hint="Use canonical rationals and inspect the operation's point/line schema.",
        ),
    )
