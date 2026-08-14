"""Exact rational planar-geometry operation declarations."""

from __future__ import annotations

from jacobian.domains.geometry.checkers import GEOMETRY_EXACT_REPLAY_CHECKERS
from jacobian.domains.geometry.lines import LINE_OPERATIONS
from jacobian.domains.geometry.points import POINT_OPERATIONS
from jacobian.domains.geometry.polygons import POLYGON_OPERATIONS
from jacobian.domains.geometry.segments import SEGMENT_OPERATIONS
from jacobian.domains.geometry.triangles import TRIANGLE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def geometry_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *POINT_OPERATIONS,
        *SEGMENT_OPERATIONS,
        *LINE_OPERATIONS,
        *TRIANGLE_OPERATIONS,
        *POLYGON_OPERATIONS,
    )


CHECKER_DECLARATIONS = GEOMETRY_EXACT_REPLAY_CHECKERS
