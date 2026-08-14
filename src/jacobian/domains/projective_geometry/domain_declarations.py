"""Exact rational projective-geometry operation declarations."""

from __future__ import annotations

from jacobian.domains.projective_geometry.arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
)
from jacobian.domains.projective_geometry.checkers import (
    PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS,
)
from jacobian.operation_declarations import OperationDeclarations


def projective_geometry_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (PROJECTIVE_LINE_ARRANGEMENT_OPERATION,)


__all__ = ["projective_geometry_operations"]

CHECKER_DECLARATIONS = PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS
