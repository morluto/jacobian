"""Installation bundle for exact rational projective geometry."""

from __future__ import annotations

from jacobian.domains.projective_geometry.arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
)
from jacobian.domains.projective_geometry.checkers import (
    PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS,
)
from jacobian.operation_declarations import OperationDeclarations


def build_projective_geometry_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (PROJECTIVE_LINE_ARRANGEMENT_OPERATION,)


__all__ = ["build_projective_geometry_bundle"]

CHECKER_DECLARATIONS = PROJECTIVE_GEOMETRY_EXACT_REPLAY_CHECKERS
