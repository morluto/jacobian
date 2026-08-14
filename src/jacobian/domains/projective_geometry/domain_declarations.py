"""Exact rational projective-geometry operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.projective_geometry.arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
)
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def projective_geometry_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (PROJECTIVE_LINE_ARRANGEMENT_OPERATION,),
        OperationDiagnostic(
            code="INVALID_PROJECTIVE_ARRANGEMENT_REQUEST",
            stage="projective_arrangement_input_validation",
            message=(
                "Input does not satisfy the bounded labelled rational projective "
                "line-arrangement contract."
            ),
            hint="Use distinct normalized rational lines with unique labels.",
        ),
    )


__all__ = ["projective_geometry_operations"]
