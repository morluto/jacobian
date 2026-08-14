"""Exact matrix and lattice operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operation_declarations import OperationDeclarations

__all__ = ["lattice_operations", "matrix_operations"]


def matrix_operations() -> OperationDeclarations:
    """Load matrix declarations without polluting native API imports."""

    from jacobian.domains.matrix_lattice.domain_declarations import (
        matrix_operations as build,
    )

    return build()


def lattice_operations() -> OperationDeclarations:
    """Load lattice declarations without polluting native API imports."""

    from jacobian.domains.matrix_lattice.lattice_declarations import (
        lattice_operations as build,
    )

    return build()
