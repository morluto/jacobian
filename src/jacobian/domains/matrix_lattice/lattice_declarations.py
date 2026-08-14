"""Bounded lattice-reduction operation declarations."""

from jacobian.domains.matrix_lattice.lattice import (
    LATTICE_OPERATIONS,
)
from jacobian.operation_declarations import OperationDeclarations


def lattice_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return LATTICE_OPERATIONS


CHECKER_DECLARATIONS = ()
