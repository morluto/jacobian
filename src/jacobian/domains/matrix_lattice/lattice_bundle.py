"""Installation bundle for bounded lattice reduction."""

from jacobian.domains.matrix_lattice.lattice import (
    LATTICE_OPERATIONS,
)
from jacobian.operation_declarations import OperationDeclarations


def build_lattice_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return LATTICE_OPERATIONS


CHECKER_DECLARATIONS = ()
