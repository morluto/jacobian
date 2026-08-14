"""Exact matrix operation declarations."""

from __future__ import annotations

from jacobian.domains.matrix_lattice.checkers import MATRIX_EXACT_REPLAY_CHECKERS
from jacobian.domains.matrix_lattice.hnf import HERMITE_NORMAL_FORM_OPERATION
from jacobian.domains.matrix_lattice.operation_declarations import MATRIX_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def matrix_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (*MATRIX_OPERATIONS, HERMITE_NORMAL_FORM_OPERATION)


CHECKER_DECLARATIONS = MATRIX_EXACT_REPLAY_CHECKERS
