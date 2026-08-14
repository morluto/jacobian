"""Lean frontend services: declarations, exploration, proof inspection, statements."""

from jacobian.lean_frontend.declarations import (
    LeanDeclarationService,
    installed_lean_declaration_service,
)
from jacobian.lean_frontend.exploration import (
    LeanExplorationInstallation,
    install_lean_exploration_operations,
)
from jacobian.lean_frontend.metavariable_fields import (
    LeanMetavariableFieldsAdapter,
    install_lean_metavariable_fields_operation,
)
from jacobian.lean_frontend.proof_axioms import (
    LeanProofAxiomsInstallation,
    install_lean_proof_axioms_operation,
)
from jacobian.lean_frontend.proof_edit import (
    LeanProofEditInstallation,
    install_lean_proof_edit_operation,
)
from jacobian.lean_frontend.proof_state_inspect import (
    LeanProofStateInspectAdapter,
    install_lean_proof_state_inspect_operation,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.lean_frontend.statement import (
    LeanStatementInstallation,
    install_lean_statement_operations,
)
from jacobian.lean_frontend.term_apply import (
    LeanTermApplyAdapter,
    install_lean_term_apply_operation,
)

__all__ = [
    "LeanDeclarationService",
    "LeanExplorationInstallation",
    "LeanMetavariableFieldsAdapter",
    "LeanProofAxiomsInstallation",
    "LeanProofEditInstallation",
    "LeanProofStateInspectAdapter",
    "LeanService",
    "LeanStatementInstallation",
    "LeanTermApplyAdapter",
    "install_lean_exploration_operations",
    "install_lean_metavariable_fields_operation",
    "install_lean_proof_axioms_operation",
    "install_lean_proof_edit_operation",
    "install_lean_proof_state_inspect_operation",
    "install_lean_statement_operations",
    "install_lean_term_apply_operation",
    "installed_lean_declaration_service",
]
