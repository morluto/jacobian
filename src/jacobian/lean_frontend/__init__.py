"""Lean frontend services: declarations, exploration, proof inspection, statements."""

from jacobian.lean_frontend.declarations import (
    LeanDeclarationService,
    installed_lean_declaration_service,
)
from jacobian.lean_frontend.exploration import (
    LeanExplorationInstallation,
    install_lean_exploration_capabilities,
)
from jacobian.lean_frontend.metavariable_fields import (
    LeanMetavariableFieldsAdapter,
    install_lean_metavariable_fields_capability,
)
from jacobian.lean_frontend.proof_axioms import (
    LeanProofAxiomsInstallation,
    install_lean_proof_axioms_capability,
)
from jacobian.lean_frontend.proof_edit import (
    LeanProofEditInstallation,
    install_lean_proof_edit_capability,
)
from jacobian.lean_frontend.proof_state_inspect import (
    LeanProofStateInspectAdapter,
    install_lean_proof_state_inspect_capability,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.lean_frontend.statement import (
    LeanStatementInstallation,
    install_lean_statement_capabilities,
)
from jacobian.lean_frontend.term_apply import (
    LeanTermApplyAdapter,
    install_lean_term_apply_capability,
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
    "install_lean_exploration_capabilities",
    "install_lean_metavariable_fields_capability",
    "install_lean_proof_axioms_capability",
    "install_lean_proof_edit_capability",
    "install_lean_proof_state_inspect_capability",
    "install_lean_statement_capabilities",
    "install_lean_term_apply_capability",
    "installed_lean_declaration_service",
]
