"""Deterministic formal-dataset operation declarations."""

from __future__ import annotations

from jacobian.contracts.formal_datasets import (
    FormalDatasetArtifact,
    FormalDatasetMaterializeRequest,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains._examples import example
from jacobian.formal_datasets import _materialize_operation
from jacobian.operation_bindings import durable_operation
from jacobian.operation_declarations import (
    OperationDeclaration,
    OperationDeclarations,
    with_invalid_request,
)
from jacobian_checkers.lean4 import LEAN_VERSION


def formal_dataset_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            durable_operation(
                OperationDeclaration(
                    operation_id="dataset.formal.materialize",
                    version="3",
                    title="Materialize one pinned formal-dataset row",
                    description=(
                        "Normalize one MiniF2F or ProofNet row and bind its dataset, "
                        "source, Lean-project, preprocessing, and execution provenance."
                    ),
                    request_type=FormalDatasetMaterializeRequest,
                    result_type=FormalDatasetArtifact,
                    execute=_materialize_operation,
                    tags=("dataset", "formal-mathematics", "lean", "provenance"),
                    examples=(
                        example(
                            "minif2f_core_true",
                            "Materialize a pinned MiniF2F-style CORE fixture.",
                            {
                                "dataset_revision": "fixture-revision-1",
                                "sample_id": "core_true",
                                "source_url": "https://example.invalid/minif2f/core_true",
                                "row": {
                                    "dataset_id": "MINIF2F",
                                    "name": "core_true",
                                    "split": "test",
                                    "formal_statement": (
                                        "theorem core_true : True := by trivial"
                                    ),
                                    "goal": "True",
                                    "informal_statement": "True holds.",
                                    "header": "",
                                },
                                "environment": {
                                    "lean_version": LEAN_VERSION,
                                    "project_source_url": (
                                        "https://example.invalid/formal-project"
                                    ),
                                    "project_revision": "fixture-project-1",
                                    "imports": [],
                                    "project_files": [],
                                },
                            },
                        ),
                    ),
                ),
                resource_reason=(
                    "durable identity is required to bind the normalized row to "
                    "pinned dataset and formal-provider provenance"
                ),
            ),
        ),
        OperationDiagnostic(
            code="INVALID_FORMAL_DATASET_ROW",
            stage="request_validation",
            message="The formal-dataset materialization request is invalid.",
            hint="Provide a supported row with pinned dataset and environment data.",
        ),
    )
