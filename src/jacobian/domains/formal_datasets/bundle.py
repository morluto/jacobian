"""Installation bundle for deterministic formal-dataset materialization."""

from __future__ import annotations

import platform

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.contracts.formal_datasets import (
    FormalDatasetArtifact,
    FormalDatasetMaterializeRequest,
)
from jacobian.domains._examples import example
from jacobian.formal_datasets import _materialize_operation
from jacobian.operations import (
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
    MaterializedOperation,
)
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian_checkers.lean4 import LEAN_VERSION, MATHLIB_COMMIT


def build_formal_dataset_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="formal_datasets",
        schema_namespace="jacobian.formal-datasets",
        semantics=DomainSemantics(
            name="jacobian.formal-dataset-materialization",
            version="1",
            definition={
                "description": (
                    "deterministic formal-dataset row normalization and environment "
                    "provenance binding"
                ),
                "verification": "none; materialization never establishes theorem truth",
            },
        ),
        provider_runtime=jacobian_provider_runtime(
            "jacobian.formal-datasets",
            features=("MINIF2F", "PROOFNET", "deterministic-materialization"),
        ),
        backend_version=(
            f"python-{platform.python_version()};lean-{LEAN_VERSION};"
            f"mathlib-{MATHLIB_COMMIT}"
        ),
        capabilities=(
            MaterializedOperation(
                capability_id="dataset.formal.materialize",
                title="Materialize one pinned formal-dataset row",
                description=(
                    "Normalize one MiniF2F or ProofNet row and bind its dataset, "
                    "source, Lean-project, preprocessing, and execution provenance."
                ),
                request_model=FormalDatasetMaterializeRequest,
                result_model=FormalDatasetArtifact,
                implementation=_materialize_operation,
                relation_id="dataset.formal.materialization",
                tags=("dataset", "formal-mathematics", "lean", "provenance"),
                invocation_examples=(
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
                version="3",
            ),
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_FORMAL_DATASET_ROW",
                stage="request_validation",
                message="The formal-dataset materialization request is invalid.",
                hint="Provide a supported row with pinned dataset and environment data.",
            )
        ),
        scope_description="one complete pinned formal-dataset row",
        completeness_basis=(
            "the complete declared row was normalized and provenance-bound"
        ),
        assurance_basis=(
            "deterministic materialization only; no theorem truth, proof validity, "
            "or informal-formal correspondence was assessed"
        ),
    )
