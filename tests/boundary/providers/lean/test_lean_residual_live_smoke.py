"""Live Python-to-Lean smoke for the #95 residual contracts.

This is a focused, opt-in smoke that exercises the real pinned Lean 4.31.0
REPL and the rebuilt ``jacobian_lean_proof_state`` helper end-to-end. It is
skipped automatically when the provider runtime is unavailable so it never
turns a missing-toolchain environment into a false failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.operations import (
    OperationRequest,
    ProviderAvailability,
)
from jacobian.lean_frontend.exploration import install_lean_exploration_operations
from jacobian.operation_projection import project_operation_result
from jacobian.providers.lean_runtime import lean_provider_runtime
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def _installation(environment: LeanEnvironment) -> LeanCheckerInstallation:
    digest = "artifact://sha256/" + (
        "a" * 64 if environment is LeanEnvironment.CORE else "b" * 64
    )
    checker = "checker://sha256/" + (
        "c" * 64 if environment is LeanEnvironment.CORE else "d" * 64
    )
    return LeanCheckerInstallation(
        environment=environment,
        lean_version="4.31.0",
        lean_commit="68218e876d2a38b1985b8590fff244a83c321783",
        import_name=None if environment is LeanEnvironment.CORE else "Mathlib",
        mathlib_commit=(
            None
            if environment is LeanEnvironment.CORE
            else "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
        ),
        allowed_axioms=(),
        checker_timeout_seconds=30,
        semantics_uri=digest,
        claim_schema_uri=digest,
        candidate_schema_uri=digest,
        certificate_schema_uri=digest,
        checker_id=checker,
    )


def _live_adapters(tmp_path: Path):
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installations = {
        environment: _installation(environment) for environment in LeanEnvironment
    }
    runtime = lean_provider_runtime(
        profiles={
            env.value: {
                "semantics_uri": inst.semantics_uri,
                "import_name": inst.import_name,
                "mathlib_commit": inst.mathlib_commit,
                "allowed_axioms": list(inst.allowed_axioms),
                "checker_timeout_seconds": inst.checker_timeout_seconds,
            }
            for env, inst in installations.items()
        },
        checker_ids=tuple(inst.checker_id for inst in installations.values()),
    )
    if runtime.availability is not ProviderAvailability.AVAILABLE:
        pytest.skip(f"pinned Lean runtime unavailable: {runtime.diagnostic}")
    adapters, _ = install_lean_exploration_operations(
        store,
        schemas,
        artifacts,
        installations,
        runtime,
    )
    return adapters  # (proof_state, premise, term_apply, inspect, metavariable)


def test_live_term_apply_and_inspect_and_metavariable_round_trip(
    tmp_path: Path,
) -> None:
    proof_state, _, term_apply, inspect, metavariable = _live_adapters(tmp_path)

    # 1. lean.term.apply: open a state for `True` and apply the term `True.intro`.
    #    `exact True.intro` closes the goal `⊢ True` via the constructor term.
    opened = project_operation_result(
        term_apply.invoke(
            term_apply.prepare(
                OperationRequest(
                    operation_id="lean.term.apply",
                    input={
                        "environment": "CORE",
                        "statement": "True",
                        "term": "True.intro",
                    },
                )
            )
        )
    )
    assert opened.output["accepted"] is True
    assert opened.output["completed"] is True
    assert opened.output["tactic"] == "exact True.intro"
    assert opened.output["term_application"] == "LEAN_EXACT_ELABORATION"

    # 2. lean.proof_state.inspect: read the input state without replay.
    #    Reopen a non-completed state for inspection.
    reopened = project_operation_result(
        proof_state.invoke(
            proof_state.prepare(
                OperationRequest(
                    operation_id="lean.proof_state.apply_tactic",
                    input={
                        "environment": "CORE",
                        "statement": "P → P",
                        "proof_prefix": ["intro P"],
                        "tactic": "skip",
                    },
                )
            )
        )
    )
    open_state_uri = reopened.output["successor_states"][0]["state_uri"]
    inspected = project_operation_result(
        inspect.invoke(
            inspect.prepare(
                OperationRequest(
                    operation_id="lean.proof_state.inspect",
                    input={"environment": "CORE", "state_uri": open_state_uri},
                )
            )
        )
    )
    assert inspected.output["inspection"] == "READ_ONLY_NO_REPLAY"
    assert inspected.output["goal_count"] == 1
    assert "⊢ P" in inspected.output["normalized_goals"][0]

    # 3. lean.proof_state.metavariable_fields: structured fields via the helper.
    fields = project_operation_result(
        metavariable.invoke(
            metavariable.prepare(
                OperationRequest(
                    operation_id="lean.proof_state.metavariable_fields",
                    input={"environment": "CORE", "state_uri": open_state_uri},
                )
            )
        )
    )
    assert fields.output["coercion_provenance"] == "UNAVAILABLE"
    mvar = fields.output["structured_metavariables"][0]
    assert mvar["kind"] in ("NATURAL", "SYNTHETIC", "SYNTHETIC_OPAQUE")
    assert mvar["is_assigned"] is False
    assert "P" in mvar["target_type"]
    elab = fields.output["elaboration_context"]
    assert isinstance(elab["may_postpone"], bool)
