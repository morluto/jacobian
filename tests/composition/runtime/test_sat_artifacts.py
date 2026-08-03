from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactValidationError
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.sat import (
    CanonicalCnf,
    SatAssignmentArtifact,
    SatProofArtifact,
    SatResourceBudget,
)
from jacobian.runtime import create_runtime
from jacobian.sat_smt.sat import SatArtifactError


def _producer() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="cadical",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="2.1.3",
        digest="sha256:" + "d" * 64,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T2,
        license_id="MIT",
    )


def test_sat_service_materializes_one_identity_for_equivalent_cnf_input(
    attached_complete_runtime,
) -> None:

    runtime = attached_complete_runtime
    first = runtime.core.sat.put_cnf(
        variable_names=("b", "a"),
        clauses=((1, -2, 1), (2,), (1, -1)),
    )
    second = runtime.core.sat.put_cnf(
        variable_names=("a", "b"),
        clauses=((1,), (-1, 2), (-1, 2)),
    )

    assert first == second
    stored = runtime.core.store.get(first.artifact_uri)
    cnf = CanonicalCnf.model_validate(stored.payload)
    assert cnf.to_dimacs_bytes() == b"p cnf 2 2\n-1 2 0\n1 0\n"
    assert stored.manifest.schema_uri == runtime.core.sat.installation.cnf_schema_uri
    assert stored.manifest.semantics_uri == runtime.core.sat.installation.semantics_uri


def test_sat_cnf_materialization_capability_exposes_reusable_identity(
    attached_complete_runtime,
) -> None:

    runtime = attached_complete_runtime
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.cnf.materialize",
            mode=CapabilityMode.EXPLORE,
            input={
                "variable_names": ["b", "a"],
                "clauses": [[1, -2, 1], [2], [1, -1]],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level.value == "COMPUTED"
    assert result.completeness.status.value == "NOT_APPLICABLE"
    assert result.output["cnf_uri"] in result.artifact_uris
    resolved = runtime.core.sat.resolve_cnf(result.output["cnf_uri"])
    assert resolved.cnf.to_dimacs_bytes() == b"p cnf 2 2\n-1 2 0\n1 0\n"
    assert result.output["schema_uri"] == runtime.core.sat.installation.cnf_schema_uri
    assert result.output["semantics_uri"] == runtime.core.sat.installation.semantics_uri
    assert result.output["variable_map_digest"] == resolved.cnf.variable_map_digest
    assert result.output["dimacs_digest"] == resolved.cnf.dimacs_digest
    assert result.output["caller_order_changed"] is True
    assert result.output["variable_bindings_complete"] is True
    assert result.output["variable_bindings"] == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]
    assert "named assignment map" in result.output["variable_order_note"]


def test_sat_materialization_makes_lexicographic_name_order_explicit(
    attached_complete_runtime,
) -> None:

    runtime = attached_complete_runtime
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.cnf.materialize",
            mode=CapabilityMode.EXPLORE,
            input={
                "variable_names": ["n1", "n2", "n10"],
                "clauses": [[1], [-2], [3]],
            },
        )
    )

    assert result.output["caller_order_changed"] is True
    assert result.output["variable_bindings"] == [
        {"id": 1, "name": "n1"},
        {"id": 2, "name": "n10"},
        {"id": 3, "name": "n2"},
    ]
    resolved = runtime.core.sat.resolve_cnf(result.output["cnf_uri"])
    assert resolved.cnf.to_dimacs_bytes() == b"p cnf 3 3\n1 0\n2 0\n-3 0\n"


def test_sat_cnf_materialization_validates_before_artifact_write(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = attached_complete_runtime
    called = False

    def unexpected_put_cnf(**_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("invalid request reached artifact write")

    monkeypatch.setattr(runtime.core.sat, "put_cnf", unexpected_put_cnf)
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.cnf.materialize",
            mode=CapabilityMode.EXPLORE,
            input={
                "variable_names": ["a", "a"],
                "clauses": [[1]],
            },
        )
    )

    assert called is False
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_CNF"


def test_model_backed_schema_rejects_noncanonical_generic_artifact_put(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    cnf_result = runtime.core.sat.put_cnf(
        variable_names=("a", "b"),
        clauses=((-1, 2), (1,)),
    )
    payload = runtime.core.store.get(cnf_result.artifact_uri).payload
    payload["clauses"] = list(reversed(payload["clauses"]))

    with pytest.raises(ArtifactValidationError):
        runtime.core.artifacts.put(
            schema_uri=runtime.core.sat.installation.cnf_schema_uri,
            semantics_uri=runtime.core.sat.installation.semantics_uri,
            payload=payload,
        )

    restarted = create_runtime(tmp_path)
    with pytest.raises(ArtifactValidationError):
        restarted.core.artifacts.put(
            schema_uri=restarted.core.sat.installation.cnf_schema_uri,
            semantics_uri=restarted.core.sat.installation.semantics_uri,
            payload=payload,
        )


def test_assignment_and_raw_proof_bind_exact_cnf_identity_and_lineage(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    cnf_result = runtime.core.sat.put_cnf(
        variable_names=("a", "b"),
        clauses=((-1, 2), (1,)),
    )
    budget = SatResourceBudget(
        wall_seconds=30,
        memory_bytes=256 * 1024 * 1024,
        conflicts=10_000,
    )
    assignment_result = runtime.core.sat.put_assignment(
        cnf_uri=cnf_result.artifact_uri,
        values=(True, False),
        producer=_producer(),
        resource_budget=budget,
    )
    proof_result = runtime.core.sat.put_proof(
        cnf_uri=cnf_result.artifact_uri,
        proof=b"d -1 2 0\n0\n",
        producer=_producer(),
        resource_budget=budget,
    )

    cnf_artifact = runtime.core.store.get(cnf_result.artifact_uri)
    assignment_artifact = runtime.core.store.get(assignment_result.artifact_uri)
    proof_artifact = runtime.core.store.get(proof_result.artifact_uri)
    assignment = SatAssignmentArtifact.model_validate(assignment_artifact.payload)
    proof = SatProofArtifact.model_validate(proof_artifact.payload)

    for binding in (assignment.cnf, proof.cnf):
        assert binding.cnf_artifact_uri == cnf_result.artifact_uri
        assert binding.cnf_object_digest == cnf_artifact.manifest.object_digest
        assert binding.cnf_payload_digest == cnf_artifact.manifest.payload_digest
        assert (
            binding.variable_map_digest == cnf_artifact.payload["variable_map_digest"]
        )
        assert binding.dimacs_digest == cnf_artifact.payload["dimacs_digest"]
    assert assignment_artifact.manifest.parents == (cnf_result.artifact_uri,)
    assert proof_artifact.manifest.parents == (cnf_result.artifact_uri,)
    assert proof.raw_bytes() == b"d -1 2 0\n0\n"


def test_sat_service_rejects_a_non_cnf_source_before_writing_evidence(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    cnf_result = runtime.core.sat.put_cnf(variable_names=("x",), clauses=((1,),))
    assignment_result = runtime.core.sat.put_assignment(
        cnf_uri=cnf_result.artifact_uri,
        values=(True,),
        producer=_producer(),
        resource_budget=SatResourceBudget(wall_seconds=30),
    )

    with pytest.raises(SatArtifactError, match="canonical CNF artifact"):
        runtime.core.sat.put_proof(
            cnf_uri=assignment_result.artifact_uri,
            proof=b"0\n",
            producer=_producer(),
            resource_budget=SatResourceBudget(wall_seconds=30),
        )
