from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.sat import (
    CanonicalCnf,
    SatAssignmentArtifact,
    SatCnfBinding,
    SatExplorationRequest,
    SatModelFindOutput,
    SatProofArtifact,
    SatResourceBudget,
    SatUnsatProofFindOutput,
    canonicalize_cnf,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_DIGEST_D = "sha256:" + "d" * 64
_ARTIFACT_A = "artifact://sha256/" + "a" * 64


def _producer() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="cadical",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="2.1.3",
        digest=_DIGEST_D,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T2,
        license_id="MIT",
    )


def _binding(*, variable_count: int = 2) -> SatCnfBinding:
    return SatCnfBinding(
        cnf_artifact_uri=_ARTIFACT_A,
        cnf_object_digest=_DIGEST_A,
        cnf_payload_digest=_DIGEST_B,
        variable_map_digest=_DIGEST_C,
        dimacs_digest=_DIGEST_D,
        projection_format="DIMACS-CNF",
        projection_version="jacobian.dimacs.cnf/v1",
        variable_count=variable_count,
        clause_count=2,
    )


def test_cnf_canonicalization_normalizes_map_literals_clauses_and_tautologies() -> None:
    first = canonicalize_cnf(
        variable_names=("b", "a"),
        clauses=((1, -2, 1), (2,), (1, -1)),
    )
    second = canonicalize_cnf(
        variable_names=("a", "b"),
        clauses=((1,), (-1, 2), (-1, 2)),
    )

    assert first == second
    assert tuple((variable.id, variable.name) for variable in first.variables) == (
        (1, "a"),
        (2, "b"),
    )
    assert tuple(clause.literals for clause in first.clauses) == ((-1, 2), (1,))
    assert first.to_dimacs_bytes() == b"p cnf 2 2\n-1 2 0\n1 0\n"
    assert first.variable_map_digest.startswith("sha256:")
    assert first.dimacs_digest.startswith("sha256:")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("variables", "variables must use contiguous IDs and ascending names"),
        ("literal_order", "clause literals must be unique and canonically ordered"),
        ("clause_order", "clauses must be unique and canonically ordered"),
        ("literal_range", "literal references an undeclared variable"),
        ("variable_map_digest", "variable-map digest"),
        ("dimacs_digest", "DIMACS digest"),
    ),
)
def test_canonical_cnf_rejects_noncanonical_or_misbound_payloads(
    mutation: str,
    message: str,
) -> None:
    payload = canonicalize_cnf(
        variable_names=("a", "b"),
        clauses=((-1, 2), (1,)),
    ).model_dump(mode="json")
    if mutation == "variables":
        payload["variables"] = list(reversed(payload["variables"]))
    elif mutation == "literal_order":
        payload["clauses"][0]["literals"] = [2, -1]
    elif mutation == "clause_order":
        payload["clauses"] = list(reversed(payload["clauses"]))
    elif mutation == "literal_range":
        payload["clauses"][0]["literals"] = [-1, 3]
    elif mutation == "variable_map_digest":
        payload["variable_map_digest"] = _DIGEST_A
    else:
        payload["dimacs_digest"] = _DIGEST_A

    with pytest.raises(ValidationError, match=message):
        CanonicalCnf.model_validate(payload)


def test_cnf_canonicalization_rejects_invalid_variable_maps_and_literals() -> None:
    with pytest.raises(ValueError, match="variable names must be unique"):
        canonicalize_cnf(variable_names=("x", "x"), clauses=((1,),))
    with pytest.raises(ValueError, match="nonzero integers"):
        canonicalize_cnf(variable_names=("x",), clauses=((True,),))
    with pytest.raises(ValueError, match="undeclared variable"):
        canonicalize_cnf(variable_names=("x",), clauses=((2,),))


def test_assignment_is_total_strict_and_cannot_claim_verification() -> None:
    assignment = SatAssignmentArtifact(
        cnf=_binding(),
        values=(True, False),
        producer=_producer(),
        resource_budget=SatResourceBudget(wall_seconds=30),
    )
    assert assignment.declared_scope == "FULL_CNF"

    payload = assignment.model_dump(mode="json")
    payload["values"] = [True]
    with pytest.raises(ValidationError, match="one value for every bound variable"):
        SatAssignmentArtifact.model_validate(payload)

    payload = assignment.model_dump(mode="json")
    payload["values"] = [1, 0]
    with pytest.raises(ValidationError):
        SatAssignmentArtifact.model_validate(payload)

    payload = assignment.model_dump(mode="json")
    payload["verification"] = "VERIFIED"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SatAssignmentArtifact.model_validate(payload)


def test_assignment_and_proof_require_an_available_exact_producer() -> None:
    unavailable = _producer().model_copy(
        update={
            "availability": CapabilityProviderAvailability.UNAVAILABLE,
            "version": None,
            "digest": None,
            "digest_kind": None,
            "diagnostic": "not installed",
        }
    )
    assignment = {
        "cnf": _binding().model_dump(mode="json"),
        "values": [True, False],
        "producer": unavailable.model_dump(mode="json"),
        "resource_budget": {"wall_seconds": 30},
    }
    with pytest.raises(ValidationError, match="available producer runtime"):
        SatAssignmentArtifact.model_validate(assignment)

    proof = {
        "cnf": _binding().model_dump(mode="json"),
        "proof_base64": "",
        "proof_digest": "sha256:" + "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855",
        "producer": unavailable.model_dump(mode="json"),
        "resource_budget": {"wall_seconds": 30},
    }
    with pytest.raises(ValidationError, match="available producer runtime"):
        SatProofArtifact.model_validate(proof)


def test_proof_preserves_exact_bytes_and_rejects_digest_or_encoding_mutation() -> None:
    proof = SatProofArtifact.from_bytes(
        cnf=_binding(),
        proof=b"d -1 2 0\n0\n",
        producer=_producer(),
        resource_budget=SatResourceBudget(wall_seconds=30, conflicts=1000),
    )

    assert proof.raw_bytes() == b"d -1 2 0\n0\n"
    assert proof.proof_format == "DRAT"
    assert proof.proof_format_version == "drat-text/v1"

    payload = proof.model_dump(mode="json")
    payload["proof_digest"] = _DIGEST_A
    with pytest.raises(ValidationError, match="raw proof digest"):
        SatProofArtifact.model_validate(payload)

    payload = proof.model_dump(mode="json")
    payload["proof_base64"] = payload["proof_base64"] + "\n"
    with pytest.raises(ValidationError):
        SatProofArtifact.model_validate(payload)


def test_binding_and_budget_fields_are_required_and_fail_closed() -> None:
    assignment = SatAssignmentArtifact(
        cnf=_binding(),
        values=(True, False),
        producer=_producer(),
        resource_budget=SatResourceBudget(wall_seconds=30),
    ).model_dump(mode="json")

    for required in (
        "cnf_artifact_uri",
        "cnf_object_digest",
        "cnf_payload_digest",
        "variable_map_digest",
        "dimacs_digest",
        "projection_version",
    ):
        mutated = deepcopy(assignment)
        mutated["cnf"].pop(required)
        with pytest.raises(ValidationError):
            SatAssignmentArtifact.model_validate(mutated)

    assignment["resource_budget"] = {"wall_seconds": 0}
    with pytest.raises(ValidationError):
        SatAssignmentArtifact.model_validate(assignment)


def test_exploration_request_exposes_only_enforced_budget_fields() -> None:
    request = SatExplorationRequest.model_validate(
        {
            "cnf_uri": _ARTIFACT_A,
            "resource_budget": {
                "wall_seconds": 5,
                "conflicts": 100,
            },
        }
    )

    assert request.resource_budget.artifact_budget() == SatResourceBudget(
        wall_seconds=5,
        conflicts=100,
    )
    with pytest.raises(ValidationError):
        SatExplorationRequest.model_validate(
            {
                "cnf_uri": _ARTIFACT_A,
                "resource_budget": {
                    "wall_seconds": 5,
                    "memory_bytes": 1024,
                },
            }
        )
    with pytest.raises(ValidationError):
        SatExplorationRequest.model_validate(
            {
                "cnf_uri": _ARTIFACT_A,
                "resource_budget": {"wall_seconds": 151},
            }
        )


def test_exploration_outputs_do_not_project_a_mathematical_conclusion() -> None:
    model = SatModelFindOutput(
        status="NO_ASSIGNMENT_PRODUCED",
        solver_status="UNSATISFIABLE",
        cnf_uri=_ARTIFACT_A,
        detail="no assignment was produced",
    )
    proof = SatUnsatProofFindOutput(
        status="NO_PROOF_PRODUCED",
        solver_status="SATISFIABLE",
        cnf_uri=_ARTIFACT_A,
        detail="no proof was produced",
    )

    assert "conclusion" not in model.model_dump(mode="json")
    assert "conclusion" not in proof.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SatModelFindOutput(
            status="ASSIGNMENT_PRODUCED",
            solver_status="SATISFIABLE",
            cnf_uri=_ARTIFACT_A,
            assignment_uri=_ARTIFACT_A,
            detail="missing named assignment",
        )
    with pytest.raises(ValidationError):
        SatModelFindOutput(
            status="ASSIGNMENT_PRODUCED",
            solver_status="SATISFIABLE",
            cnf_uri=_ARTIFACT_A,
            assignment={"x": True},
            detail="missing candidate URI",
        )
    with pytest.raises(ValidationError):
        SatUnsatProofFindOutput(
            status="PROOF_PRODUCED",
            solver_status="UNSATISFIABLE",
            cnf_uri=_ARTIFACT_A,
            detail="missing proof URI",
        )
