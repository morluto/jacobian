from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tests.support.artifacts import canonical_digest as _digest

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope
from jacobian.contracts.sat import (
    SatAssignmentArtifact,
    SatCnfBinding,
    SatResourceBudget,
    canonicalize_cnf,
)
from jacobian_checkers.sat import check_assignment

_CNF_URI = "artifact://sha256/" + "1" * 64
_ASSIGNMENT_URI = "artifact://sha256/" + "2" * 64
_WITNESS_URI = "artifact://sha256/" + "3" * 64
_SCHEMA_URI = "artifact://sha256/" + "4" * 64
_SEMANTICS_URI = "artifact://sha256/" + "5" * 64
_CNF_OBJECT_DIGEST = "sha256:" + "a" * 64
_ASSIGNMENT_OBJECT_DIGEST = "sha256:" + "b" * 64
_SEMANTICS_DIGEST = "sha256:" + "c" * 64


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


def _request() -> dict[str, Any]:
    cnf = canonicalize_cnf(
        variable_names=("a", "b"),
        clauses=((-1, 2), (1, 2)),
    )
    cnf_payload = cnf.model_dump(mode="json")
    cnf_payload_digest = _digest(cnf_payload)
    assignment = SatAssignmentArtifact(
        cnf=SatCnfBinding(
            cnf_artifact_uri=_CNF_URI,
            cnf_object_digest=_CNF_OBJECT_DIGEST,
            cnf_payload_digest=cnf_payload_digest,
            variable_map_digest=cnf.variable_map_digest,
            dimacs_digest=cnf.dimacs_digest,
            projection_format=cnf.projection_format,
            projection_version=cnf.projection_version,
            variable_count=2,
            clause_count=2,
        ),
        values=(False, True),
        producer=_producer(),
        resource_budget=SatResourceBudget(wall_seconds=30),
    )
    assignment_payload = assignment.model_dump(mode="json")
    bindings = EvidenceBindings(
        claim_digest=_CNF_OBJECT_DIGEST,
        semantics_digest=_SEMANTICS_DIGEST,
        candidate_digest=_ASSIGNMENT_OBJECT_DIGEST,
    )
    witness = WitnessEnvelope(
        witness_format="sat.assignment",
        format_version="1",
        role="SUPPORTS_CLAIM",
        bindings=bindings,
        payload={
            "cnf_uri": _CNF_URI,
            "assignment_uri": _ASSIGNMENT_URI,
        },
    )
    return {
        "request_version": "1",
        "claim": {
            "artifact_uri": _CNF_URI,
            "object_digest": _CNF_OBJECT_DIGEST,
            "payload_digest": cnf_payload_digest,
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [],
            "payload": cnf_payload,
        },
        "candidate": {
            "artifact_uri": _ASSIGNMENT_URI,
            "object_digest": _ASSIGNMENT_OBJECT_DIGEST,
            "payload_digest": _digest(assignment_payload),
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [_CNF_URI],
            "payload": assignment_payload,
        },
        "scope": None,
        "witness": {
            "artifact_uri": _WITNESS_URI,
            "object_digest": "sha256:" + "e" * 64,
            "payload_digest": "sha256:" + "f" * 64,
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [_ASSIGNMENT_URI, _CNF_URI],
            "payload": witness.model_dump(mode="json"),
        },
        "expected_bindings": bindings.model_dump(mode="json"),
    }


def test_checker_accepts_a_total_assignment_satisfying_every_clause() -> None:
    decision = check_assignment(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["method"] == "DIRECT_WITNESS"
    assert decision["coverage"] == "NOT_APPLICABLE"


def _mutate_cnf(request: dict[str, Any], mutation: str) -> None:
    cnf = request["claim"]["payload"]
    if mutation == "omit_clause":
        cnf["clauses"].pop()
    elif mutation == "add_clause":
        cnf["clauses"].append({"literals": [2]})
    elif mutation == "reorder_clauses":
        cnf["clauses"] = list(reversed(cnf["clauses"]))
    elif mutation == "renumber_literal":
        cnf["clauses"][0]["literals"][1] = 1
    elif mutation == "change_variable_map":
        cnf["variables"][1]["name"] = "c"


def _mutate_assignment(request: dict[str, Any], mutation: str) -> None:
    assignment = request["candidate"]["payload"]
    if mutation == "flip_value":
        assignment["values"][1] = False
    elif mutation == "change_source_uri":
        assignment["cnf"]["cnf_artifact_uri"] = "artifact://sha256/" + "9" * 64
    elif mutation == "partial_assignment":
        assignment["values"].pop()
    elif mutation == "boolean_variable_count":
        assignment["cnf"]["variable_count"] = True
    elif mutation == "extra_candidate_field":
        assignment["verification"] = "VERIFIED"


def _apply_assignment_mutation(mutation: str, request: dict[str, Any]) -> None:
    if mutation in {
        "omit_clause",
        "add_clause",
        "reorder_clauses",
        "renumber_literal",
        "change_variable_map",
    }:
        _mutate_cnf(request, mutation)
    elif mutation in {
        "flip_value",
        "change_source_uri",
        "partial_assignment",
        "boolean_variable_count",
        "extra_candidate_field",
    }:
        _mutate_assignment(request, mutation)
    elif mutation == "mismatched_bindings":
        request["witness"]["payload"]["bindings"]["candidate_digest"] = (
            "sha256:" + "9" * 64
        )
    elif mutation == "missing_lineage":
        request["candidate"]["parents"] = []


@pytest.mark.parametrize(
    "mutation",
    (
        "flip_value",
        "omit_clause",
        "add_clause",
        "reorder_clauses",
        "renumber_literal",
        "change_variable_map",
        "change_source_uri",
        "partial_assignment",
        "boolean_variable_count",
        "mismatched_bindings",
        "missing_lineage",
        "extra_candidate_field",
    ),
)
def test_checker_rejects_mutated_or_misbound_assignment_evidence(
    mutation: str,
) -> None:
    request = _request()
    _apply_assignment_mutation(mutation, request)

    decision = check_assignment(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_a_valid_assignment_rebound_to_another_cnf() -> None:
    request = _request()
    other = canonicalize_cnf(variable_names=("a", "b"), clauses=((1,),))
    request["claim"]["payload"] = other.model_dump(mode="json")
    request["claim"]["payload_digest"] = _digest(request["claim"]["payload"])
    request["claim"]["object_digest"] = "sha256:" + "8" * 64
    request["expected_bindings"]["claim_digest"] = request["claim"]["object_digest"]
    request["witness"]["payload"]["bindings"] = deepcopy(request["expected_bindings"])

    decision = check_assignment(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
