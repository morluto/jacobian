from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tests.support.artifacts import canonical_digest as _digest
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope
from jacobian.contracts.linear import (
    LinearRationalResourceBudget,
    LinearRationalSolutionArtifact,
    LinearRationalSystem,
    LinearSystemBinding,
    linear_variable_order_digest,
)
from jacobian_checkers.linear import check_rational_solution

_SYSTEM_URI = "artifact://sha256/" + "1" * 64
_SOLUTION_URI = "artifact://sha256/" + "2" * 64
_WITNESS_URI = "artifact://sha256/" + "3" * 64
_SCHEMA_URI = "artifact://sha256/" + "4" * 64
_SEMANTICS_URI = "artifact://sha256/" + "5" * 64
_SYSTEM_OBJECT_DIGEST = "sha256:" + "a" * 64
_SOLUTION_OBJECT_DIGEST = "sha256:" + "b" * 64
_SEMANTICS_DIGEST = "sha256:" + "c" * 64


def _producer() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="python-flint",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="0.9.0",
        digest="sha256:" + "d" * 64,
        digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        license_files=("python_flint-0.9.0.dist-info/licenses/LICENSE",),
        features=("exact-rational", "dense-matrix", "reduced-row-echelon-form"),
        configuration={
            "distribution": "python-flint",
            "domain": "QQ",
            "operation": "fmpq_mat.rref",
            "maximum_rows": 32,
            "maximum_columns": 32,
            "free_variable_policy": "ZERO",
        },
    )


def _request() -> dict[str, Any]:
    system = LinearRationalSystem.model_validate(
        {
            "variables": ["x", "y"],
            "coefficients": {
                "entries": [
                    [_q(2), _q(1)],
                    [_q(1), _q(-1)],
                ]
            },
            "rhs": [_q(5), _q(1)],
        }
    )
    system_payload = system.model_dump(mode="json")
    system_payload_digest = _digest(system_payload)
    solution = LinearRationalSolutionArtifact(
        system=LinearSystemBinding(
            system_artifact_uri=_SYSTEM_URI,
            system_object_digest=_SYSTEM_OBJECT_DIGEST,
            system_payload_digest=system_payload_digest,
            variable_order_digest=linear_variable_order_digest(system.variables),
            row_count=2,
            column_count=2,
        ),
        values=(_q(2), _q(1)),
        producer=_producer(),
        resource_budget=LinearRationalResourceBudget(wall_seconds=5),
    )
    solution_payload = solution.model_dump(mode="json")
    bindings = EvidenceBindings(
        claim_digest=_SYSTEM_OBJECT_DIGEST,
        semantics_digest=_SEMANTICS_DIGEST,
        candidate_digest=_SOLUTION_OBJECT_DIGEST,
    )
    witness = WitnessEnvelope(
        witness_format="linear.rational_solution",
        format_version="1",
        role="SUPPORTS_CLAIM",
        bindings=bindings,
        payload={
            "system_uri": _SYSTEM_URI,
            "solution_uri": _SOLUTION_URI,
        },
    )
    witness_payload = witness.model_dump(mode="json")
    return {
        "request_version": "1",
        "claim": {
            "artifact_uri": _SYSTEM_URI,
            "object_digest": _SYSTEM_OBJECT_DIGEST,
            "payload_digest": system_payload_digest,
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [],
            "payload": system_payload,
        },
        "candidate": {
            "artifact_uri": _SOLUTION_URI,
            "object_digest": _SOLUTION_OBJECT_DIGEST,
            "payload_digest": _digest(solution_payload),
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [_SYSTEM_URI],
            "payload": solution_payload,
        },
        "scope": None,
        "witness": {
            "artifact_uri": _WITNESS_URI,
            "object_digest": "sha256:" + "e" * 64,
            "payload_digest": _digest(witness_payload),
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [_SOLUTION_URI, _SYSTEM_URI],
            "payload": witness_payload,
        },
        "expected_bindings": bindings.model_dump(mode="json"),
    }


def test_checker_accepts_vector_satisfying_every_exact_equation() -> None:
    decision = check_rational_solution(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_RATIONAL"
    assert decision["method"] == "DIRECT_WITNESS"


def _mutate_system(request: dict[str, Any], mutation: str) -> None:
    system = request["claim"]["payload"]
    if mutation == "omit_equation":
        system["coefficients"]["entries"].pop()
        system["rhs"].pop()
    elif mutation == "change_rhs":
        system["rhs"][0] = _q(6)
    elif mutation == "change_variable_order":
        system["variables"] = ["y", "x"]
    request["claim"]["payload_digest"] = _digest(system)


def _mutate_solution(request: dict[str, Any], mutation: str) -> None:
    solution = request["candidate"]["payload"]
    if mutation == "wrong_value":
        solution["values"][0] = _q(3)
    elif mutation == "partial_vector":
        solution["values"].pop()
    elif mutation == "noncanonical_rational":
        solution["values"][0] = {"num": "4", "den": "2"}
    elif mutation == "change_source_uri":
        solution["system"]["system_artifact_uri"] = "artifact://sha256/" + "9" * 64
    elif mutation == "wrong_provider":
        solution["producer"]["provider"] = "sympy"
    elif mutation == "extra_candidate_field":
        solution["verification"] = "VERIFIED"
    request["candidate"]["payload_digest"] = _digest(solution)


def _apply_solution_mutation(mutation: str, request: dict[str, Any]) -> None:
    if mutation in {"omit_equation", "change_rhs", "change_variable_order"}:
        _mutate_system(request, mutation)
    elif mutation in {
        "wrong_value",
        "partial_vector",
        "noncanonical_rational",
        "change_source_uri",
        "wrong_provider",
        "extra_candidate_field",
    }:
        _mutate_solution(request, mutation)
    elif mutation == "mismatched_bindings":
        request["witness"]["payload"]["bindings"]["candidate_digest"] = (
            "sha256:" + "9" * 64
        )
        request["witness"]["payload_digest"] = _digest(request["witness"]["payload"])
    elif mutation == "missing_candidate_lineage":
        request["candidate"]["parents"] = []
    elif mutation == "missing_witness_lineage":
        request["witness"]["parents"] = [_SOLUTION_URI]


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_value",
        "omit_equation",
        "change_rhs",
        "change_variable_order",
        "partial_vector",
        "noncanonical_rational",
        "change_source_uri",
        "mismatched_bindings",
        "missing_candidate_lineage",
        "missing_witness_lineage",
        "wrong_provider",
        "extra_candidate_field",
    ),
)
def test_checker_rejects_mutated_or_misbound_solution_evidence(
    mutation: str,
) -> None:
    request = _request()
    _apply_solution_mutation(mutation, request)

    decision = check_rational_solution(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_valid_vector_rebound_to_another_system() -> None:
    request = _request()
    request["claim"]["payload"]["rhs"] = [_q(4), _q(1)]
    request["claim"]["payload_digest"] = _digest(request["claim"]["payload"])
    request["claim"]["object_digest"] = "sha256:" + "8" * 64
    request["expected_bindings"]["claim_digest"] = request["claim"]["object_digest"]
    request["witness"]["payload"]["bindings"] = deepcopy(request["expected_bindings"])
    request["witness"]["payload_digest"] = _digest(request["witness"]["payload"])

    decision = check_rational_solution(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
