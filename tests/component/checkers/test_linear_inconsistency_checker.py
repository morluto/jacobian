from __future__ import annotations

import copy
from typing import Any

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
    LinearRationalInconsistencyArtifact,
    LinearRationalResourceBudget,
    LinearRationalSystem,
    LinearSystemBinding,
    linear_variable_order_digest,
)
from jacobian_checkers.linear import check_rational_inconsistency

_SYSTEM_URI = "artifact://sha256/" + "1" * 64
_CERTIFICATE_URI = "artifact://sha256/" + "2" * 64
_WITNESS_URI = "artifact://sha256/" + "3" * 64
_SCHEMA_URI = "artifact://sha256/" + "4" * 64
_SEMANTICS_URI = "artifact://sha256/" + "5" * 64
_SYSTEM_OBJECT_DIGEST = "sha256:" + "a" * 64
_CERTIFICATE_OBJECT_DIGEST = "sha256:" + "b" * 64
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
                    [_q(1), _q(1)],
                    [_q(2), _q(2)],
                ]
            },
            "rhs": [_q(1), _q(3)],
        }
    )
    system_payload = system.model_dump(mode="json")
    system_payload_digest = _digest(system_payload)
    certificate = LinearRationalInconsistencyArtifact(
        system=LinearSystemBinding(
            system_artifact_uri=_SYSTEM_URI,
            system_object_digest=_SYSTEM_OBJECT_DIGEST,
            system_payload_digest=system_payload_digest,
            variable_order_digest=linear_variable_order_digest(system.variables),
            row_count=2,
            column_count=2,
        ),
        left_witness=(_q(-2), _q(1)),
        rhs_pairing=_q(1),
        producer=_producer(),
        resource_budget=LinearRationalResourceBudget(wall_seconds=5),
    )
    certificate_payload = certificate.model_dump(mode="json")
    bindings = EvidenceBindings(
        claim_digest=_SYSTEM_OBJECT_DIGEST,
        semantics_digest=_SEMANTICS_DIGEST,
        candidate_digest=_CERTIFICATE_OBJECT_DIGEST,
    )
    witness = WitnessEnvelope(
        witness_format="linear.rational_inconsistency",
        format_version="1",
        role="SUPPORTS_CLAIM",
        bindings=bindings,
        payload={
            "system_uri": _SYSTEM_URI,
            "certificate_uri": _CERTIFICATE_URI,
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
            "artifact_uri": _CERTIFICATE_URI,
            "object_digest": _CERTIFICATE_OBJECT_DIGEST,
            "payload_digest": _digest(certificate_payload),
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [_SYSTEM_URI],
            "payload": certificate_payload,
        },
        "scope": None,
        "witness": {
            "artifact_uri": _WITNESS_URI,
            "object_digest": "sha256:" + "e" * 64,
            "payload_digest": _digest(witness_payload),
            "schema_uri": _SCHEMA_URI,
            "semantics_uri": _SEMANTICS_URI,
            "parents": [_CERTIFICATE_URI, _SYSTEM_URI],
            "payload": witness_payload,
        },
        "expected_bindings": bindings.model_dump(mode="json"),
    }


def test_checker_accepts_exact_left_nullspace_inconsistency_witness() -> None:
    decision = check_rational_inconsistency(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["relation_id"] == "linear.relation.inconsistency-certificate-of"


def test_checker_rejects_mutated_witness_even_with_refreshed_digest() -> None:
    request = _request()
    request["candidate"]["payload"]["left_witness"][0] = _q(-3)
    request["candidate"]["payload_digest"] = _digest(request["candidate"]["payload"])

    decision = check_rational_inconsistency(copy.deepcopy(request))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_forged_pairing_and_rebound_system() -> None:
    forged_pairing = _request()
    forged_pairing["candidate"]["payload"]["rhs_pairing"] = _q(2)
    forged_pairing["candidate"]["payload_digest"] = _digest(
        forged_pairing["candidate"]["payload"]
    )

    rebound = _request()
    rebound["candidate"]["payload"]["system"]["system_object_digest"] = (
        "sha256:" + "9" * 64
    )
    rebound["candidate"]["payload_digest"] = _digest(rebound["candidate"]["payload"])

    for request in (forged_pairing, rebound):
        decision = check_rational_inconsistency(copy.deepcopy(request))
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
