from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
from typing import Any

import pytest

from jacobian.canonical import canonicalize_json
from jacobian.contracts.sat import SatLratResourceLimits, canonicalize_cnf
from jacobian_checkers.sat_lrat import check_lrat


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _artifact(
    uri_suffix: str,
    payload: dict[str, Any],
    *,
    parents: tuple[str, ...] = (),
) -> dict[str, Any]:
    uri = "artifact://sha256/" + uri_suffix * 64
    return {
        "artifact_uri": uri,
        "object_digest": _sha(uri.encode()),
        "payload_digest": _sha(canonicalize_json(payload)),
        "schema_uri": "artifact://sha256/" + "e" * 64,
        "semantics_uri": "artifact://sha256/" + "f" * 64,
        "parents": list(parents),
        "payload": payload,
    }


def _request(
    proof: bytes = b"3 0 1 2 0\n",
    *,
    limits: SatLratResourceLimits | None = None,
) -> dict[str, Any]:
    cnf = canonicalize_cnf(variable_names=("x",), clauses=((-1,), (1,)))
    claim = _artifact("a", cnf.model_dump(mode="json"))
    limits = limits or SatLratResourceLimits()
    cnf_binding = {
        "binding_version": "1",
        "cnf_artifact_uri": claim["artifact_uri"],
        "cnf_object_digest": claim["object_digest"],
        "cnf_payload_digest": claim["payload_digest"],
        "variable_map_digest": claim["payload"]["variable_map_digest"],
        "dimacs_digest": claim["payload"]["dimacs_digest"],
        "projection_format": claim["payload"]["projection_format"],
        "projection_version": claim["payload"]["projection_version"],
        "variable_count": len(claim["payload"]["variables"]),
        "clause_count": len(claim["payload"]["clauses"]),
    }
    encoded = base64.b64encode(proof).decode("ascii")
    candidate_payload = {
        "proof_format": "LRAT-ASCII",
        "proof_format_version": "jacobian.lrat.rup/v1",
        "cnf": cnf_binding,
        "proof_base64": encoded,
        "proof_digest": _sha(proof),
        "proof_byte_count": len(proof),
        "limits": limits.model_dump(mode="json"),
    }
    candidate = _artifact("b", candidate_payload, parents=(claim["artifact_uri"],))
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": _sha(b"semantics"),
        "candidate_digest": candidate["object_digest"],
        "scope_digest": None,
        "encoding_digest": None,
    }
    certificate_inner = {
        "cnf_uri": claim["artifact_uri"],
        "proof_uri": candidate["artifact_uri"],
        "proof_digest": _sha(proof),
        "limits": limits.model_dump(mode="json"),
    }
    certificate_payload = {
        "certificate_type": "sat.lrat-proof",
        "format_version": "1",
        "bindings": bindings,
        "payload": certificate_inner,
        "payload_digest": _sha(canonicalize_json(certificate_inner)),
    }
    certificate = _artifact(
        "c",
        certificate_payload,
        parents=(claim["artifact_uri"], candidate["artifact_uri"]),
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "scope": None,
        "certificate": certificate,
        "expected_bindings": bindings,
    }


def test_lrat_checker_accepts_the_exact_frozen_binding_shape() -> None:
    decision = check_lrat(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    ("proof", "limits", "message"),
    (
        (
            b"3 0 1 2 0\n4 0 1 2 0\n",
            SatLratResourceLimits(max_steps=1),
            "max_steps",
        ),
        (
            b"3 1 0 1 2 0\n",
            SatLratResourceLimits(max_clause_literals=0),
            "max_clause_literals",
        ),
    ),
)
def test_lrat_checker_propagates_resource_exhaustion(
    proof: bytes,
    limits: SatLratResourceLimits,
    message: str,
) -> None:
    with pytest.raises(OverflowError, match=message):
        check_lrat(_request(proof, limits=limits))


def test_lrat_checker_rejects_binding_and_lineage_mutations() -> None:
    original = _request()
    mutations: list[dict[str, Any]] = []

    changed = deepcopy(original)
    changed["claim"]["payload"]["clauses"] = [{"literals": [-1]}]
    changed["claim"]["payload_digest"] = _sha(
        canonicalize_json(changed["claim"]["payload"])
    )
    mutations.append(changed)

    changed = deepcopy(original)
    changed["candidate"]["payload"]["cnf"]["dimacs_digest"] = _sha(b"forged")
    changed["candidate"]["payload_digest"] = _sha(
        canonicalize_json(changed["candidate"]["payload"])
    )
    mutations.append(changed)

    changed = deepcopy(original)
    changed["candidate"]["payload"]["proof_base64"] = base64.b64encode(
        b"3 0 1 0\n"
    ).decode("ascii")
    changed["candidate"]["payload_digest"] = _sha(
        canonicalize_json(changed["candidate"]["payload"])
    )
    mutations.append(changed)

    changed = deepcopy(original)
    changed["certificate"]["parents"] = []
    mutations.append(changed)

    changed = deepcopy(original)
    changed["expected_bindings"]["candidate_digest"] = _sha(b"forged-candidate")
    mutations.append(changed)

    for mutation in mutations:
        decision = check_lrat(mutation)
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
