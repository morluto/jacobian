from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest

from jacobian_checkers.matrix_normal_forms import check_hermite_normal_form


def _matrix(entries: list[list[int | str]]) -> dict[str, Any]:
    return {
        "matrix_schema_version": "1",
        "domain": "ZZ",
        "entries": [[str(value) for value in row] for row in entries],
    }


def _artifact(
    character: str,
    payload: dict[str, Any],
    *,
    semantics_uri: str,
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(character),
        "object_digest": "sha256:" + character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri("a"),
        "semantics_uri": semantics_uri,
        "parents": parents,
        "payload": payload,
    }


def _request(
    *,
    source: list[list[int | str]] | None = None,
    normal_form: list[list[int | str]] | None = None,
    transformation: list[list[int | str]] | None = None,
) -> dict[str, Any]:
    source = source or [[2, 4], [6, 8]]
    normal_form = normal_form or [[2, 0], [0, 4]]
    transformation = transformation or [[-2, 1], [3, -1]]
    semantics_uri = _uri("e")
    semantics = _artifact(
        "e",
        {"kind": "matrix.normal_form.hermite"},
        semantics_uri=_uri("0"),
        parents=[],
    )
    claim = _artifact(
        "a",
        {
            "matrix": _matrix(source),
            "resource_budget": {"budget_version": "1", "wall_seconds": 5},
        },
        semantics_uri=semantics_uri,
        parents=[],
    )
    candidate = _artifact(
        "c",
        {
            "result_schema_version": "1",
            "normal_form": _matrix(normal_form),
            "transformation": _matrix(transformation),
            "method": "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM",
            "backend": "python-flint",
            "backend_version": "0.9.0",
            "flint_library_version": "3.6.0",
        },
        semantics_uri=semantics_uri,
        parents=[claim["artifact_uri"]],
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": semantics["object_digest"],
        "candidate_digest": candidate["object_digest"],
        "scope_digest": None,
        "encoding_digest": None,
    }
    witness_payload = {
        "evidence_schema_version": "1",
        "witness_format": "matrix.normal_form.hermite",
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "bindings": bindings,
        "payload": {
            "operation_id": "matrix.normal_form.hermite.materialize",
            "input_uri": claim["artifact_uri"],
            "result_uri": candidate["artifact_uri"],
        },
    }
    witness = _artifact(
        "f",
        witness_payload,
        semantics_uri=semantics_uri,
        parents=[claim["artifact_uri"], candidate["artifact_uri"]],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "semantics": semantics,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


def _refresh_payload_digest(artifact: dict[str, Any]) -> None:
    artifact["payload_digest"] = _digest(artifact["payload"])


def test_checker_accepts_retained_hnf_certificate() -> None:
    decision = check_hermite_normal_form(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_INTEGER"


def test_checker_accepts_the_contract_scalar_bound() -> None:
    scalar = "1" + "0" * 299
    decision = check_hermite_normal_form(
        _request(
            source=[[scalar, "0"], ["0", scalar]],
            normal_form=[[scalar, "0"], ["0", scalar]],
            transformation=[["1", "0"], ["0", "1"]],
        )
    )

    assert decision["accepted"] is True


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 0], [1, 0]],
        [[0, 1], [1, 0]],
        [[-1, 0], [0, 1]],
        [[1, 2], [0, 2]],
    ],
)
def test_checker_rejects_each_row_hnf_structure_violation(
    matrix: list[list[int]],
) -> None:
    decision = check_hermite_normal_form(
        _request(
            source=matrix,
            normal_form=matrix,
            transformation=[[1, 0], [0, 1]],
        )
    )

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_mutation_even_with_refreshed_payload_digest() -> None:
    request = _request()
    request["candidate"]["payload"]["normal_form"]["entries"][0][1] = "1"
    _refresh_payload_digest(request["candidate"])

    decision = check_hermite_normal_form(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_rebound_source_identity() -> None:
    request = _request()
    request["witness"]["payload"]["payload"]["input_uri"] = _uri("7")
    _refresh_payload_digest(request["witness"])

    decision = check_hermite_normal_form(request)

    assert decision["accepted"] is False


def test_checker_rejects_malformed_or_noncanonical_payloads() -> None:
    requests = []

    extra_field = _request()
    extra_field["candidate"]["payload"]["unexpected"] = True
    _refresh_payload_digest(extra_field["candidate"])
    requests.append(extra_field)

    leading_zero = _request()
    leading_zero["candidate"]["payload"]["normal_form"]["entries"][0][0] = "02"
    _refresh_payload_digest(leading_zero["candidate"])
    requests.append(leading_zero)

    duplicate_parent = _request()
    duplicate_parent["witness"]["parents"].append(
        duplicate_parent["claim"]["artifact_uri"]
    )
    requests.append(duplicate_parent)

    for request in requests:
        decision = check_hermite_normal_form(copy.deepcopy(request))
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
