from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import json_digest as _digest

from jacobian_checkers.matrix_normal_forms import check_hermite_normal_form


def _matrix(entries: list[list[int | str]]) -> dict[str, Any]:
    return {
        "matrix_schema_version": "1",
        "domain": "ZZ",
        "entries": [[str(value) for value in row] for row in entries],
    }


def _artifact(
    *,
    uri_character: str,
    object_character: str,
    schema_character: str,
    semantics_uri: str,
    payload: dict[str, Any],
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(uri_character),
        "object_digest": "sha256:" + object_character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri(schema_character),
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
    source_payload = _matrix(source)
    claim = _artifact(
        uri_character="a",
        object_character="1",
        schema_character="b",
        semantics_uri=semantics_uri,
        payload=source_payload,
        parents=[],
    )
    candidate_payload = {
        "normal_form_schema_version": "1",
        "source": {
            "binding_version": "1",
            "matrix_artifact_uri": claim["artifact_uri"],
            "matrix_object_digest": claim["object_digest"],
            "matrix_payload_digest": claim["payload_digest"],
            "row_count": len(source),
            "column_count": len(source[0]),
        },
        "declared_scope": "FULL_MATRIX",
        "normal_form": _matrix(normal_form),
        "transformation": _matrix(transformation),
        "producer": {
            "runtime_version": "1",
            "provider": "python-flint",
            "availability": "AVAILABLE",
            "version": "0.9.0",
            "digest": "sha256:" + "9" * 64,
            "digest_kind": "PYTHON_DISTRIBUTION_RECORD",
            "platform": "linux-x86_64",
            "install_tier": "T1",
            "license_id": "MIT AND LGPL-3.0-or-later",
            "license_files": ["python_flint-0.9.0.dist-info/licenses/LICENSE"],
            "features": [
                "exact-integer",
                "dense-matrix",
                "row-hermite-normal-form",
                "left-transformation",
            ],
            "checker_ids": [],
            "configuration": {
                "distribution": "python-flint",
                "domain": "ZZ",
                "operation": "fmpz_mat.hnf(transform=True)",
                "flint_library_version": "3.6.0",
                "maximum_rows": 32,
                "maximum_columns": 32,
                "normal_form_convention": "FLINT_ROW_HNF",
                "relation": "H=U*A",
            },
            "distribution_import_name": "flint",
            "distribution_required_attributes": ["fmpz", "fmpz_mat"],
            "diagnostic": "",
        },
        "resource_budget": {"budget_version": "1", "wall_seconds": 5},
        "method": "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM",
    }
    candidate = _artifact(
        uri_character="c",
        object_character="2",
        schema_character="d",
        semantics_uri=semantics_uri,
        payload=candidate_payload,
        parents=[claim["artifact_uri"]],
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": "sha256:" + "3" * 64,
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
            "matrix_uri": claim["artifact_uri"],
            "normal_form_uri": candidate["artifact_uri"],
        },
    }
    witness = _artifact(
        uri_character="f",
        object_character="4",
        schema_character="5",
        semantics_uri=semantics_uri,
        payload=witness_payload,
        parents=[claim["artifact_uri"], candidate["artifact_uri"]],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


def _refresh_payload_digest(artifact: dict[str, Any]) -> None:
    artifact["payload_digest"] = _digest(artifact["payload"])


def test_checker_accepts_exact_hnf_transformation_evidence() -> None:
    decision = check_hermite_normal_form(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_INTEGER"


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 0], [1, 0]],
        [[0, 1], [1, 0]],
        [[-1, 0], [0, 1]],
        [[1, 2], [0, 2]],
    ],
    ids=(
        "zero_row_not_last",
        "pivot_columns_not_increasing",
        "negative_pivot",
        "unreduced_above_pivot",
    ),
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
    request["candidate"]["payload"]["source"]["matrix_object_digest"] = (
        "sha256:" + "8" * 64
    )
    _refresh_payload_digest(request["candidate"])

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

    wrong_witness = _request()
    wrong_witness["witness"]["payload"]["payload"]["normal_form_uri"] = _uri("7")
    _refresh_payload_digest(wrong_witness["witness"])
    requests.append(wrong_witness)

    wrong_profile = _request()
    wrong_profile["candidate"]["payload"]["producer"]["configuration"]["relation"] = (
        "H=A*U"
    )
    _refresh_payload_digest(wrong_profile["candidate"])
    requests.append(wrong_profile)

    duplicate_parent = _request()
    duplicate_parent["witness"]["parents"].append(
        duplicate_parent["claim"]["artifact_uri"]
    )
    requests.append(duplicate_parent)

    for request in requests:
        decision = check_hermite_normal_form(copy.deepcopy(request))
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
