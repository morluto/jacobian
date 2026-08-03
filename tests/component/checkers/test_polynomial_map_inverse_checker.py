from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri

from jacobian_checkers.polynomial_maps import check_map_inverse


def _term(coefficient: str, exponents: list[int]) -> dict[str, Any]:
    return {
        "coefficient": {"num": coefficient, "den": "1"},
        "exponents": exponents,
    }


def _map(variables: list[str], *, inverse: bool) -> dict[str, Any]:
    return {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": variables,
        "coordinates": [
            {
                "terms": [
                    _term("1", [1, 0]),
                    _term("-1" if inverse else "1", [0, 2]),
                ]
            },
            {"terms": [_term("1", [0, 1])]},
        ],
    }


def _request() -> dict[str, Any]:
    forward_uri = _uri("1")
    inverse_uri = _uri("2")
    residual_uri = _uri("3")
    left_records = [_uri("4"), _uri("5")]
    right_records = [_uri("6"), _uri("7")]
    bindings = {
        "claim_digest": "sha256:" + "8" * 64,
        "semantics_digest": "sha256:" + "9" * 64,
        "candidate_digest": "sha256:" + "a" * 64,
        "scope_digest": "sha256:" + "b" * 64,
        "encoding_digest": None,
    }
    replay = {
        "method": "DIRECT_TWO_SIDED_SPARSE_REPLAY",
        "forward_map_uri": forward_uri,
        "inverse_map_uri": inverse_uri,
        "residuals_uri": residual_uri,
        "source_variables": ["x", "y"],
        "target_variables": ["u", "v"],
        "inverse_after_forward_checker_records": left_records,
        "forward_after_inverse_checker_records": right_records,
    }
    return {
        "request_version": "1",
        "claim": {
            "artifact_uri": _uri("c"),
            "payload": {
                "claim_schema_version": "1",
                "predicate": "POLYNOMIAL_MAP_TWO_SIDED_INVERSE",
                "domain": "QQ",
                "forward_map_uri": forward_uri,
                "inverse_map_uri": inverse_uri,
                "source_variables": ["x", "y"],
                "target_variables": ["u", "v"],
            },
        },
        "scope": {
            "artifact_uri": forward_uri,
            "payload": _map(["x", "y"], inverse=False),
        },
        "candidate": {
            "artifact_uri": residual_uri,
            "payload": {
                "residual_schema_version": "1",
                "domain": "QQ",
                "forward_map_uri": forward_uri,
                "inverse_map_uri": inverse_uri,
                "source_variables": ["x", "y"],
                "target_variables": ["u", "v"],
                "inverse_after_forward": [{"terms": []}, {"terms": []}],
                "forward_after_inverse": [{"terms": []}, {"terms": []}],
                "inverse_after_forward_checker_records": left_records,
                "forward_after_inverse_checker_records": right_records,
            },
        },
        "certificate": {
            "payload": {
                "evidence_schema_version": "1",
                "certificate_type": "polynomial.map.inverse.two_sided_replay",
                "format_version": "1",
                "bindings": deepcopy(bindings),
                "payload_digest": "sha256:" + "d" * 64,
                "payload": replay,
            }
        },
        "supporting_artifacts": [
            {"artifact_uri": inverse_uri, "payload": _map(["u", "v"], inverse=True)},
            *[
                {"artifact_uri": uri, "payload": {}}
                for uri in (*left_records, *right_records)
            ],
        ],
        "expected_bindings": deepcopy(bindings),
    }


def test_inverse_checker_accepts_both_exact_compositions() -> None:
    decision = check_map_inverse(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    ("zero_direction", "forged_direction"),
    (
        ("inverse_after_forward", "forward_after_inverse"),
        ("forward_after_inverse", "inverse_after_forward"),
    ),
)
def test_inverse_checker_never_accepts_one_declared_identity_direction(
    zero_direction: str,
    forged_direction: str,
) -> None:
    request = _request()
    request["candidate"]["payload"][zero_direction] = [
        {"terms": []},
        {"terms": []},
    ]
    request["candidate"]["payload"][forged_direction] = [
        {"terms": [_term("1", [0, 0])]},
        {"terms": []},
    ]

    decision = check_map_inverse(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize("source", ("forward", "inverse"))
def test_inverse_checker_rejects_source_coefficient_tampering(source: str) -> None:
    request = _request()
    artifact = (
        request["scope"] if source == "forward" else request["supporting_artifacts"][0]
    )
    artifact["payload"]["coordinates"][0]["terms"][0]["coefficient"]["num"] = "2"

    decision = check_map_inverse(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize("tamper", ("domain", "source_order", "target_order"))
def test_inverse_checker_rejects_domain_and_order_mismatch(tamper: str) -> None:
    request = _request()
    if tamper == "domain":
        request["scope"]["payload"]["domain"] = "ZZ"
    elif tamper == "source_order":
        request["candidate"]["payload"]["source_variables"] = ["y", "x"]
    else:
        request["candidate"]["payload"]["target_variables"] = ["v", "u"]

    decision = check_map_inverse(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    "family",
    (
        "inverse_after_forward_checker_records",
        "forward_after_inverse_checker_records",
    ),
)
def test_inverse_checker_rejects_incomplete_checker_record_family(
    family: str,
) -> None:
    request = _request()
    request["candidate"]["payload"][family] = request["candidate"]["payload"][family][
        :-1
    ]
    request["certificate"]["payload"]["payload"][family] = request["certificate"][
        "payload"
    ]["payload"][family][:-1]

    decision = check_map_inverse(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
