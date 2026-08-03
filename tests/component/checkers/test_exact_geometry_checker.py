from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import json_digest as _digest
from tests.support.rationals import rational_payload as _rational

from jacobian_checkers.exact_geometry import check_exact_geometry


def _point(x: int, y: int) -> dict[str, dict[str, str]]:
    return {"x": _rational(x), "y": _rational(y)}


def _segment(
    start: dict[str, object],
    end: dict[str, object],
) -> dict[str, object]:
    return {"start": start, "end": end}


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
    operation_id: str = "geometry.points.compute.squared_distance",
) -> dict[str, Any]:
    semantics_uri = _uri("e")
    semantics = _artifact(
        uri_character="e",
        object_character="3",
        schema_character="6",
        semantics_uri=_uri("0"),
        payload={"kind": "semantics"},
        parents=[],
    )
    claim_payload: dict[str, Any]
    candidate_payload: dict[str, Any]
    if operation_id == "geometry.points.compute.squared_distance":
        claim_payload = {"first": _point(1, 2), "second": _point(4, 6)}
        candidate_payload = {"value": _rational(25)}
    elif operation_id == "geometry.points.compute.convex_hull":
        claim_payload = {
            "points": [
                _point(1, 1),
                _point(0, 0),
                _point(2, 0),
                _point(2, 2),
                _point(0, 2),
            ]
        }
        candidate_payload = {
            "points": [
                _point(0, 0),
                _point(2, 0),
                _point(2, 2),
                _point(0, 2),
            ]
        }
    elif operation_id == "geometry.segment.compute.midpoint":
        claim_payload = {"first": _point(1, 2), "second": _point(5, 8)}
        candidate_payload = {"point": _point(3, 5)}
    elif operation_id == "geometry.segments.intersection.compute":
        claim_payload = {
            "first": _segment(_point(0, 0), _point(2, 2)),
            "second": _segment(_point(0, 2), _point(2, 0)),
        }
        candidate_payload = {
            "status": "POINT",
            "point": _point(1, 1),
            "contact_kind": "PROPER",
            "overlap": None,
        }
    elif operation_id == "geometry.polygon.simple.decide":
        claim_payload = {
            "points": [
                _point(0, 0),
                _point(2, 2),
                _point(0, 2),
                _point(2, 0),
            ]
        }
        candidate_payload = {
            "vertex_count": 4,
            "is_simple": False,
            "checked_edge_pairs": 2,
            "witness": {
                "first_edge_index": 0,
                "second_edge_index": 2,
                "intersection": {
                    "status": "POINT",
                    "point": _point(1, 1),
                    "contact_kind": "PROPER",
                    "overlap": None,
                },
            },
        }
    elif operation_id == "geometry.polygon.point.classify":
        claim_payload = {
            "polygon": {
                "points": [
                    _point(0, 0),
                    _point(2, 0),
                    _point(2, 2),
                    _point(0, 2),
                ]
            },
            "point": _point(1, 1),
        }
        candidate_payload = {
            "polygon_vertex_count": 4,
            "classification": "INSIDE",
            "boundary_edge_index": None,
        }
    elif operation_id == "geometry.triangle.compute.orientation":
        claim_payload = {
            "first": _point(0, 0),
            "second": _point(2, 0),
            "third": _point(1, 3),
        }
        candidate_payload = {"orientation": 1}
    else:
        claim_payload = {
            "first": _point(0, 0),
            "second": _point(3, 0),
            "third": _point(0, 6),
        }
        candidate_payload = {"point": _point(1, 2)}
    claim = _artifact(
        uri_character="a",
        object_character="1",
        schema_character="b",
        semantics_uri=semantics_uri,
        payload=claim_payload,
        parents=[],
    )
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
        "witness_format": "geometry.exact_rational_result",
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "bindings": copy.deepcopy(bindings),
        "payload": {
            "operation_id": operation_id,
            "input_uri": claim["artifact_uri"],
            "result_uri": candidate["artifact_uri"],
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
        "semantics": semantics,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


def _refresh(artifact: dict[str, Any]) -> None:
    artifact["payload_digest"] = _digest(artifact["payload"])


@pytest.mark.parametrize(
    "operation_id",
    [
        "geometry.points.compute.squared_distance",
        "geometry.points.compute.convex_hull",
        "geometry.segment.compute.midpoint",
        "geometry.segments.intersection.compute",
        "geometry.polygon.simple.decide",
        "geometry.polygon.point.classify",
        "geometry.triangle.compute.orientation",
        "geometry.triangle.compute.centroid",
    ],
)
def test_checker_accepts_selected_exact_geometry_results(operation_id: str) -> None:
    decision = check_exact_geometry(_request(operation_id))

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_RATIONAL"


def test_checker_rejects_candidate_mutation_with_fresh_digest() -> None:
    request = _request()
    request["candidate"]["payload"]["value"] = _rational(24)
    _refresh(request["candidate"])

    decision = check_exact_geometry(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("operation_id", "mutation"),
    (
        (
            "geometry.points.compute.convex_hull",
            lambda payload: payload["points"].pop(),
        ),
        (
            "geometry.segments.intersection.compute",
            lambda payload: payload.update(contact_kind="ENDPOINT_TOUCH"),
        ),
        (
            "geometry.polygon.simple.decide",
            lambda payload: payload.update(
                vertex_count=4,
                is_simple=True,
                checked_edge_pairs=6,
                witness=None,
            ),
        ),
        (
            "geometry.polygon.point.classify",
            lambda payload: payload.update(classification="OUTSIDE"),
        ),
    ),
)
def test_checker_rejects_planar_mutation_with_fresh_digest(
    operation_id: str,
    mutation: Any,
) -> None:
    request = _request(operation_id)
    mutation(request["candidate"]["payload"])
    _refresh(request["candidate"])

    decision = check_exact_geometry(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_source_mutation_with_fresh_digest() -> None:
    request = _request()
    request["claim"]["payload"]["second"] = _point(5, 6)
    _refresh(request["claim"])

    assert check_exact_geometry(request)["accepted"] is False


def test_checker_rejects_operation_substitution() -> None:
    request = _request("geometry.segment.compute.midpoint")
    request["witness"]["payload"]["payload"]["operation_id"] = (
        "geometry.points.compute.squared_distance"
    )
    _refresh(request["witness"])

    assert check_exact_geometry(request)["accepted"] is False


def test_checker_rejects_forged_semantics_digest() -> None:
    request = _request()
    forged = "sha256:" + "9" * 64
    request["expected_bindings"]["semantics_digest"] = forged
    request["witness"]["payload"]["bindings"]["semantics_digest"] = forged
    _refresh(request["witness"])

    assert check_exact_geometry(request)["accepted"] is False


def test_checker_rejects_tampered_semantics_payload() -> None:
    request = _request()
    request["semantics"]["payload"]["kind"] = "tampered-semantics"

    decision = check_exact_geometry(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request["candidate"].update(semantics_uri=_uri("9")),
        lambda request: request["candidate"]["parents"].append(_uri("8")),
        lambda request: request["claim"]["payload"]["first"]["x"].update(num="01"),
        lambda request: request["witness"]["payload"]["bindings"].update(
            candidate_digest="sha256:" + "7" * 64
        ),
    ],
    ids=(
        "wrong-semantics",
        "ambiguous-lineage",
        "noncanonical-rational",
        "rebound-witness",
    ),
)
def test_checker_rejects_semantic_or_binding_mutations(mutation: Any) -> None:
    request = copy.deepcopy(_request())
    mutation(request)
    for artifact in (request["claim"], request["candidate"], request["witness"]):
        _refresh(artifact)

    assert check_exact_geometry(request)["accepted"] is False
