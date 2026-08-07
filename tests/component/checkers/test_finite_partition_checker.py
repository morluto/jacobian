from __future__ import annotations

from typing import Any

import pytest

from jacobian_checkers.finite_partition import check_partition


def _request(*, cases: list[dict[str, object]]) -> dict[str, Any]:
    bindings = {
        "claim_digest": "sha256:" + "1" * 64,
        "semantics_digest": "sha256:" + "2" * 64,
        "candidate_digest": "sha256:" + "3" * 64,
        "scope_digest": "sha256:" + "4" * 64,
        "encoding_digest": None,
    }
    return {
        "request_version": "1",
        "claim": {
            "artifact_uri": "artifact://sha256/" + "5" * 64,
            "payload": {
                "predicate": "finite_partition",
                "require_disjoint": True,
            },
        },
        "candidate": {
            "artifact_uri": "artifact://sha256/" + "6" * 64,
            "payload": {"cases": cases},
        },
        "scope": {
            "artifact_uri": "artifact://sha256/" + "7" * 64,
            "payload": {"elements": ["a", "b", "c", "d"]},
        },
        "certificate": {
            "payload": {
                "certificate_type": "finite.partition",
                "format_version": "1",
                "bindings": bindings,
                "payload": {
                    "relation_id": "case.relation.partitions",
                    "obligation_uri": "artifact://sha256/" + "5" * 64,
                },
            }
        },
        "expected_bindings": bindings,
    }


def test_finite_partition_checker_accepts_exact_partition() -> None:
    decision = check_partition(
        _request(
            cases=[
                {"case_id": "left", "members": ["a", "b"]},
                {"case_id": "right", "members": ["c", "d"]},
            ]
        )
    )

    assert decision["accepted"] is True
    assert decision["coverage"] == "EXHAUSTIVE"
    assert decision["method"] == "EXHAUSTIVE_FINITE"
    assert decision["relationship_source_artifact_uris"] == [
        "artifact://sha256/" + "7" * 64
    ]
    assert decision["relationship_target_artifact_uris"] == [
        "artifact://sha256/" + "6" * 64
    ]


def test_finite_partition_checker_rejects_gap_and_overlap() -> None:
    gap = check_partition(
        _request(cases=[{"case_id": "only", "members": ["a", "b", "c"]}])
    )
    overlap = check_partition(
        _request(
            cases=[
                {"case_id": "first", "members": ["a", "b", "c"]},
                {"case_id": "second", "members": ["c", "d"]},
            ]
        )
    )

    assert gap["accepted"] is False
    assert "cover" in gap["detail"]
    assert overlap["accepted"] is False
    assert "overlap" in overlap["detail"]


_PARTITION_CASES = [
    {"case_id": "left", "members": ["a", "b"]},
    {"case_id": "right", "members": ["c", "d"]},
]


def _unbound_relationship_metadata(request: dict) -> None:
    request["certificate"]["payload"]["payload"]["obligation_uri"] = (
        "artifact://sha256/" + "9" * 64
    )


def _rebound_bindings(request: dict) -> None:
    rebound = dict(request["certificate"]["payload"]["bindings"])
    rebound["scope_digest"] = "sha256:" + "9" * 64
    request["certificate"]["payload"]["bindings"] = rebound


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        (_unbound_relationship_metadata, "relationship metadata"),
        (_rebound_bindings, "bindings"),
    ],
    ids=["unbound_relationship_metadata", "binding_substitution"],
)
def test_finite_partition_checker_rejects_corrupted_certificate(
    mutation: Any, expected_detail: str
) -> None:
    request = _request(cases=_PARTITION_CASES)
    mutation(request)

    decision = check_partition(request)

    assert decision["accepted"] is False
    assert expected_detail in decision["detail"]


def test_finite_partition_checker_rejects_unknown_format() -> None:
    request = _request(
        cases=[
            {"case_id": "left", "members": ["a", "b"]},
            {"case_id": "right", "members": ["c", "d"]},
        ]
    )
    request["certificate"]["payload"]["format_version"] = "2"

    decision = check_partition(request)

    assert decision["accepted"] is False
    assert "format" in decision["detail"]
