"""Deliberately simple untrusted search-plugin entrypoints."""

from __future__ import annotations

import time
from typing import Any


def evaluate_candidate(request: dict[str, Any]) -> dict[str, Any]:
    value = request["candidate"]["value"]
    return {
        "conclusion": "FALSE" if value == 3 else "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "objectives": {"violations": "1" if value == 3 else "0"},
        "features": {"value": str(value)},
        "failure_classifications": (["fixture_violation"] if value == 3 else []),
        "detail": "fixture evaluation",
    }


def enumerate_invalid_candidate(_request: dict[str, Any]) -> dict[str, Any]:
    """Return a complete page whose candidate violates the installed schema."""

    return {
        "response_version": "1",
        "candidates": [{"not": "a matrix"}],
        "next_cursor": None,
        "complete": True,
        "scope": {"fixture": "invalid candidate"},
    }


def find_fixture_witness(request: dict[str, Any]) -> dict[str, Any]:
    value = request["candidate"]["value"]
    return {
        "status": "FOUND",
        "witness": {"observed": str(value)},
        "witness_format": "fixture.value",
        "format_version": "1",
        "role": request["witness_role"],
        "arithmetic": "EXACT_INTEGER",
        "coverage": "NOT_APPLICABLE",
        "detail": "direct fixture witness",
    }


def propose_fixture_values(request: dict[str, Any]) -> dict[str, Any]:
    cursor = int(request["state"].get("cursor", 0))
    batch_size = int(request["batch_size"])
    stop = min(4, cursor + batch_size)
    return {
        "response_version": "1",
        "candidates": [{"value": value} for value in range(cursor, stop)],
        "state": {"cursor": stop},
        "complete": stop == 4,
        "detail": "finite fixture proposal",
    }


def propose_fixture_values_with_strategy_state(
    request: dict[str, Any],
) -> dict[str, Any]:
    response = propose_fixture_values(request)
    response["state"] = {**request["state"], **response["state"]}
    return response


def refine_fixture_search(request: dict[str, Any]) -> dict[str, Any]:
    feedback = request["feedback"]
    nominations = (
        [
            {
                "candidate_uri": feedback[0]["candidate_uri"],
                "reason": "first candidate in the evaluated batch",
            }
        ]
        if feedback
        else []
    )
    return {
        "response_version": "1",
        "state": request["state"],
        "nominations": nominations,
        "detail": "fixture refinement",
    }


def refine_with_bounded_previous_batch_nominations(
    request: dict[str, Any],
) -> dict[str, Any]:
    limit = int(request["max_additional_lineage_parents"])
    previous = request["state"].get("previous_candidate_uris", [])
    current = [item["candidate_uri"] for item in request["feedback"]]
    return {
        "response_version": "1",
        "state": {
            **request["state"],
            "previous_candidate_uris": current,
            "observed_lineage_parent_limit": limit,
        },
        "nominations": [
            {
                "candidate_uri": candidate_uri,
                "reason": "previously evaluated candidate",
            }
            for candidate_uri in previous[:limit]
        ],
        "detail": "nominated within the advertised lineage capacity",
    }


def refine_ignoring_previous_batch_nomination_limit(
    request: dict[str, Any],
) -> dict[str, Any]:
    previous = request["state"].get("previous_candidate_uris", [])
    current = [item["candidate_uri"] for item in request["feedback"]]
    return {
        "response_version": "1",
        "state": {**request["state"], "previous_candidate_uris": current},
        "nominations": [
            {
                "candidate_uri": candidate_uri,
                "reason": "previously evaluated candidate",
            }
            for candidate_uri in previous
        ],
        "detail": "ignored the advertised lineage capacity",
    }


def refine_from_verified_counterexample(request: dict[str, Any]) -> dict[str, Any]:
    feedback = request["feedback"]
    return {
        "response_version": "1",
        "state": {
            **request["state"],
            "saw_verified_counterexample": any(
                item["counterexample_verified"] for item in feedback
            ),
        },
        "nominations": [],
        "detail": "recorded independently verified feedback only",
    }


def propose_fixture_values_slowly(request: dict[str, Any]) -> dict[str, Any]:
    time.sleep(0.15)
    return propose_fixture_values(request)


def propose_search_forever(_request: dict[str, Any]) -> dict[str, Any]:
    time.sleep(60)
    return {"unreachable": True}


def propose_malformed_search(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_version": "1",
        "candidates": [],
        "state": {},
        "complete": False,
    }


def propose_declared_failure(_request: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("declared fixture failure")


def propose_large_search_output(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_version": "1",
        "candidates": [{"value": 0}],
        "state": {"padding": "x" * 4096},
        "complete": True,
    }


def propose_beyond_authority(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_version": "1",
        "candidates": [{"value": 0}, {"value": 1}],
        "state": {"cursor": 2},
        "complete": True,
    }


def propose_partially_invalid_search(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_version": "1",
        "candidates": [{"value": 1}, {"not_value": 2}],
        "state": {},
        "complete": True,
    }


def refine_with_verification_claim(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_version": "1",
        "state": request["state"],
        "nominations": [],
        "verification": "VERIFIED",
    }
