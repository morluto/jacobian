from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from tests.helpers.artifacts import artifact_uri as _uri
from tests.helpers.artifacts import canonical_digest as _digest
from tests.helpers.rationals import rational_payload as _q

from jacobian_checkers.exact_probability_operations import (
    check_finite_condition,
    check_finite_convolution,
    check_finite_event_probability,
    check_finite_pushforward,
    check_finite_raw_moment,
)

_META = {
    "exactness": "EXACT_RATIONAL",
    "determinism": "DETERMINISTIC",
    "backend": "python-flint",
    "backend_version": "0.9.0",
    "verification": "UNVERIFIED",
}
_BIT = {
    "atoms": [
        {"value": _q(0), "probability": _q(1, 2)},
        {"value": _q(1), "probability": _q(1, 2)},
    ]
}


def _artifact(
    character: str,
    payload: dict[str, Any],
    *,
    semantics: str,
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(character),
        "object_digest": "sha256:" + character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri(chr(ord(character) + 1)),
        "semantics_uri": semantics,
        "parents": parents,
        "payload": payload,
    }


def _request(
    operation_id: str,
    witness_format: str,
    source: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    semantics = _uri("e")
    semantics_artifact = _artifact(
        "e", {"kind": "semantics"}, semantics=_uri("0"), parents=[]
    )
    semantics_artifact["object_digest"] = "sha256:" + "8" * 64
    claim = _artifact("1", source, semantics=semantics, parents=[])
    candidate = _artifact(
        "3", result, semantics=semantics, parents=[claim["artifact_uri"]]
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": semantics_artifact["object_digest"],
        "candidate_digest": candidate["object_digest"],
        "scope_digest": None,
        "encoding_digest": None,
    }
    witness_payload = {
        "evidence_schema_version": "1",
        "witness_format": witness_format,
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "bindings": bindings,
        "payload": {
            "operation_id": operation_id,
            "input_uri": claim["artifact_uri"],
            "result_uri": candidate["artifact_uri"],
        },
    }
    witness = _artifact(
        "5",
        witness_payload,
        semantics=semantics,
        parents=[claim["artifact_uri"], candidate["artifact_uri"]],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "semantics": semantics_artifact,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
    (
        check_finite_raw_moment,
        _request(
            "probability.finite_distribution.raw_moment.compute",
            "probability.finite-raw-moment.fraction-replay",
            {"atoms": _BIT["atoms"], "order": 2},
            {
                "order": 2,
                "moment": _q(1, 2),
                "contributions": [
                    {
                        "value": _q(0),
                        "probability": _q(1, 2),
                        "powered_value": _q(0),
                        "contribution": _q(0),
                    },
                    {
                        "value": _q(1),
                        "probability": _q(1, 2),
                        "powered_value": _q(1),
                        "contribution": _q(1, 2),
                    },
                ],
                **_META,
            },
        ),
    ),
    (
        check_finite_event_probability,
        _request(
            "probability.finite_distribution.event_probability.compute",
            "probability.finite-event.fraction-replay",
            {"distribution": _BIT, "event_values": [_q(1)]},
            {
                "event_probability": _q(1, 2),
                "selected_atoms": [_BIT["atoms"][1]],
                **_META,
            },
        ),
    ),
    (
        check_finite_condition,
        _request(
            "probability.finite_distribution.condition.compute",
            "probability.finite-condition.fraction-replay",
            {"distribution": _BIT, "event_values": [_q(1)]},
            {
                "event_probability": _q(1, 2),
                "distribution": {"atoms": [{"value": _q(1), "probability": _q(1)}]},
                "contributions": [
                    {
                        "value": _q(1),
                        "source_probability": _q(1, 2),
                        "conditioned_probability": _q(1),
                    }
                ],
                **_META,
            },
        ),
    ),
    (
        check_finite_pushforward,
        _request(
            "probability.finite_distribution.pushforward.compute",
            "probability.finite-pushforward.fraction-replay",
            {
                "distribution": _BIT,
                "mapping": [
                    {"source": _q(0), "target": _q(0)},
                    {"source": _q(1), "target": _q(0)},
                ],
            },
            {
                "distribution": {"atoms": [{"value": _q(0), "probability": _q(1)}]},
                "contributions": [
                    {"source": _q(0), "target": _q(0), "probability": _q(1, 2)},
                    {"source": _q(1), "target": _q(0), "probability": _q(1, 2)},
                ],
                **_META,
            },
        ),
    ),
    (
        check_finite_convolution,
        _request(
            "probability.finite_distribution.convolution.compute",
            "probability.finite-convolution.fraction-replay",
            {"left": _BIT, "right": _BIT},
            {
                "distribution": {
                    "atoms": [
                        {"value": _q(0), "probability": _q(1, 4)},
                        {"value": _q(1), "probability": _q(1, 2)},
                        {"value": _q(2), "probability": _q(1, 4)},
                    ]
                },
                "contributions": [
                    {
                        "left_value": _q(left),
                        "right_value": _q(right),
                        "sum_value": _q(left + right),
                        "probability": _q(1, 4),
                    }
                    for left in (0, 1)
                    for right in (0, 1)
                ],
                "independence": "PRODUCT_MEASURE",
                **_META,
            },
        ),
    ),
)


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_probability_checkers_accept_complete_exact_replay(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    checked = checker(case_request)

    assert checked["accepted"] is True
    assert checked["conclusion"] == "TRUE"
    assert "Fraction replay accepted" in checked["detail"]


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_probability_checkers_reject_payload_substitution_with_fresh_digest(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    forged = copy.deepcopy(case_request)
    forged["candidate"]["payload"]["backend_version"] = "0.9.1"
    forged["candidate"]["payload_digest"] = _digest(forged["candidate"]["payload"])

    checked = checker(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_convolution_checker_rejects_missing_pair_with_fresh_digest() -> None:
    checker, case_request = _CASES[-1]
    forged = copy.deepcopy(case_request)
    forged["candidate"]["payload"]["contributions"].pop()
    forged["candidate"]["payload_digest"] = _digest(forged["candidate"]["payload"])

    checked = checker(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"
