from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from typing import Any

import pytest
from tests.unit.contracts.artifacts import artifact_uri as _uri
from tests.unit.contracts.artifacts import canonical_digest as _digest
from tests.unit.contracts.rationals import rational_payload as _q

import jacobian_checkers.recurrence_series as checker_module
from jacobian_checkers.recurrence_series import (
    check_linear_recurrence_evaluation,
    check_rational_generating_function_coefficients,
)

_META = {
    "exactness": "EXACT_RATIONAL",
    "determinism": "DETERMINISTIC",
    "backend": "sympy",
    "backend_version": "1.14.0",
    "verification": "UNVERIFIED",
}
_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"


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
    witness = _artifact(
        "5",
        {
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
        },
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


_RECURRENCE_VALUES = (0, 1, 1, 2, 3, 5)
_SERIES_VALUES = (1, 1, 2, 3, 5, 8)
_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
    (
        check_linear_recurrence_evaluation,
        _request(
            "combinatorics.recurrence.linear.evaluate",
            "combinatorics.linear-recurrence.fraction-replay",
            {
                "coefficients": [_q(1), _q(1)],
                "initial_values": [_q(0), _q(1)],
                "coefficient_convention": _RECURRENCE_CONVENTION,
                "scope": "PREFIX",
                "term_count": 6,
                "indices": [],
            },
            {
                "coefficient_convention": _RECURRENCE_CONVENTION,
                "scope": "PREFIX",
                "values": [
                    {"index": index, "value": _q(value)}
                    for index, value in enumerate(_RECURRENCE_VALUES)
                ],
                "replay_prefix": [_q(value) for value in _RECURRENCE_VALUES],
                "replay_scope_end": 5,
                **_META,
            },
        ),
    ),
    (
        check_rational_generating_function_coefficients,
        _request(
            "combinatorics.generating_function.coefficients.compute",
            "combinatorics.rational-series.fraction-residual-replay",
            {
                "numerator": [_q(1)],
                "denominator": [_q(1), _q(-1), _q(-1)],
                "coefficient_convention": "ASCENDING_POWERS_OF_X",
                "expansion_point": "0",
                "truncation_order": 6,
            },
            {
                "coefficient_convention": "ASCENDING_POWERS_OF_X",
                "expansion_point": "0",
                "truncation_order": 6,
                "coefficients": [_q(value) for value in _SERIES_VALUES],
                "residual_congruence": (
                    "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
                ),
                "residual_coefficients": [_q(0) for _ in range(6)],
                **_META,
            },
        ),
    ),
)


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_recurrence_series_checkers_accept_complete_exact_replay(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    checked = checker(case_request)
    assert checked["accepted"] is True
    assert checked["conclusion"] == "TRUE"
    assert "Fraction replay accepted" in checked["detail"]


@pytest.mark.parametrize(("checker", "case_request"), _CASES)
def test_recurrence_series_checkers_reject_false_candidates_with_fresh_digest(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    case_request: dict[str, Any],
) -> None:
    forged = copy.deepcopy(case_request)
    if checker is check_linear_recurrence_evaluation:
        forged["candidate"]["payload"]["replay_prefix"][5] = _q(6)
        forged["candidate"]["payload"]["values"][5]["value"] = _q(6)
    else:
        forged["candidate"]["payload"]["coefficients"][5] = _q(9)
    forged["candidate"]["payload_digest"] = _digest(forged["candidate"]["payload"])

    checked = checker(forged)

    assert checked["accepted"] is False
    assert checked["conclusion"] == "UNKNOWN"


def test_recurrence_series_checker_has_no_sympy_or_producer_dependency() -> None:
    source = inspect.getsource(checker_module)
    assert "import sympy" not in source
    assert "domains.combinatorics" not in source
