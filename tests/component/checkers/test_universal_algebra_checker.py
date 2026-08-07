from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

import jacobian_checkers.universal_algebra as checker_module
from jacobian_checkers.universal_algebra import check_law_evaluation


def _variable(name: str) -> dict[str, object]:
    return {"kind": "VARIABLE", "variable": name, "left": None, "right": None}


def _product(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    return {"kind": "PRODUCT", "variable": None, "left": left, "right": right}


def _request() -> dict[str, Any]:
    bindings = {
        "claim_digest": "sha256:" + "a" * 64,
        "semantics_digest": "sha256:" + "b" * 64,
        "candidate_digest": "sha256:" + "c" * 64,
        "scope_digest": "sha256:" + "d" * 64,
        "encoding_digest": None,
    }
    problem_uri = "artifact://sha256/" + "1" * 64
    evaluation_uri = "artifact://sha256/" + "2" * 64
    x = _variable("x")
    y = _variable("y")
    return {
        "request_version": "1",
        "claim": {
            "payload": {
                "claim_schema_version": "1",
                "predicate": "EXACT_FINITE_MAGMA_LAW_EVALUATION",
                "problem_uri": problem_uri,
            }
        },
        "scope": {
            "artifact_uri": problem_uri,
            "payload": {
                "problem_schema_version": "1",
                "structure": {
                    "structure_schema_version": "1",
                    "operation": "binary",
                    "order": 2,
                    "table": [[0, 0], [1, 1]],
                },
                "laws": [
                    {
                        "law_id": "commutative",
                        "variables": ["x", "y"],
                        "left": _product(x, y),
                        "right": _product(y, x),
                    }
                ],
            },
        },
        "candidate": {
            "artifact_uri": evaluation_uri,
            "payload": {
                "evaluation_schema_version": "1",
                "problem_uri": problem_uri,
                "records": [
                    {
                        "law_id": "commutative",
                        "holds": False,
                        "coverage": "COUNTEREXAMPLE_FOUND",
                        "checked_valuations": 2,
                        "counterexample": {
                            "assignment": [
                                {"variable": "x", "value": 0},
                                {"variable": "y", "value": 1},
                            ],
                            "left_value": 0,
                            "right_value": 1,
                        },
                    }
                ],
                "arithmetic": "EXACT_FINITE",
                "determinism": "DETERMINISTIC",
            },
        },
        "certificate": {
            "payload": {
                "evidence_schema_version": "1",
                "certificate_type": "universal_algebra.law_evaluation",
                "format_version": "1",
                "bindings": deepcopy(bindings),
                "payload_digest": "sha256:" + "e" * 64,
                "payload": {
                    "method": "EXHAUSTIVE_LEXICOGRAPHIC_REPLAY",
                    "problem_uri": problem_uri,
                    "evaluation_uri": evaluation_uri,
                },
            }
        },
        "expected_bindings": deepcopy(bindings),
    }


def test_checker_replays_exact_law_evaluation() -> None:
    decision = check_law_evaluation(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_rejects_forged_counterexample_order() -> None:
    request = _request()
    record = request["candidate"]["payload"]["records"][0]
    record["checked_valuations"] = 3

    decision = check_law_evaluation(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    "broken_term",
    [
        ("VARIABLE", None, None, None),
        ("PRODUCT", None, None, ("VARIABLE", "x", None, None)),
        (),
        ("VARIABLE", "x", None, None, "extra"),
        ("PRODUCT", None, (), ("VARIABLE", "x", None, None)),
    ],
)
def test_checker_rejects_broken_parsed_term_invariants(
    monkeypatch: pytest.MonkeyPatch,
    broken_term: object,
) -> None:
    valid_term: checker_module.Term = ("VARIABLE", "x", None, None)
    law: checker_module.Law = (
        "commutative",
        ("x",),
        cast(checker_module.Term, broken_term),
        valid_term,
    )
    monkeypatch.setattr(
        checker_module,
        "_parse_problem",
        lambda _problem: (2, ((0, 0), (1, 1)), (law,)),
    )

    decision = check_law_evaluation(_request())

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
