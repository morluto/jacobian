"""Protocol and precedence tests for the shared metric-DAG worker."""

from __future__ import annotations

import time

import pytest

from jacobian.math.geometry.differential.metrics import _dag_process, _dag_worker


def _minimal_request() -> dict[str, object]:
    return {
        "variables": ["x"],
        "nodes": [
            {"operation": "ZERO", "arguments": []},
            {"operation": "ONE", "arguments": []},
        ],
        "fractions": [[1, 1]],
        "determinants": [0],
        "undefined_numerators": [0],
    }


def test_undefined_precedes_singular() -> None:
    assert _dag_worker._run(_minimal_request()) == {"status": "undefined"}


def test_zero_fraction_denominator_is_undefined() -> None:
    request = _minimal_request()
    request["fractions"] = [[1, 0]]
    request["undefined_numerators"] = []
    assert _dag_worker._run(request) == {"status": "undefined"}


def test_success_preserves_fraction_and_determinant_order() -> None:
    request = _minimal_request()
    request["determinants"] = [1, 1]
    request["undefined_numerators"] = []
    response = _dag_worker._run(request)
    assert response["status"] == "ok"
    assert response["fractions"] == [
        {"numerator": [[0, "1", "1"]], "denominator": [[0, "1", "1"]]}
    ]
    assert response["determinants"] == [
        [[0, "1", "1"]],
        [[0, "1", "1"]],
    ]


def test_parent_rejects_unbound_worker_response() -> None:
    with pytest.raises(RuntimeError, match="unbound result"):
        _dag_process._decode_response(
            {"protocol_version": 1, "request_digest": "wrong", "status": "ok"},
            expected_digest="expected",
            owner="test DAG",
            deadline=time.monotonic() + 1,
        )


def test_parent_rejects_unknown_protocol() -> None:
    with pytest.raises(RuntimeError, match="unknown protocol"):
        _dag_process._decode_response(
            {"protocol_version": 2, "request_digest": "expected", "status": "ok"},
            expected_digest="expected",
            owner="test DAG",
            deadline=time.monotonic() + 1,
        )
