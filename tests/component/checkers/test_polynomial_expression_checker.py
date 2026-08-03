from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import json_digest as _digest

from jacobian_checkers.polynomial_expressions import (
    check_polynomial_expression_normalization,
)


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


def _expression() -> dict[str, Any]:
    return {
        "kind": "multiply",
        "operands": [
            {
                "kind": "add",
                "operands": [
                    {"kind": "variable", "name": "x"},
                    {"kind": "variable", "name": "y"},
                ],
            },
            {
                "kind": "add",
                "operands": [
                    {"kind": "variable", "name": "x"},
                    {
                        "kind": "negate",
                        "operand": {"kind": "variable", "name": "y"},
                    },
                ],
            },
        ],
    }


def _request() -> dict[str, Any]:
    semantics_uri = _uri("e")
    source_payload = {
        "expression_schema_version": "1",
        "domain": "QQ",
        "variables": ["x", "y"],
        "expression": _expression(),
    }
    claim = _artifact(
        uri_character="a",
        object_character="1",
        schema_character="b",
        semantics_uri=semantics_uri,
        payload=source_payload,
        parents=[],
    )
    candidate_payload = {
        "normalization_schema_version": "1",
        "source": {
            "binding_version": "1",
            "expression_artifact_uri": claim["artifact_uri"],
            "expression_object_digest": claim["object_digest"],
            "expression_payload_digest": claim["payload_digest"],
            "variables": ["x", "y"],
            "node_count": 8,
            "depth": 4,
            "expanded_term_upper_bound": 4,
            "coefficient_digit_budget": 8,
        },
        "declared_scope": "FULL_EXPRESSION",
        "normalized": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [2, 0],
                },
                {
                    "coefficient": {"num": "-1", "den": "1"},
                    "exponents": [0, 2],
                },
            ]
        },
        "producer": {
            "runtime_version": "1",
            "provider": "jacobian.sympy",
            "availability": "AVAILABLE",
            "version": "1.14.0",
            "digest": "sha256:" + "9" * 64,
            "digest_kind": "PYTHON_DISTRIBUTION_RECORD",
            "platform": "linux-x86_64",
            "install_tier": "T0",
            "license_id": "BSD-3-Clause",
            "license_files": ["sympy-1.14.0.dist-info/LICENSE"],
            "features": [
                "typed-polynomial-expression",
                "exact-rational",
                "canonical-sparse-normalization",
            ],
            "checker_ids": [],
            "configuration": {
                "distribution": "sympy",
                "domain": "QQ",
                "operation": "Poly(expression, *variables, domain=QQ).terms()",
                "expression_schema_version": "1",
                "maximum_variables": 4,
                "maximum_nodes": 128,
                "maximum_depth": 16,
                "maximum_expanded_terms": 1024,
                "maximum_exponent_per_variable": 127,
                "maximum_coefficient_digit_budget": 4096,
            },
            "distribution_import_name": "sympy",
            "distribution_required_attributes": [
                "Add",
                "Mul",
                "Poly",
                "Pow",
                "QQ",
                "Rational",
                "Symbol",
            ],
            "diagnostic": None,
        },
        "resource_budget": {"budget_version": "1", "wall_seconds": 5},
        "method": "SYMPY_POLY_QQ_CANONICAL_TERMS",
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
        "witness_format": "polynomial.expression_normalization",
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "bindings": bindings,
        "payload": {
            "expression_uri": claim["artifact_uri"],
            "normalization_uri": candidate["artifact_uri"],
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


def test_checker_accepts_full_typed_ast_normalization() -> None:
    decision = check_polynomial_expression_normalization(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_RATIONAL"
    assert decision["relation_id"] == (
        "polynomial.relation.expression-normalization-of"
    )


def test_checker_rejects_wrong_coefficient_with_refreshed_digest() -> None:
    request = _request()
    request["candidate"]["payload"]["normalized"]["terms"][1]["coefficient"]["num"] = (
        "-2"
    )
    _refresh_payload_digest(request["candidate"])

    decision = check_polynomial_expression_normalization(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_noncanonical_output_order() -> None:
    request = _request()
    request["candidate"]["payload"]["normalized"]["terms"].reverse()
    _refresh_payload_digest(request["candidate"])

    decision = check_polynomial_expression_normalization(request)

    assert decision["accepted"] is False


def test_checker_rejects_mutated_source_even_when_payload_digest_is_refreshed() -> None:
    request = _request()
    request["claim"]["payload"]["expression"]["operands"][0]["operands"][0]["name"] = (
        "y"
    )
    _refresh_payload_digest(request["claim"])

    decision = check_polynomial_expression_normalization(request)

    assert decision["accepted"] is False


def test_checker_rejects_rebound_source_identity() -> None:
    request = _request()
    request["candidate"]["payload"]["source"]["expression_object_digest"] = (
        "sha256:" + "8" * 64
    )
    _refresh_payload_digest(request["candidate"])

    decision = check_polynomial_expression_normalization(request)

    assert decision["accepted"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            ("claim", "payload", "expression"),
            {"kind": "parsed_string", "value": "__import__('os').system('id')"},
        ),
        (
            ("candidate", "payload", "producer", "version"),
            "1.13.3",
        ),
        (
            ("candidate", "payload", "source", "expanded_term_upper_bound"),
            3,
        ),
        (
            ("witness", "payload", "payload", "normalization_uri"),
            _uri("7"),
        ),
    ],
    ids=("unsafe_node", "wrong_provider", "wrong_bound", "rebound_witness"),
)
def test_checker_rejects_malformed_or_misbound_evidence(
    path: tuple[str, ...],
    value: object,
) -> None:
    request = _request()
    target: Any = request
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] in {"claim", "candidate", "witness"}:
        _refresh_payload_digest(request[path[0]])

    decision = check_polynomial_expression_normalization(copy.deepcopy(request))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_noncanonical_rational_literal() -> None:
    request = _request()
    request["claim"]["payload"]["expression"] = {
        "kind": "rational",
        "value": {"num": "2", "den": "2"},
    }
    _refresh_payload_digest(request["claim"])

    decision = check_polynomial_expression_normalization(request)

    assert decision["accepted"] is False


def test_checker_rejects_missing_lineage_and_payload_digest_forgery() -> None:
    requests = []

    missing_parent = _request()
    missing_parent["candidate"]["parents"] = []
    requests.append(missing_parent)

    forged_digest = _request()
    forged_digest["candidate"]["payload_digest"] = "sha256:" + "0" * 64
    requests.append(forged_digest)

    duplicate_parent = _request()
    duplicate_parent["witness"]["parents"].append(
        duplicate_parent["claim"]["artifact_uri"]
    )
    requests.append(duplicate_parent)

    for request in requests:
        decision = check_polynomial_expression_normalization(request)
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
